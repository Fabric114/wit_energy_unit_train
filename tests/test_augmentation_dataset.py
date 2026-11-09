import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from energy_train.augmentation import AugmentSample, CandidateRejected, augment_one
from energy_train.augmentation_dataset import (
    AugmentationError, augment_dataset, main, prepare_augmentation,
)
from energy_train.common import PROJECT_ROOT, sha256
from test_snapshot import downgrade_to_schema2
from test_training import make_dataset


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def shift_augment(image_bgr, boxes_yolo, *, sample_seed):
    """确定性替身：像素 +3、框保持不变，保证图像确实改变；只用于验证发布逻辑。"""
    image = image_bgr + np.uint8(3)
    return AugmentSample(image, [list(b) for b in boxes_yolo],
                         [0] * len(boxes_yolo), list(range(len(boxes_yolo))))


class AugmentDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = make_dataset(self.root)  # train: 2 张（正/负各 1），val: 1 张
        self.base = self.data.parent
        self.out = self.root / "datasets" / "energy_v1_aug"

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot_bytes(self):
        return {p.relative_to(self.base).as_posix(): p.read_bytes()
                for p in sorted(self.base.rglob("*")) if p.is_file()}

    def staging_leftovers(self):
        return [p.name for p in self.out.parent.iterdir() if p.name.startswith(f".{self.out.name}-")]

    def train_rows(self, plan):
        return [row for row in plan.index.rows if row["split"] == "train"]

    def test_happy_path_publishes_complete_snapshot(self):
        before = self.snapshot_bytes()
        plan = prepare_augmentation(self.data, copies=2, seed=42)
        self.assertEqual(plan.expected_copies, 4)
        summary = augment_dataset(plan, self.out, augment=shift_augment)

        self.assertEqual(summary["split_images"], {"train": 6, "val": 1})
        self.assertEqual(summary["schema_version"], 4)
        self.assertEqual(summary["dataset_kind"], "augmented")
        self.assertEqual(summary["augmentation"]["recipe"], "light_v1")
        self.assertEqual(summary["augmentation"]["generated"], 4)
        self.assertEqual(summary["augmentation"]["skipped"], 0)
        self.assertEqual(before, self.snapshot_bytes())  # 输入快照未被改写

        rows = read_jsonl(self.out / "manifest.jsonl")
        originals = {r["image"]: r for r in rows if r["kind"] == "original"}
        augmented = [r for r in rows if r["kind"] == "augmentation"]
        self.assertEqual(len(augmented), 4)
        self.assertTrue(all(r["split"] == "train" for r in augmented))
        for row in augmented:
            parent = originals[row["parent_image"]]
            self.assertEqual(row["scene"], parent["scene"])
            self.assertEqual(row["group_id"], parent["group_id"])
            self.assertEqual(row["split_group_id"], parent["split_group_id"])
            self.assertEqual(row["parent_image_sha256"], parent["image_sha256"])
            self.assertEqual(row["parent_label_sha256"], parent["label_sha256"])
            self.assertNotIn("applied_transforms", row)
            self.assertEqual(sha256(self.out / row["image"]), row["image_sha256"])
            self.assertEqual(sha256(self.out / row["label"]), row["label_sha256"])
        negatives = [r for r in augmented if r["instances"] == 0]
        self.assertEqual(len(negatives), 2)
        for row in negatives:
            self.assertEqual((self.out / row["label"]).read_bytes(), b"")
        self.assertFalse((self.out / "preview").exists())

        for row in rows:
            if row["split"] != "train":
                self.assertEqual((self.out / row["image"]).read_bytes(), (self.base / row["image"]).read_bytes())
                self.assertEqual((self.out / row["label"]).read_bytes(), (self.base / row["label"]).read_bytes())

    def test_schema2_input_gets_group_and_scene_fields(self):
        downgrade_to_schema2(self.data)
        plan = prepare_augmentation(self.data, copies=1)
        augment_dataset(plan, self.out, augment=shift_augment)
        for row in read_jsonl(self.out / "manifest.jsonl"):
            self.assertEqual(row["split_group_id"], row["group_id"])
            self.assertEqual(row["scene"], row["group_id"])

    def test_real_light_v1_is_reproducible_and_counts_add_up(self):
        plan = prepare_augmentation(self.data, copies=1, seed=7)
        first = augment_dataset(plan, self.out)
        second_out = self.root / "datasets" / "energy_v1_aug2"
        second = augment_dataset(plan, second_out)
        self.assertEqual(first["augmentation"]["generated"] + first["augmentation"]["skipped"], plan.expected_copies)
        hashes_first = [r["image_sha256"] for r in read_jsonl(self.out / "manifest.jsonl") if r["kind"] == "augmentation"]
        hashes_second = [r["image_sha256"] for r in read_jsonl(second_out / "manifest.jsonl") if r["kind"] == "augmentation"]
        self.assertEqual(hashes_first, hashes_second)
        self.assertEqual(second["augmentation"]["generated"], first["augmentation"]["generated"])
        for row in read_jsonl(self.out / "manifest.jsonl"):
            if row["kind"] == "augmentation":
                self.assertTrue((self.out / row["image"]).is_file())
                self.assertTrue((self.out / row["label"]).is_file())

    def test_failed_copies_are_skipped_individually(self):
        def reject_boxed(image, boxes, *, sample_seed):
            if boxes:
                raise CandidateRejected("dropped_box")
            return shift_augment(image, boxes, sample_seed=sample_seed)

        plan = prepare_augmentation(self.data, copies=1)
        summary = augment_dataset(plan, self.out, augment=reject_boxed)
        self.assertEqual(summary["augmentation"]["generated"], 1)
        self.assertEqual(summary["augmentation"]["skipped"], 1)
        item = summary["augmentation"]["skipped_items"][0]
        self.assertEqual(item["reasons"], ["dropped_box"] * 10)

    def test_all_copies_failing_publishes_nothing(self):
        def always_reject(image, boxes, *, sample_seed):
            raise CandidateRejected("unchanged_image")

        before = self.snapshot_bytes()
        plan = prepare_augmentation(self.data, copies=1)
        with self.assertRaises(AugmentationError):
            augment_dataset(plan, self.out, augment=always_reject)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.staging_leftovers(), [])
        self.assertEqual(before, self.snapshot_bytes())

    def test_candidate_equal_to_val_image_is_retried(self):
        plan = prepare_augmentation(self.data, copies=1)
        val_row = next(r for r in plan.index.rows if r["split"] == "val")
        val_file = self.base / val_row["image"]
        val_image = cv2.imread(str(val_file), cv2.IMREAD_COLOR)
        self.assertEqual(cv2.imencode(".png", val_image)[1].tobytes(), val_file.read_bytes(),
                         "前提：同像素 PNG 的编码字节必须与快照文件一致")
        calls = []

        def collide_first(image, boxes, *, sample_seed):
            calls.append(sample_seed)
            if len(calls) == 1:
                return AugmentSample(val_image.copy(), [list(b) for b in boxes],
                                     [0] * len(boxes), list(range(len(boxes))))
            return shift_augment(image, boxes, sample_seed=sample_seed)

        augment_dataset(plan, self.out, augment=collide_first)
        rows = read_jsonl(self.out / "manifest.jsonl")
        attempts = sorted(r["attempt"] for r in rows if r["kind"] == "augmentation")
        self.assertEqual(attempts, [0, 1])
        for row in rows:
            if row["kind"] == "augmentation":
                self.assertNotEqual(row["image_sha256"], val_row["image_sha256"])

    def test_input_mutation_during_augmentation_aborts(self):
        plan = prepare_augmentation(self.data, copies=1)
        target = self.base / self.train_rows(plan)[0]["image"]

        def mutate(image, boxes, *, sample_seed):
            target.write_bytes(cv2.imencode(".png", np.full((80, 100, 3), 77, np.uint8))[1].tobytes())
            return shift_augment(image, boxes, sample_seed=sample_seed)

        with self.assertRaises(ValueError):
            augment_dataset(plan, self.out, augment=mutate)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.staging_leftovers(), [])

    def test_copy_failure_cleans_staging(self):
        real_copy = shutil.copyfile
        calls = {"n": 0}

        def flaky(src, dst, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("磁盘已满（模拟）")
            return real_copy(src, dst, *args, **kwargs)

        plan = prepare_augmentation(self.data, copies=1)
        with mock.patch("energy_train.augmentation_dataset.shutil.copyfile", side_effect=flaky):
            with self.assertRaises(OSError):
                augment_dataset(plan, self.out, augment=shift_augment)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.staging_leftovers(), [])

    def test_png_encode_failure_aborts(self):
        plan = prepare_augmentation(self.data, copies=1)
        with mock.patch("energy_train.augmentation_dataset.cv2.imencode", return_value=(False, None)):
            with self.assertRaises(AugmentationError):
                augment_dataset(plan, self.out, augment=shift_augment)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.staging_leftovers(), [])

    def test_existing_output_is_rejected(self):
        self.out.mkdir(parents=True)
        plan = prepare_augmentation(self.data, copies=1)
        with self.assertRaises(FileExistsError):
            augment_dataset(plan, self.out, augment=shift_augment)

    def test_output_overlapping_input_is_rejected(self):
        plan = prepare_augmentation(self.data, copies=1)
        with self.assertRaises(ValueError):
            augment_dataset(plan, self.base / "inside", augment=shift_augment)
        with self.assertRaises(ValueError):
            augment_dataset(plan, self.root / "datasets", augment=shift_augment)

    def test_schema4_input_is_rejected(self):
        summary_path = self.base / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["schema_version"] = 4
        summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(ValueError):
            prepare_augmentation(self.data, copies=1)

    def test_reserved_aug_directory_in_input_is_rejected(self):
        (self.base / "images" / "train" / "__aug__").mkdir(parents=True)
        with self.assertRaises(AugmentationError):
            prepare_augmentation(self.data, copies=1)

    def test_invalid_copies_and_seed_are_rejected(self):
        for bad in (0, -1, True, "1", 10000):
            with self.subTest(copies=bad), self.assertRaises(AugmentationError):
                prepare_augmentation(self.data, copies=bad)
        with self.assertRaises(AugmentationError):
            prepare_augmentation(self.data, copies=1, seed=True)

    def test_cli_check_only_creates_nothing_from_any_cwd(self):
        script = PROJECT_ROOT / "script" / "augment_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--data", str(self.data), "--copies", "1", "--check-only"],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planned_generated", result.stdout)
        self.assertFalse(self.out.exists())

    def test_cli_bad_input_returns_2(self):
        with mock.patch("sys.stderr"):
            code = main(["--data", str(self.root / "missing" / "data.yaml"), "--check-only"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
