import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from energy_train.augmentation import AugmentSample
from energy_train.augmentation_dataset import augment_dataset, prepare_augmentation
from energy_train.snapshot import SnapshotError, load_snapshot, verify_snapshot_unchanged
from energy_train.training import validate_dataset
from test_training import make_dataset


def read_manifest(dataset):
    return [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]


def write_manifest(dataset, rows):
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def downgrade_to_schema2(data_yaml):
    dataset = data_yaml.parent
    summary = json.loads((dataset / "summary.json").read_text())
    for key in ("algorithm", "group_definition", "split_config", "split_stats", "warnings", "excluded_images"):
        summary.pop(key, None)
    summary["schema_version"] = 2
    rows = read_manifest(dataset)
    for row in rows:
        for key in ("split_group_id", "scene", "tags"):
            row.pop(key, None)
    write_manifest(dataset, rows)
    (dataset / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))


def first_file(dataset, folder, suffix):
    return next(p for p in sorted((dataset / folder / "train").rglob(f"*{suffix}")))


def rewrite_label(dataset, text):
    first_file(dataset, "labels", ".txt").write_text(text, encoding="utf-8")


def tamper_cases():
    def label(text):
        return lambda d: rewrite_label(d, text)

    def manifest(change):
        def apply(dataset):
            rows = read_manifest(dataset)
            change(rows)
            write_manifest(dataset, rows)
        return apply

    def extra_png(d):
        shutil.copyfile(first_file(d, "images", ".png"), d / "images/train/s01/extra.png")

    def missing_png(d):
        first_file(d, "images", ".png").unlink()

    def changed_pixels(d):
        cv2.imwrite(str(first_file(d, "images", ".png")), np.zeros((80, 100, 3), np.uint8))

    return {
        "非 energy_unit 类别": label("1 0.5 0.5 0.4 0.4\n"),
        "NaN 坐标": label("0 nan 0.5 0.4 0.4\n"),
        "零宽度": label("0 0.5 0.5 0 0.4\n"),
        "框越界": label("0 0.99 0.5 0.4 0.4\n"),
        "清单外 PNG": extra_png,
        "缺少 PNG": missing_png,
        "图片像素改动": changed_pixels,
        "标签与清单框不一致": label("0 0.2 0.2 0.1 0.1\n"),
        "未知 split": manifest(lambda rows: rows[0].update(split="dev")),
        "绝对路径": manifest(lambda rows: rows[0].update(image="/etc/passwd")),
        "路径穿越": manifest(lambda rows: rows[0].update(image="images/train/../train/x.png")),
        "重复清单行": manifest(lambda rows: rows.append(dict(rows[0]))),
    }


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = make_dataset(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema3_loads_and_does_not_need_raw_after_build(self):
        index = load_snapshot(self.data)
        self.assertEqual(index.summary["schema_version"], 3)
        self.assertEqual({row["split_group_id"] for row in index.rows}, {"g1", "g2"})
        shutil.rmtree(self.root / "raw")
        self.assertEqual(load_snapshot(self.data).hashes, index.hashes)

    def test_schema2_snapshot_still_loads_with_group_id_as_group(self):
        downgrade_to_schema2(self.data)
        index = load_snapshot(self.data)
        self.assertEqual(index.summary["schema_version"], 2)
        self.assertNotIn("split_group_id", index.rows[0])

    def test_unknown_schema_versions_are_rejected(self):
        for value in (1, 999, True, "3"):
            with self.subTest(value=value):
                summary = self.data.parent / "summary.json"
                doc = json.loads(summary.read_text())
                doc["schema_version"] = value
                summary.write_text(json.dumps(doc))
                with self.assertRaises(SnapshotError):
                    load_snapshot(self.data)
                doc["schema_version"] = 3
                summary.write_text(json.dumps(doc))

    def test_tampering_with_labels_images_and_manifest_is_rejected(self):
        for label, mutate in tamper_cases().items():
            with self.subTest(case=label):
                tmp = tempfile.TemporaryDirectory()
                try:
                    data = make_dataset(Path(tmp.name))
                    mutate(data.parent)
                    with self.assertRaises(SnapshotError):
                        load_snapshot(data)
                finally:
                    tmp.cleanup()

    def test_scene_mismatch_inside_one_group_is_rejected(self):
        rows = read_manifest(self.data.parent)
        rows[0]["scene"] = "other_scene"
        write_manifest(self.data.parent, rows)
        with self.assertRaisesRegex(SnapshotError, "scene"):
            load_snapshot(self.data)

    def test_post_load_verification_detects_a_later_edit(self):
        index = load_snapshot(self.data)
        verify_snapshot_unchanged(index)
        first_file(self.data.parent, "labels", ".txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "发生变化"):
            verify_snapshot_unchanged(index)


def shift_augment(image_bgr, boxes_yolo, *, sample_seed):
    return AugmentSample(image_bgr + np.uint8(3), [list(b) for b in boxes_yolo],
                         [0] * len(boxes_yolo), list(range(len(boxes_yolo))))


def make_augmented(root):
    """基于 make_dataset 生成 schema 4 增强快照，返回其 data.yaml。"""
    data = make_dataset(root)
    out = Path(root) / "datasets" / "energy_v1_aug"
    augment_dataset(prepare_augmentation(data, copies=1), out, augment=shift_augment)
    return out / "data.yaml"


def read_summary(data):
    return json.loads((data.parent / "summary.json").read_text(encoding="utf-8"))


def write_summary(data, doc):
    (data.parent / "summary.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def augmented_row(rows):
    return next(r for r in rows if r.get("kind") == "augmentation")


def val_row(rows):
    return next(r for r in rows if r["split"] == "val")


class Schema4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = make_augmented(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema4_loads_with_split_counts(self):
        index = load_snapshot(self.data)
        self.assertEqual(index.summary["schema_version"], 4)
        self.assertEqual(index.summary["split_images"], {"train": 4, "val": 1})
        self.assertEqual(sum(1 for r in index.rows if r["kind"] == "augmentation"), 2)

    def test_schema4_remains_loadable_after_base_removed(self):
        shutil.rmtree(self.root / "datasets" / "energy_v1")
        shutil.rmtree(self.root / "raw")
        self.assertEqual(load_snapshot(self.data).summary["schema_version"], 4)

    def test_training_validation_accepts_schema4(self):
        info = validate_dataset(self.data, self.root / "datasets")
        self.assertEqual(info["schema_version"], 4)
        self.assertFalse(info["has_test"])

    def test_tampered_schema4_rows_are_rejected(self):
        cases = {
            "增强行 split 改为 val": lambda rows: augmented_row(rows).update(split="val"),
            "父图指向 val 图": lambda rows: augmented_row(rows).update(parent_image=val_row(rows)["image"]),
            "父图不存在": lambda rows: augmented_row(rows).update(parent_image="images/train/s09/none.png"),
            "父图像素哈希伪造": lambda rows: augmented_row(rows).update(parent_image_sha256="0" * 64),
            "父标签哈希伪造": lambda rows: augmented_row(rows).update(parent_label_sha256="0" * 64),
            "副本 scene 不一致": lambda rows: augmented_row(rows).update(scene="other"),
            "副本 group_id 不一致": lambda rows: augmented_row(rows).update(group_id="g2"),
            "副本 split_group_id 不一致": lambda rows: augmented_row(rows).update(split_group_id="g2"),
            "副本尺寸与父图不符": lambda rows: augmented_row(rows).update(width=99),
            "副本 status 非 generated": lambda rows: augmented_row(rows).update(status="passed"),
            "kind 非法": lambda rows: augmented_row(rows).update(kind="synthetic"),
            "原图行缺少 kind": lambda rows: next(r for r in rows if r["kind"] == "original").pop("kind"),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label):
                tmp = tempfile.TemporaryDirectory()
                try:
                    data = make_augmented(Path(tmp.name))
                    rows = read_manifest(data.parent)
                    mutate(rows)
                    write_manifest(data.parent, rows)
                    with self.assertRaises(SnapshotError):
                        load_snapshot(data)
                finally:
                    tmp.cleanup()

    def test_tampered_schema4_summary_is_rejected(self):
        def base_original_key(doc):
            return next(k for k in doc["base_dataset"]["files_sha256"]
                        if k.startswith("images/train/") and "__aug__" not in k)

        cases = {
            "缺少 dataset_kind": lambda doc: doc.pop("dataset_kind"),
            "缺少 base_dataset": lambda doc: doc.pop("base_dataset"),
            "instances 统计错误": lambda doc: doc.update(instances=doc["instances"] + 1),
            "generated 与行数不符": lambda doc: doc["augmentation"].update(generated=5),
            "原图 base 摘要不符": lambda doc: doc["base_dataset"]["files_sha256"].update(
                {base_original_key(doc): "0" * 64}),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label):
                tmp = tempfile.TemporaryDirectory()
                try:
                    data = make_augmented(Path(tmp.name))
                    doc = read_summary(data)
                    mutate(doc)
                    write_summary(data, doc)
                    with self.assertRaises(SnapshotError):
                        load_snapshot(data)
                finally:
                    tmp.cleanup()

    def test_augmentation_rows_are_rejected_without_schema4(self):
        plain = make_dataset(self.root / "plain")
        rows = read_manifest(plain.parent)
        rows[0]["kind"] = "augmentation"
        write_manifest(plain.parent, rows)
        with self.assertRaises(SnapshotError):
            load_snapshot(plain)


if __name__ == "__main__":
    unittest.main()
