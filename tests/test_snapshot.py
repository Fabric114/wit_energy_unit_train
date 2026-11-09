import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from energy_train.snapshot import SnapshotError, load_snapshot, verify_snapshot_unchanged
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

    def test_schema3_scene_and_group_mapping_must_agree_with_rows(self):
        rows = read_manifest(self.data.parent)
        rows[0]["split"] = "val" if rows[0]["split"] == "train" else "train"
        write_manifest(self.data.parent, rows)
        with self.assertRaises(SnapshotError):
            load_snapshot(self.data)

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


if __name__ == "__main__":
    unittest.main()
