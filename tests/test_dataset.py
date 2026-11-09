import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from energy_train.capture import CaptureSession
from energy_train.common import sha256
from energy_train.dataset import inspect_dataset, assign_groups, build_dataset, prepare_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "dataset"
        self.images = []
        for session, group, values in (("s01", "g1", [10, 20]), ("s02", "g2", [30]), ("s03", "g1", [40])):
            with CaptureSession(self.raw, session, group, {"width": 100, "height": 80, "serial": "test"}, {}) as capture:
                for value in values:
                    path = capture.save(np.full((80, 100, 3), value, np.uint8), value, "manual", {})
                    shapes = [] if value == 20 else [{"label": "energy_unit", "shape_type": "rectangle",
                        "group_id": None, "points": [[10, 20], [50, 20], [50, 60], [10, 60]]}]
                    path.with_suffix(".json").write_text(json.dumps({"version": "4.3.1", "checked": True,
                        "imagePath": path.name, "imageWidth": 100, "imageHeight": 80, "shapes": shapes}))
                    self.images.append(path)
        self.label = self.images[0].with_suffix(".json")

    def tearDown(self):
        self.tmp.cleanup()

    def source_bytes(self):
        return {p.relative_to(self.raw): p.read_bytes() for p in self.raw.rglob("*") if p.is_file()}

    def test_scan_uses_native_json_ignores_deleted_and_stale_templates(self):
        trash = self.raw / "_delete_"; trash.mkdir()
        shutil.copyfile(self.images[0], trash / "discarded.png")
        nested = self.raw / "s01" / "_delete_"; nested.mkdir()
        shutil.copyfile(self.images[0], nested / "discarded.png")
        old = self.root / "annotations" / "s01"; old.mkdir(parents=True)
        (old / self.label.name).write_text('{"shapes": []}')
        before = self.source_bytes()
        report = inspect_dataset(self.raw)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["summary"]["images"], 4)
        self.assertEqual(report["summary"]["groups"], 2)
        self.assertEqual(report["summary"]["positive"], 3)
        self.assertEqual(report["summary"]["negative"], 1)
        self.assertEqual(before, self.source_bytes())

    def test_pending_failed_missing_and_invalid_never_become_negative(self):
        original = self.label.read_text()
        for value, kind in ((False, "pending"), ("failed", "failed"), (1, "invalid"), ("missing", "missing")):
            with self.subTest(value=value):
                doc = json.loads(original)
                if value == "missing":
                    self.label.unlink()
                else:
                    doc["checked"] = value
                    self.label.write_text(json.dumps(doc))
                report = inspect_dataset(self.raw)
                self.assertTrue(any(issue["kind"] == kind for issue in report["issues"]))
                self.assertEqual(report["summary"]["negative"], 1)
                with self.assertRaises(ValueError):
                    build_dataset(self.raw, self.output)
                self.assertFalse(self.output.exists())
                self.label.write_text(original)

    def test_build_one_snapshot_with_group_split_and_correct_yolo_labels(self):
        before = self.source_bytes()
        result = build_dataset(self.raw, self.output, val_ratio=float("nan"),
                               splits={"train": ["g1"], "val": ["g2"]})
        self.assertIsNone(result["val_ratio"])
        self.assertIsNone(result["test_ratio"])
        self.assertEqual(result["images"], 4)
        data = yaml.safe_load((self.output / "data.yaml").read_text())
        self.assertEqual(data["names"], {0: "energy_unit"})
        self.assertEqual(Path(data["path"]), self.output)
        for key in ("train", "val"):
            self.assertTrue((self.output / data[key]).is_dir())
        rows = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual({r["split"] for r in rows if r["group_id"] == "g1"}, {"train"})
        self.assertEqual({r["split"] for r in rows if r["group_id"] == "g2"}, {"val"})
        positive = next(r for r in rows if r["source_image"] == str(self.images[0]))
        values = list(map(float, (self.output / positive["label"]).read_text().split()))
        np.testing.assert_allclose(values, [0, .3, .5, .4, .5])
        negative = next(r for r in rows if r["source_image"] == str(self.images[1]))
        self.assertEqual((self.output / negative["label"]).read_text(), "")
        for row in rows:
            self.assertEqual(sha256(self.output / row["image"]), row["image_sha256"])
            self.assertEqual(sha256(self.output / row["label"]), row["label_sha256"])
        self.assertEqual(before, self.source_bytes())
        frozen = (self.output / positive["image"]).read_bytes()
        self.images[0].write_bytes(b"source edited later")
        self.assertEqual((self.output / positive["image"]).read_bytes(), frozen)
        with self.assertRaises(FileExistsError):
            build_dataset(self.raw, self.output)

    def test_same_group_across_sessions_cannot_fill_two_splits(self):
        with self.assertRaisesRegex(ValueError, "至少.*2.*组"):
            build_dataset(self.raw, self.output, sessions=["s01", "s03"])
        self.assertFalse(self.output.exists())

    def test_group_assignment_reproducible_with_test_partition(self):
        groups = [f"g{i}" for i in range(10)]
        assigned = assign_groups(groups, val_ratio=.2, test_ratio=.1, seed=7)
        self.assertEqual(assigned, assign_groups(list(reversed(groups)), val_ratio=.2, test_ratio=.1, seed=7))
        self.assertEqual([list(assigned.values()).count(s) for s in ("train", "val", "test")], [7, 2, 1])
        with self.assertRaises(ValueError):
            assign_groups(["g1", "g2"], test_ratio=.1)

    def test_invalid_ratios_and_fixed_group_assignments_rejected(self):
        for val, test in ((0, 0), (1, 0), (.5, .5), (.2, -1), (float("nan"), 0), (.2, float("inf"))):
            with self.subTest(val=val, test=test), self.assertRaises(ValueError):
                assign_groups(["g1", "g2"], val_ratio=val, test_ratio=test)
        for splits in ({"train": ["g1"], "val": ["g1", "g2"]},
                       {"train": ["g1"], "val": []},
                       {"train": ["g1"], "val": ["g3"]},
                       {"train": "g1", "val": ["g2"]},
                       {"train": ["g1"], "val": ["g2"], "typo": []}):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                assign_groups(["g1", "g2"], splits=splits)

    def test_clipping_is_recorded_and_does_not_modify_source(self):
        doc = json.loads(self.label.read_text())
        doc["shapes"][0]["points"] = [[-.7, 20], [50, 20], [50, 60], [-.7, 60]]
        self.label.write_text(json.dumps(doc)); before = self.label.read_bytes()
        result = build_dataset(self.raw, self.output)
        self.assertEqual(result["clipped_boxes"], 1)
        self.assertEqual(self.label.read_bytes(), before)

    def test_source_changes_during_copy_never_publish(self):
        baseline = self.source_bytes()
        real_copy = shutil.copyfile
        for kind in ("json", "image", "meta", "new_image", "removed_image"):
            with self.subTest(kind=kind):
                changed = False

                def copy_then_change(source, destination):
                    nonlocal changed
                    result = real_copy(source, destination)
                    if not changed:
                        changed = True
                        if kind == "json":
                            self.label.write_text(self.label.read_text() + "\n")
                        elif kind == "image":
                            cv2.imwrite(str(self.images[0]), np.zeros((80, 100, 3), np.uint8))
                        elif kind == "meta":
                            meta = self.raw / "s01" / "meta.json"
                            doc = json.loads(meta.read_text()); doc["group_id"] = "new_group"; meta.write_text(json.dumps(doc))
                        elif kind == "new_image":
                            real_copy(self.images[0], self.raw / "s01" / "s01_99999.png")
                        else:
                            self.images[-1].unlink()
                    return result

                with patch("energy_train.dataset.shutil.copyfile", side_effect=copy_then_change):
                    with self.assertRaises((ValueError, OSError)):
                        build_dataset(self.raw, self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".dataset-*/")), [])
                for p in self.raw.rglob("*"):
                    if p.is_file() and p.relative_to(self.raw) not in baseline:
                        p.unlink()
                for name, payload in baseline.items():
                    (self.raw / name).write_bytes(payload)

    def test_copy_failure_and_source_output_overlap_leave_no_dataset(self):
        with patch("energy_train.dataset.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                build_dataset(self.raw, self.output)
        self.assertFalse(self.output.exists())
        for output in (self.raw / "dataset", self.raw, self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                build_dataset(self.raw, output)

    def test_cli_check_then_build_from_another_working_directory(self):
        tool = Path(__file__).resolve().parents[1] / "script" / "build_dataset.py"
        base = [sys.executable, str(tool), "--raw-root", str(self.raw)]
        before = self.source_bytes()
        checked = subprocess.run(base + ["--check-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)["summary"]["passed"], 4)
        self.assertEqual(self.source_bytes(), before)
        split_file = self.root / "splits.yaml"
        split_file.write_text("train: [g1]\nval: [g2]\n")
        built = subprocess.run(base + ["--output", str(self.output), "--split-file", str(split_file)],
                               cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertTrue((self.output / "data.yaml").exists())
        doc = json.loads(self.label.read_text()); doc["checked"] = False; self.label.write_text(json.dumps(doc))
        pending = subprocess.run(base + ["--check-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(pending.returncode, 2)
        self.assertEqual(json.loads(pending.stdout)["summary"]["pending"], 1)

    def group_text(self, confirmed=True, exclude=""):
        a, b, c, d = self.ids()
        return (f"schema_version: 1\nconfirmed: {str(confirmed).lower()}\ngroups:\n"
                f"  - {{id: first, reason: 第一段, images: [{a}, {b}]}}\n"
                f"  - {{id: second, reason: 第二段, images: [{c}, {d}]}}\n" + exclude)

    def ids(self):
        return [f"{p.parent.name}/{p.name}" for p in self.images]

    def group_file(self, text):
        path = self.root / "groups.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def manifest(self, output):
        return [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]

    def test_publish_is_schema3_with_algorithm_stats_and_reusable_splits_file(self):
        result = build_dataset(self.raw, self.output)
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["algorithm"], "shuffled_groups_v1")
        self.assertEqual(result["split_method"], "shuffled_groups_v1")
        self.assertEqual(set(result["group_splits"]), {"g1", "g2"})
        self.assertEqual(result["group_definition"]["mode"], "meta")
        self.assertIn("split_stats", result)
        self.assertIsInstance(result["warnings"], list)
        saved = yaml.safe_load((self.output / "splits.yaml").read_text())
        self.assertEqual({g: s for s, groups in saved.items() for g in groups}, result["group_splits"])
        self.assertEqual(prepare_dataset(self.raw).assignment, result["group_splits"])

    def test_group_file_splits_one_session_and_joins_groups_across_sessions(self):
        a, b, c, d = self.ids()
        path = self.group_file(self.group_text())
        result = build_dataset(self.raw, self.output, group_file=path)
        rows = {Path(r["source_image"]).name: r for r in self.manifest(self.output)}
        self.assertEqual(rows[Path(b).name]["split_group_id"], "first")
        self.assertEqual(rows[Path(c).name]["split_group_id"], "second")
        self.assertEqual(rows[Path(b).name]["group_id"], "g1")
        self.assertEqual(rows[Path(c).name]["group_id"], "g2")
        self.assertEqual(rows[Path(c).name]["scene"], "second")
        self.assertEqual(result["group_definition"]["confirmed"], True)
        for name in {r["split_group_id"] for r in rows.values()}:
            self.assertEqual(len({r["split"] for r in rows.values() if r["split_group_id"] == name}), 1)

    def test_excluded_image_leaves_snapshot_but_still_needs_passed_review(self):
        a, b, c, d = self.ids()
        exclude = f"exclude:\n  - {{reason: 边界模糊, images: [{b}]}}\n"
        text = (f"schema_version: 1\nconfirmed: true\ngroups:\n"
                f"  - {{id: first, reason: 第一段, images: [{a}]}}\n"
                f"  - {{id: second, reason: 第二段, images: [{c}, {d}]}}\n" + exclude)
        result = build_dataset(self.raw, self.output, group_file=self.group_file(text))
        self.assertEqual(result["images"], 4)
        self.assertEqual(result["excluded_images"], 1)
        self.assertEqual(sum(result["split_images"].values()), 3)
        self.assertNotIn(Path(b).name, {Path(r["source_image"]).name for r in self.manifest(self.output)})
        self.assertEqual(result["group_definition"]["exclude"][0]["image"], b)
        doc = json.loads(self.images[1].with_suffix(".json").read_text())
        doc["checked"] = "failed"
        self.images[1].with_suffix(".json").write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            build_dataset(self.raw, self.output.parent / "again", group_file=self.group_file(text))

    def test_unconfirmed_group_file_is_reported_and_never_built(self):
        path = self.group_file(self.group_text(confirmed=False))
        prepared = prepare_dataset(self.raw, group_file=path)
        self.assertIn("group_file_unconfirmed", [w["code"] for w in prepared.warnings])
        self.assertFalse(prepared.group_definition["confirmed"])
        with self.assertRaisesRegex(ValueError, "confirmed"):
            build_dataset(self.raw, self.output, group_file=path)
        self.assertFalse(self.output.exists())

    def test_group_file_changed_during_copy_never_publishes(self):
        text = self.group_text()
        path = self.group_file(text)
        real_copy = shutil.copyfile

        def copy_then_edit(source, destination):
            result = real_copy(source, destination)
            path.write_text(text + "# edited\n", encoding="utf-8")
            return result

        with patch("energy_train.dataset.shutil.copyfile", side_effect=copy_then_edit):
            with self.assertRaisesRegex(ValueError, "被修改"):
                build_dataset(self.raw, self.output, group_file=path)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".dataset-*/")), [])

    def test_split_file_from_publish_reproduces_the_same_assignment(self):
        build_dataset(self.raw, self.output)
        saved = yaml.safe_load((self.output / "splits.yaml").read_text())
        again = self.root / "again"
        result = build_dataset(self.raw, again, split_file=self.output / "splits.yaml")
        self.assertEqual(result["algorithm"], "fixed")
        self.assertEqual(result["group_splits"], {g: s for s, groups in saved.items() for g in groups})

    def test_cli_group_file_check_exit_codes_from_another_directory(self):
        tool = Path(__file__).resolve().parents[1] / "script" / "build_dataset.py"
        path = self.group_file(self.group_text(confirmed=False))
        command = [sys.executable, str(tool), "--raw-root", str(self.raw), "--group-file", str(path)]
        pending = subprocess.run(command + ["--check-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(pending.returncode, 2)
        self.assertFalse(self.output.exists())
        path.write_text(self.group_text(confirmed=True), encoding="utf-8")
        passed = subprocess.run(command + ["--check-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertEqual(json.loads(passed.stdout)["group_splits"].keys(), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
