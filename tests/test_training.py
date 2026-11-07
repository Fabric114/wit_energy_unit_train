import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from energy_train.capture import CaptureSession
from energy_train.common import PROJECT_ROOT, sha256
from energy_train.dataset import build_dataset
from energy_train.training import (
    CONTROLLED_KEYS, DatasetCheckError, TrainingError, load_config, validate_dataset,
)


def make_dataset(root):
    """用 build_dataset 生成合格数据集：g1 训练、g2 验证。返回 data.yaml 路径。"""
    root = Path(root)
    raw = root / "raw"
    for session, group, values in (("s01", "g1", [10, 20]), ("s02", "g2", [30])):
        with CaptureSession(raw, session, group, {"width": 100, "height": 80, "serial": "test"}, {}) as capture:
            for value in values:
                path = capture.save(np.full((80, 100, 3), value, np.uint8), value, "manual", {})
                shapes = [] if value == 20 else [{"label": "energy_unit", "shape_type": "rectangle",
                    "group_id": None, "points": [[10, 20], [50, 20], [50, 60], [10, 60]]}]
                path.with_suffix(".json").write_text(json.dumps({"version": "4.3.1", "checked": True,
                    "imagePath": path.name, "imageWidth": 100, "imageHeight": 80, "shapes": shapes}))
    build_dataset(raw, root / "datasets" / "energy_v1", splits={"train": ["g1"], "val": ["g2"]})
    return root / "datasets" / "energy_v1" / "data.yaml"


def _edit_yaml(path, change):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _edit_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = make_dataset(self.root)
        self.datasets = self.root / "datasets"

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_dataset_returns_hashes(self):
        info = validate_dataset(self.data, self.datasets)
        summary = json.loads((self.data.parent / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(info["split_images"], summary["split_images"])
        self.assertEqual(info["summary_sha256"], sha256(self.data.parent / "summary.json"))
        self.assertEqual(info["manifest_sha256"], sha256(self.data.parent / "manifest.jsonl"))
        self.assertEqual(info["data_yaml_sha256"], sha256(self.data))
        self.assertFalse(info["has_test"])

    def test_ultralytics_cache_files_are_ignored(self):
        (self.data.parent / "labels" / "train.cache").write_bytes(b"x")
        validate_dataset(self.data, self.datasets)

    def test_directory_instead_of_data_yaml(self):
        with self.assertRaises(TrainingError) as ctx:
            validate_dataset(self.data.parent, self.datasets)
        self.assertIn("data.yaml", str(ctx.exception))

    def test_rejections(self):
        def fresh():
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            return root, make_dataset(root), root / "datasets"

        def move_out(root, data, datasets):
            other = root / "other"
            shutil.copytree(data.parent, other / "energy_v1")
            return other / "energy_v1" / "data.yaml"

        def drop(name):
            return lambda root, data, datasets: (data.parent / name).unlink() or data

        def set_schema(root, data, datasets):
            _edit_json(data.parent / "summary.json", lambda s: s.update(schema_version=1))
            return data

        def set_class_names(root, data, datasets):
            _edit_json(data.parent / "summary.json", lambda s: s.update(class_names={"0": "armor"}))
            return data

        def set_yaml_names(root, data, datasets):
            _edit_yaml(data, lambda d: d.update(names={0: "armor"}))
            return data

        def drop_yaml_val(root, data, datasets):
            _edit_yaml(data, lambda d: d.pop("val"))
            return data

        def zero_val_split(root, data, datasets):
            _edit_json(data.parent / "summary.json", lambda s: s["split_images"].update(val=0))
            return data

        def move_path(root, data, datasets):
            _edit_yaml(data, lambda d: d.update(path=str(root / "elsewhere")))
            return data

        def drop_val_png(root, data, datasets):
            next((data.parent / "images" / "val").rglob("*.png")).unlink()
            return data

        def drop_train_txt(root, data, datasets):
            next((data.parent / "labels" / "train").rglob("*.txt")).unlink()
            return data

        def add_test_key(root, data, datasets):
            _edit_yaml(data, lambda d: d.update(test="images/test"))
            return data

        cases = {
            "数据集根目录": move_out,
            "summary.json": drop("summary.json"),
            "manifest.jsonl": drop("manifest.jsonl"),
            "schema_version": set_schema,
            "class_names": set_class_names,
            "names": set_yaml_names,
            "val": drop_yaml_val,
            "path": move_path,
            "test": add_test_key,
        }
        for keyword, mutate in cases.items():
            with self.subTest(keyword=keyword):
                root, data, datasets = fresh()
                data = mutate(root, data, datasets)
                with self.assertRaises(DatasetCheckError) as ctx:
                    validate_dataset(data, datasets)
                self.assertTrue(any(keyword in issue for issue in ctx.exception.issues), ctx.exception.issues)
        for keyword, mutate in {"val": zero_val_split, "val_png": drop_val_png,
                                "train_txt": drop_train_txt}.items():
            with self.subTest(keyword=keyword):
                root, data, datasets = fresh()
                data = mutate(root, data, datasets)
                with self.assertRaises(DatasetCheckError) as ctx:
                    validate_dataset(data, datasets)
                self.assertTrue(any(("val" if keyword.startswith("val") else "train") in issue
                                    for issue in ctx.exception.issues), ctx.exception.issues)

    def test_reports_all_issues_together(self):
        _edit_yaml(self.data, lambda d: d.update(path=str(self.root / "elsewhere")))
        _edit_json(self.data.parent / "summary.json", lambda s: s.update(class_names={"0": "armor"}))
        with self.assertRaises(DatasetCheckError) as ctx:
            validate_dataset(self.data, self.datasets)
        self.assertGreaterEqual(len(ctx.exception.issues), 2)


if __name__ == "__main__":
    unittest.main()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weight = self.root / "w.pt"
        self.weight.write_bytes(b"fake-weights")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, content):
        path = self.root / "cfg.yaml"
        path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_single_stage_defaults_to_train(self):
        cfg = load_config(self.write({"model": str(self.weight), "train": {"epochs": 150, "batch": 16}}))
        self.assertEqual([s.name for s in cfg.stages], ["train"])
        self.assertEqual(cfg.stages[0].args, {"epochs": 150, "batch": 16})
        self.assertEqual(cfg.model, self.weight.resolve())
        self.assertEqual(cfg.model_sha256, sha256(self.weight))
        self.assertEqual(cfg.sha256, sha256(cfg.path))

    def test_stages_merge_over_shared_train(self):
        cfg = load_config(self.write({"model": str(self.weight), "train": {"batch": 16, "imgsz": 640},
            "stages": [{"name": "stage1_freeze", "epochs": 20, "freeze": 10},
                       {"name": "stage2_finetune", "epochs": 30, "optimizer": "AdamW", "lr0": 0.001}]}))
        self.assertEqual(cfg.stages[0].args, {"batch": 16, "imgsz": 640, "epochs": 20, "freeze": 10})
        self.assertEqual(cfg.stages[1].args, {"batch": 16, "imgsz": 640, "epochs": 30,
                                              "optimizer": "AdamW", "lr0": 0.001})

    def test_rejects_bad_configs(self):
        base = {"model": str(self.weight), "train": {"epochs": 1}}
        cases = {
            "unknown top-level key": {**base, "modle": "x"},
            "train is not a mapping": {"model": str(self.weight), "train": [1, 2]},
            "stages is empty": {**base, "stages": []},
            "stage missing name": {**base, "stages": [{"epochs": 1}]},
            "stage name invalid": {**base, "stages": [{"name": "bad name!", "epochs": 1}]},
            "stage name duplicated": {**base, "stages": [{"name": "a"}, {"name": "a"}]},
            "stage sets data": {**base, "stages": [{"name": "a", "data": "x.yaml"}]},
            "model missing file": {"model": str(self.root / "missing.pt"), "train": {}},
            "model not .pt": {"model": str(self.root / "w.onnx"), "train": {}},
        }
        for key in sorted(CONTROLLED_KEYS):
            cases[f"train sets controlled key {key}"] = {"model": str(self.weight), "train": {key: 1}}
        (self.root / "w.onnx").write_bytes(b"x")
        for label, content in cases.items():
            with self.subTest(label):
                with self.assertRaises(TrainingError):
                    load_config(self.write(content))

    def test_bare_model_name_is_not_downloaded(self):
        with self.assertRaises(TrainingError) as ctx:
            load_config(self.write({"model": "yolo11n_not_here.pt", "train": {}}))
        self.assertIn("weights/pretrained", str(ctx.exception))

    def test_repo_configs_match_spec(self):
        one = yaml.safe_load((PROJECT_ROOT / "configs/train/yolo11n.yaml").read_text(encoding="utf-8"))
        self.assertEqual(one, {"model": "weights/pretrained/yolo11n.pt", "train":
            {"epochs": 150, "batch": 16, "imgsz": 640, "degrees": 10, "patience": 30, "seed": 42}})
        two = yaml.safe_load((PROJECT_ROOT / "configs/train/yolo11n_two_stage.yaml").read_text(encoding="utf-8"))
        self.assertEqual([s["name"] for s in two["stages"]], ["stage1_freeze", "stage2_finetune"])
        self.assertEqual(two["stages"][1]["optimizer"], "AdamW")


if __name__ == "__main__":
    unittest.main()
