import json
import tempfile
import unittest
from pathlib import Path

import yaml

from energy_train.common import PROJECT_ROOT, sha256
from energy_train.training import CONTROLLED_KEYS, TrainingError, load_config


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
