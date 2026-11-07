import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from energy_train.capture import CaptureSession
from energy_train.common import PROJECT_ROOT, sha256
from energy_train.dataset import build_dataset
from energy_train.training import (
    CONTROLLED_KEYS, DatasetCheckError, TrainingError, code_version, default_run_name,
    load_config, main, plan_run, run_training, validate_dataset,
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

    def test_cfg_key_cannot_smuggle_controlled_settings(self):
        with self.assertRaises(TrainingError):
            load_config(self.write({"model": str(self.weight), "train": {"cfg": "extra.yaml"}}))

    def test_values_that_cannot_be_recorded_are_rejected(self):
        path = self.root / "date.yaml"
        path.write_text(f"model: {self.weight}\ntrain:\n  note: 2026-11-07\n", encoding="utf-8")
        with self.assertRaises(TrainingError):
            load_config(path)
        path.write_text(f"model: {self.weight}\ntrain:\n  seed: .nan\n", encoding="utf-8")
        with self.assertRaises(TrainingError):
            load_config(path)

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


class PlanTests(unittest.TestCase):
    SINGLE = {"train": {"epochs": 150, "batch": 16}}
    TWO = {"train": {"batch": 16, "imgsz": 640, "degrees": 10, "patience": 30, "seed": 42},
           "stages": [{"name": "stage1_freeze", "epochs": 20, "freeze": 10},
                      {"name": "stage2_finetune", "epochs": 30, "optimizer": "AdamW", "lr0": 0.001}]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weight = self.root / "w.pt"
        self.weight.write_bytes(b"fake-weights")
        self.runs = self.root / "runs"
        self.dataset = validate_dataset(make_dataset(self.root), self.root / "datasets")
        self.cfg_single = load_config(self.write("single.yaml", self.SINGLE))
        self.cfg_two = load_config(self.write("two.yaml", self.TWO))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, template):
        content = {"model": str(self.weight), **template}
        path = self.root / name
        path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_default_run_name(self):
        name = default_run_name(Path("configs/train/yolo11n.yaml"), Path("datasets/energy_v1"),
                                datetime(2026, 11, 7, 20, 30, 0))
        self.assertEqual(name, "yolo11n_energy_v1_20261107-203000")

    def test_single_stage_args(self):
        plan = plan_run(self.cfg_single, self.dataset, runs_root=self.runs, name="r1", overrides={"batch": 8})
        stage = plan["stages"][0]
        self.assertEqual(plan["run_dir"], str((self.runs / "r1").resolve()))
        self.assertEqual(stage["name"], "train")
        self.assertEqual(stage["model"], str(self.cfg_single.model))
        self.assertEqual(stage["args"]["data"], self.dataset["data_yaml"])
        self.assertEqual(stage["args"]["project"], plan["run_dir"])
        self.assertEqual(stage["args"]["name"], "train")
        self.assertIs(stage["args"]["exist_ok"], False)
        self.assertEqual(stage["args"]["batch"], 8)
        self.assertEqual(stage["args"]["epochs"], 150)
        self.assertTrue(Path(stage["args"]["data"]).is_absolute())
        json.dumps(plan)

    def test_two_stage_chains_best_and_overrides_apply_to_all(self):
        plan = plan_run(self.cfg_two, self.dataset, runs_root=self.runs, name="r2", overrides={"epochs": 1})
        s1, s2 = plan["stages"]
        self.assertEqual(s1["model"], str(self.cfg_two.model))
        self.assertEqual(s2["model"], str(Path(plan["run_dir"]) / "stage1_freeze" / "weights" / "best.pt"))
        self.assertEqual((s1["args"]["epochs"], s2["args"]["epochs"]), (1, 1))
        self.assertEqual(s1["args"]["freeze"], 10)
        self.assertNotIn("freeze", s2["args"])
        self.assertEqual(s2["args"]["lr0"], 0.001)

    def test_existing_run_dir_rejected_and_nothing_created(self):
        (self.runs / "taken").mkdir(parents=True)
        with self.assertRaises(TrainingError):
            plan_run(self.cfg_single, self.dataset, runs_root=self.runs, name="taken")
        fresh = self.root / "fresh_runs"
        plan_run(self.cfg_single, self.dataset, runs_root=fresh, name="x")
        self.assertFalse(fresh.exists())

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            plan_run(self.cfg_single, self.dataset, runs_root=self.runs, name="../escape")


FAKE_ENV = {"python": "3.10.22", "platform": "linux", "hostname": "test", "ultralytics": "8.4.154",
            "torch": "2.14.0+cu130", "torchvision": "0.29.0+cu130", "cuda": "13.0", "cudnn": 91000,
            "gpus": [], "nvidia_driver": None}
FAKE_CODE = {"git_commit": None, "git_dirty": None, "files": {}}


class FakeYOLO:
    """替身：不导入 ultralytics，在 project/name 下写 weights/best.pt 与 last.pt。"""
    calls = []
    fail_on = None
    fail_with = RuntimeError
    save_suffix = ""
    skip_best = False
    seen_status = []

    def __init__(self, model):
        self.model = model

    def train(self, **kw):
        FakeYOLO.calls.append((self.model, kw))
        FakeYOLO.seen_status.append(
            json.loads((Path(kw["project"]) / "provenance.json").read_text(encoding="utf-8"))["status"])
        if kw["name"] == FakeYOLO.fail_on:
            raise FakeYOLO.fail_with("boom")
        save = Path(kw["project"]) / (kw["name"] + FakeYOLO.save_suffix)
        (save / "weights").mkdir(parents=True)
        best, last = save / "weights/best.pt", save / "weights/last.pt"
        if not FakeYOLO.skip_best:
            best.write_bytes(b"best-" + kw["name"].encode())
        last.write_bytes(b"last")
        self.trainer = SimpleNamespace(save_dir=save, best=best, last=last, metrics={
            "metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.7,
            "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.85, "bad": float("nan")})


class RunTests(unittest.TestCase):
    def setUp(self):
        FakeYOLO.calls = []
        FakeYOLO.fail_on = None
        FakeYOLO.fail_with = RuntimeError
        FakeYOLO.save_suffix = ""
        FakeYOLO.skip_best = False
        FakeYOLO.seen_status = []
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weight = self.root / "w.pt"
        self.weight.write_bytes(b"fake-weights")
        self.runs = self.root / "runs"
        self.dataset = validate_dataset(make_dataset(self.root), self.root / "datasets")
        self.cfg_single = load_config(self.write("single.yaml", PlanTests.SINGLE))
        self.cfg_two = load_config(self.write("two.yaml", PlanTests.TWO))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, template):
        content = {"model": str(self.weight), **template}
        path = self.root / name
        path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def run_it(self, cfg, *, name="run", overrides=None, command=None):
        plan = plan_run(cfg, self.dataset, runs_root=self.runs, name=name, overrides=overrides)
        return run_training(cfg, self.dataset, plan, overrides=overrides, command=command,
                            yolo_cls=FakeYOLO, env_info=lambda: dict(FAKE_ENV), code_info=lambda: FAKE_CODE)

    def saved_provenance(self, name="run"):
        return json.loads((self.runs / name / "provenance.json").read_text(encoding="utf-8"))

    def test_single_stage_completed_provenance(self):
        prov = self.run_it(self.cfg_single, overrides={"batch": 8}, command=["script/train.py"])
        saved = self.saved_provenance()
        self.assertEqual(saved, prov)
        self.assertEqual(prov["status"], "completed")
        self.assertEqual(FakeYOLO.seen_status, ["running"])
        self.assertEqual(prov["cli_overrides"], {"batch": 8})
        self.assertEqual(prov["pretrained"]["sha256"], self.cfg_single.model_sha256)
        self.assertEqual(prov["config"]["sha256"], self.cfg_single.sha256)
        self.assertEqual(prov["dataset"], self.dataset)
        self.assertEqual(prov["environment"]["ultralytics"], "8.4.154")
        self.assertEqual(prov["environment"]["code"], FAKE_CODE)
        stage = prov["stages"][0]
        self.assertEqual(stage["best_sha256"], sha256(stage["best"]))
        self.assertNotIn("bad", stage["metrics"])
        self.assertEqual(stage["metrics"]["metrics/mAP50(B)"], 0.9)
        self.assertEqual(prov["final_weights"], stage["best"])
        self.assertIsNotNone(prov["finished_at"])

    def test_two_stage_uses_stage1_best(self):
        prov = self.run_it(self.cfg_two)
        self.assertEqual(FakeYOLO.calls[1][0], prov["stages"][0]["best"])
        self.assertEqual(prov["final_weights"], prov["stages"][1]["best"])

    def test_stage1_failure_skips_stage2(self):
        FakeYOLO.fail_on = "stage1_freeze"
        with self.assertRaises(RuntimeError):
            self.run_it(self.cfg_two)
        self.assertEqual(len(FakeYOLO.calls), 1)
        self.assertEqual(self.saved_provenance()["status"], "failed")
        self.assertIn("boom", self.saved_provenance()["error"]["message"])

    def test_keyboard_interrupt_marks_interrupted(self):
        FakeYOLO.fail_on, FakeYOLO.fail_with = "train", KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_it(self.cfg_single)
        self.assertEqual(self.saved_provenance()["status"], "interrupted")

    def test_missing_best_stops_next_stage(self):
        FakeYOLO.skip_best = True
        with self.assertRaises(TrainingError):
            self.run_it(self.cfg_two)
        self.assertEqual(len(FakeYOLO.calls), 1)
        self.assertEqual(self.saved_provenance()["status"], "failed")

    def test_renamed_save_dir_rejected(self):
        FakeYOLO.save_suffix = "2"
        with self.assertRaises(TrainingError):
            self.run_it(self.cfg_single)

    def test_wrong_ultralytics_version_fails_before_training(self):
        env = dict(FAKE_ENV, ultralytics="8.3.0")
        plan = plan_run(self.cfg_single, self.dataset, runs_root=self.runs, name="v")
        with self.assertRaises(TrainingError):
            run_training(self.cfg_single, self.dataset, plan, yolo_cls=FakeYOLO,
                         env_info=lambda: env, code_info=lambda: FAKE_CODE)
        self.assertEqual(FakeYOLO.calls, [])
        self.assertEqual(self.saved_provenance("v")["status"], "failed")

    def test_run_dir_claimed_concurrently(self):
        plan = plan_run(self.cfg_single, self.dataset, runs_root=self.runs, name="race")
        Path(plan["run_dir"]).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            run_training(self.cfg_single, self.dataset, plan, yolo_cls=FakeYOLO,
                         env_info=lambda: dict(FAKE_ENV), code_info=lambda: FAKE_CODE)
        self.assertEqual(FakeYOLO.calls, [])

    def test_code_version_outside_git_and_in_empty_repo(self):
        if shutil.which("git") is None:
            self.skipTest("需要 git 命令")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "energy_train").mkdir()
            (root / "energy_train" / "x.py").write_text("x = 1\n", encoding="utf-8")
            info = code_version(root)
            self.assertIsNone(info["git_commit"])
            self.assertIsNone(info["git_dirty"])
            self.assertIn("energy_train/x.py", info["files"])
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            info = code_version(root)
            self.assertIsNone(info["git_commit"])
            self.assertIsInstance(info["git_dirty"], bool)
            self.assertIn("energy_train/x.py", info["files"])

    def test_import_does_not_load_ultralytics(self):
        out = subprocess.run([sys.executable, "-c", "import sys, energy_train.training; "
                              "print(any(m in sys.modules for m in ('ultralytics', 'torch')))"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weight = self.root / "w.pt"
        self.weight.write_bytes(b"fake-weights")
        self.data = make_dataset(self.root)
        self.datasets = self.root / "datasets"
        self.runs = self.root / "runs"
        self.cfg_path = self.root / "cfg.yaml"
        self.cfg_path.write_text(yaml.safe_dump({"model": str(self.weight), **PlanTests.SINGLE},
                                                allow_unicode=True, sort_keys=False), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def common_args(self):
        return ["--config", str(self.cfg_path), "--data", str(self.data),
                "--datasets-root", str(self.datasets), "--runs-root", str(self.runs)]

    def test_cli_check_only_from_other_cwd(self):
        tool = PROJECT_ROOT / "script" / "train.py"
        base = [sys.executable, str(tool), *self.common_args()]
        ok = subprocess.run(base + ["--check-only", "--name", "c1", "--batch", "8"],
                            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        out = json.loads(ok.stdout)
        self.assertEqual(out["stages"][0]["args"]["batch"], 8)
        self.assertFalse(self.runs.exists())
        (self.data.parent / "summary.json").unlink()
        bad = subprocess.run(base + ["--check-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("训练失败", bad.stderr)

    def test_main_maps_training_errors_to_exit_codes(self):
        with mock.patch("energy_train.training.run_training", side_effect=RuntimeError("boom")):
            self.assertEqual(main([*self.common_args(), "--name", "m1"]), 1)
        with mock.patch("energy_train.training.run_training", side_effect=KeyboardInterrupt):
            self.assertEqual(main([*self.common_args(), "--name", "m2"]), 130)


if __name__ == "__main__":
    unittest.main()
