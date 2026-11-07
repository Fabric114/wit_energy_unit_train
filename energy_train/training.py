"""YOLO 训练入口：配置与数据集校验、阶段参数组装、训练执行与 provenance。

本模块不得在导入时加载 ultralytics 或 torch；它们只在函数体内导入。
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .common import PROJECT_ROOT, project_path, sha256, valid_name

CONTROLLED_KEYS = frozenset({"data", "model", "project", "name", "exist_ok", "resume", "val"})
OVERRIDE_KEYS = ("batch", "epochs", "device", "workers")
TOP_LEVEL_KEYS = frozenset({"model", "train", "stages"})


class TrainingError(ValueError):
    """配置、数据集或运行前检查失败。"""


class DatasetCheckError(TrainingError):
    def __init__(self, issues):
        self.issues = list(issues)
        lines = [f"数据集校验发现 {len(self.issues)} 项问题："] + [f"- {item}" for item in self.issues]
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class Stage:
    name: str
    args: dict


@dataclass(frozen=True)
class TrainConfig:
    path: Path
    sha256: str
    content: dict
    model: Path
    model_sha256: str
    stages: tuple


def _check_controlled(mapping, where):
    for key in mapping:
        if key in CONTROLLED_KEYS:
            raise TrainingError(f"{where} 中不能设置 {key}，该参数由工具控制")


def _resolve_model(value):
    if not isinstance(value, str) or not value:
        raise TrainingError("model 必须是本地 .pt 文件路径")
    model = project_path(value).resolve()
    if model.suffix != ".pt":
        raise TrainingError(f"model 必须是 .pt 文件: {value}")
    if not model.is_file():
        raise TrainingError(
            f"起点权重不存在: {value}。请手动把官方权重放到 weights/pretrained/，工具不会自动下载"
        )
    return model


def _build_stages(content, shared):
    if "stages" not in content:
        return [Stage("train", dict(shared))]
    raw = content["stages"]
    if not isinstance(raw, list) or not raw:
        raise TrainingError("stages 必须是非空列表")
    stages, seen = [], set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict) or "name" not in item:
            raise TrainingError(f"第 {index} 个阶段必须是包含 name 的映射")
        name = item["name"]
        try:
            valid_name(name)
        except ValueError as exc:
            raise TrainingError(f"阶段名不合法: {name!r}，仅允许字母、数字、下划线和短横线") from exc
        if name in seen:
            raise TrainingError(f"阶段名重复: {name}")
        seen.add(name)
        overrides = {key: value for key, value in item.items() if key != "name"}
        _check_controlled(overrides, f"阶段 {name}")
        stages.append(Stage(name, {**shared, **overrides}))
    return stages


def load_config(path):
    path = project_path(path).resolve()
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainingError(f"无法读取训练配置: {path}") from exc
    if not isinstance(content, dict):
        raise TrainingError("训练配置顶层必须是映射")
    unknown = set(content) - TOP_LEVEL_KEYS
    if unknown:
        raise TrainingError(f"训练配置含未知顶层键: {sorted(unknown)}")
    if "model" not in content:
        raise TrainingError("训练配置缺少 model")
    train = content.get("train", {})
    if not isinstance(train, dict):
        raise TrainingError("train 必须是映射")
    _check_controlled(train, "train")
    model = _resolve_model(content["model"])
    stages = _build_stages(content, train)
    return TrainConfig(
        path=path,
        sha256=sha256(path),
        content=content,
        model=model,
        model_sha256=sha256(model),
        stages=tuple(stages),
    )


def _read_yaml_mapping(path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingError(f"无法解析 data.yaml: {path}") from exc
    if not isinstance(data, dict):
        raise TrainingError(f"data.yaml 顶层必须是映射: {path}")
    return data


def _count(directory, pattern):
    return len(list(directory.rglob(pattern))) if directory.is_dir() else 0


def validate_dataset(data_yaml, datasets_root=None):
    """校验 build_dataset 产出的数据集；问题集中收集后一次抛出 DatasetCheckError。"""
    data_path = project_path(data_yaml).resolve()
    if data_path.name != "data.yaml":
        raise TrainingError(f"--data 必须指向 data.yaml 文件，而不是目录或其他文件: {data_yaml}")
    if not data_path.is_file():
        raise TrainingError(f"data.yaml 不存在: {data_path}")
    root = (PROJECT_ROOT / "datasets") if datasets_root is None else project_path(datasets_root)
    root = root.resolve()
    dataset_dir = data_path.parent
    issues = []

    if dataset_dir.parent != root:
        issues.append(f"数据集必须位于数据集根目录 {root} 下的一级版本目录中，当前位于 {dataset_dir.parent}")

    data = _read_yaml_mapping(data_path)
    summary_path, manifest_path = dataset_dir / "summary.json", dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        issues.append("缺少 manifest.jsonl")
    summary = None
    if not summary_path.is_file():
        issues.append("缺少 summary.json")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            issues.append("summary.json 无法解析")
    if summary is not None:
        if summary.get("schema_version") != 2:
            issues.append(f"summary.json 的 schema_version 应为 2，当前为 {summary.get('schema_version')!r}")
        if summary.get("class_names") != {"0": "energy_unit"}:
            issues.append(f"summary.json 的 class_names 应为 {{'0': 'energy_unit'}}，当前为 {summary.get('class_names')!r}")
    split_images = summary.get("split_images") if summary is not None else None
    if not isinstance(split_images, dict):
        issues.append("summary.json 缺少 split_images")
        split_images = {}

    if data.get("names") not in ({0: "energy_unit"}, ["energy_unit"]):
        issues.append(f"data.yaml 的 names 应只包含 energy_unit，当前为 {data.get('names')!r}")
    if data.get("train") != "images/train":
        issues.append(f"data.yaml 的 train 应为 images/train，当前为 {data.get('train')!r}")
    if data.get("val") != "images/val":
        issues.append(f"data.yaml 的 val 应为 images/val，当前为 {data.get('val')!r}")
    if (dataset_dir / str(data.get("path"))).resolve() != dataset_dir.resolve():
        issues.append(f"data.yaml 的 path 与数据集目录不符（{data.get('path')!r}），请更新 path 或重新构建")

    has_test = "test" in split_images
    if ("test" in data) != has_test:
        issues.append("data.yaml 的 test 键与 summary.json 的 test 划分不一致")
    if "test" in data and data.get("test") != "images/test":
        issues.append(f"data.yaml 的 test 应为 images/test，当前为 {data.get('test')!r}")

    for split in ("train", "val"):
        count = split_images.get(split)
        if not isinstance(count, int) or count <= 0:
            issues.append(f"{split} 划分必须至少有 1 张图片，summary.json 记录为 {count!r}")
    for split, expected in split_images.items():
        if not isinstance(expected, int):
            continue
        images = _count(dataset_dir / "images" / split, "*.png")
        labels = _count(dataset_dir / "labels" / split, "*.txt")
        if images != expected:
            issues.append(f"{split} 划分的 PNG 数量为 {images}，summary.json 记录 {expected}")
        if labels != expected:
            issues.append(f"{split} 划分的标签 txt 数量为 {labels}，summary.json 记录 {expected}")

    if issues:
        raise DatasetCheckError(issues)
    return {
        "data_yaml": str(data_path),
        "dataset_dir": str(dataset_dir),
        "data_yaml_sha256": sha256(data_path),
        "summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
        "split_images": dict(split_images),
        "group_splits": dict(summary.get("group_splits", {})),
        "has_test": has_test,
    }


def default_run_name(config_path, dataset_dir, now=None):
    if now is None:
        now = datetime.now()
    return f"{Path(config_path).stem}_{Path(dataset_dir).name}_{now:%Y%m%d-%H%M%S}"


def plan_run(config, dataset, *, runs_root=None, name=None, overrides=None):
    """纯计算：给出 run 目录和每阶段参数，不创建任何目录。"""
    root = (PROJECT_ROOT / "runs") if runs_root is None else project_path(runs_root)
    root = root.resolve()
    if name is None:
        name = default_run_name(config.path, Path(dataset["dataset_dir"]))
    try:
        valid_name(name)
    except ValueError as exc:
        raise TrainingError(f"运行名不合法: {name!r}，仅允许字母、数字、下划线和短横线") from exc
    run_dir = root / name
    if run_dir.exists():
        raise TrainingError(f"运行目录已存在，请换一个 --name: {run_dir}")
    shared = {key: value for key, value in (overrides or {}).items()
              if key in OVERRIDE_KEYS and value is not None}
    stages, previous = [], None
    for stage in config.stages:
        model = str(config.model) if previous is None else str(run_dir / previous / "weights" / "best.pt")
        args = {**stage.args, **shared,
                "data": dataset["data_yaml"], "project": str(run_dir),
                "name": stage.name, "exist_ok": False}
        stages.append({"name": stage.name, "model": model, "args": args})
        previous = stage.name
    return {"run_name": name, "run_dir": str(run_dir), "stages": stages}
