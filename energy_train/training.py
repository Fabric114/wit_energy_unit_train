"""YOLO 训练入口：配置与数据集校验、阶段参数组装、训练执行与 provenance。

本模块不得在导入时加载 ultralytics 或 torch；它们只在函数体内导入。
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import project_path, sha256, valid_name

CONTROLLED_KEYS = frozenset({"data", "model", "project", "name", "exist_ok", "resume", "val"})
TOP_LEVEL_KEYS = frozenset({"model", "train", "stages"})


class TrainingError(ValueError):
    """配置、数据集或运行前检查失败。"""


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
