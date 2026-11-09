"""已发布数据集快照的只读校验，训练与离线增强共用；不导入 torch 或 ultralytics。"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import sha256

SUPPORTED_SCHEMAS = (2, 3)
SPLITS = ("train", "val", "test")
EPSILON = 1e-6
MANIFEST_KEYS = {"split", "image", "label", "image_sha256", "label_sha256", "bboxes",
                 "width", "height", "instances", "group_id"}


class SnapshotError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("；".join(self.issues))


@dataclass(frozen=True)
class SnapshotIndex:
    root: Path
    data: dict
    summary: dict
    rows: list
    hashes: dict
    image_paths: tuple
    label_paths: tuple


def _snapshot_files(dataset_dir):
    """快照内的 PNG/TXT（相对路径）；其他文件（如 Ultralytics 缓存）不属于清单。"""
    found = {}
    for folder, suffix in (("images", ".png"), ("labels", ".txt")):
        base = dataset_dir / folder
        if base.is_dir():
            for path in sorted(base.rglob(f"*{suffix}")):
                if path.is_file():
                    found[path.relative_to(dataset_dir).as_posix()] = path
    return found


def _hash_snapshot(dataset_dir):
    hashes = {}
    for name in ("data.yaml", "summary.json", "manifest.jsonl", "splits.yaml"):
        if (dataset_dir / name).is_file():
            hashes[name] = sha256(dataset_dir / name)
    for relative, path in _snapshot_files(dataset_dir).items():
        hashes[relative] = sha256(path)
    return hashes


def _yaml_mapping(payload, label, issues):
    try:
        data = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        issues.append(f"{label} 无法解析")
        return {}
    if not isinstance(data, dict):
        issues.append(f"{label} 顶层必须是映射")
        return {}
    return data


def _label_boxes(payload, where, issues):
    """解析 YOLO 文本为归一化框；只接受 ID 0 且数值有限、位于图像内的框。"""
    boxes = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        issues.append(f"{where} 不是 UTF-8 文本")
        return None
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or parts[0] != "0":
            issues.append(f"{where} 第 {number} 行不是 energy_unit（ID 0）的 5 字段 YOLO 标注")
            continue
        try:
            xc, yc, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(f"{where} 第 {number} 行含非数字坐标")
            continue
        values = (xc, yc, width, height)
        if not all(math.isfinite(v) for v in values) or width <= 0 or height <= 0:
            issues.append(f"{where} 第 {number} 行坐标非有限或宽高不为正")
            continue
        if xc - width / 2 < -EPSILON or xc + width / 2 > 1 + EPSILON \
                or yc - height / 2 < -EPSILON or yc + height / 2 > 1 + EPSILON:
            issues.append(f"{where} 第 {number} 行超出图像边界")
            continue
        boxes.append(values)
    return boxes


def _expected_yolo(bboxes, width, height):
    return [((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
            for x1, y1, x2, y2 in bboxes]


def _same_boxes(actual, expected):
    return len(actual) == len(expected) and all(
        abs(a - e) <= EPSILON for got, want in zip(actual, expected) for a, e in zip(got, want))


def _safe_relative(value, prefix, suffix):
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(suffix):
        return False
    parts = value.split("/")
    return not (value.startswith("/") or "\\" in value or any(p in ("", ".", "..") for p in parts))


def load_snapshot(data_yaml):
    """只读加载并校验快照（schema 2/3）；失败抛 SnapshotError，列出全部已发现问题。"""
    data_path = Path(data_yaml).resolve()
    dataset_dir = data_path.parent
    if not data_path.is_file():
        raise SnapshotError([f"data.yaml 不存在: {data_path}"])
    hashes = _hash_snapshot(dataset_dir)
    issues = []
    data = _yaml_mapping(data_path.read_bytes(), "data.yaml", issues)
    summary_path, manifest_path = dataset_dir / "summary.json", dataset_dir / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise SnapshotError(issues + ["缺少 summary.json 或 manifest.jsonl"])
    try:
        summary = json.loads(summary_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SnapshotError(issues + ["summary.json 无法解析"])
    if not isinstance(summary, dict):
        raise SnapshotError(issues + ["summary.json 顶层必须是对象"])
    schema = summary.get("schema_version")
    if type(schema) is not int or schema not in SUPPORTED_SCHEMAS:
        raise SnapshotError(issues + [f"不支持的 schema_version: {schema!r}，仅支持 2 与 3"])
    if summary.get("class_names") != {"0": "energy_unit"}:
        issues.append(f"summary.json 的 class_names 应为 {{'0': 'energy_unit'}}，当前为 {summary.get('class_names')!r}")
    if data.get("names") not in ({0: "energy_unit"}, ["energy_unit"]):
        issues.append(f"data.yaml 的 names 应只包含 energy_unit，当前为 {data.get('names')!r}")
    if data.get("train") != "images/train" or data.get("val") != "images/val":
        issues.append("data.yaml 的 train/val 必须分别为 images/train 与 images/val")
    declared = data.get("path")
    if not isinstance(declared, str) or not Path(declared).is_absolute() \
            or Path(declared).resolve() != dataset_dir:
        issues.append(f"data.yaml 的 path 必须是等于数据集目录的绝对路径（当前为 {declared!r}）")
    split_images = summary.get("split_images")
    if not isinstance(split_images, dict) or set(split_images) - set(SPLITS) or any(
            type(v) is not int or v < 0 for v in split_images.values()):
        issues.append("summary.json 的 split_images 无效")
        split_images = {}
    if ("test" in data) != ("test" in split_images) or ("test" in data and data["test"] != "images/test"):
        issues.append("data.yaml 的 test 键与 summary.json 的 test 划分不一致")
    for split in ("train", "val"):
        if type(split_images.get(split, 0)) is not int or split_images.get(split, 0) <= 0:
            issues.append(f"{split} 划分必须至少有 1 张图片，summary.json 记录为 {split_images.get(split)!r}")
    if schema == 3:
        for key in ("algorithm", "group_definition", "split_stats", "warnings"):
            if key not in summary:
                issues.append(f"schema 3 的 summary.json 缺少 {key}")
    group_splits = summary.get("group_splits")
    if not isinstance(group_splits, dict):
        issues.append("summary.json 缺少 group_splits")
        group_splits = {}
    rows = []
    for number, line in enumerate(manifest_path.read_bytes().decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            issues.append(f"manifest.jsonl 第 {number} 行无法解析")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            issues.append(f"manifest.jsonl 第 {number} 行必须是对象")
    required = MANIFEST_KEYS | ({"split_group_id", "scene"} if schema == 3 else set())
    counts = {split: 0 for split in SPLITS}
    owner_splits, image_splits, scenes, present, paths = {}, {}, {}, set(), set()
    valid = []
    for index, row in enumerate(rows, 1):
        where = f"manifest 第 {index} 行"
        missing = required - set(row)
        if missing:
            issues.append(f"{where} 缺少字段 {sorted(missing)}")
            continue
        split, image_rel, label_rel = row["split"], row["image"], row["label"]
        if split not in SPLITS:
            issues.append(f"{where} 的 split 未知: {split!r}")
            continue
        if not _safe_relative(image_rel, f"images/{split}/", ".png") or \
                label_rel != "labels/" + image_rel[len("images/"):-len(".png")] + ".txt":
            issues.append(f"{where} 的图片或标签路径不合规: {image_rel!r}, {label_rel!r}")
            continue
        if image_rel in paths:
            issues.append(f"{where} 图片重复: {image_rel}")
            continue
        paths.add(image_rel)
        key = row["split_group_id"] if schema == 3 else row["group_id"]
        if schema == 3:
            scenes.setdefault(key, set()).add(row["scene"])
        if group_splits.get(key) != split:
            issues.append(f"{where} 的份 {key!r} 与 summary.group_splits 不一致")
        owner_splits.setdefault(key, set()).add(split)
        present.add(key)
        image_splits.setdefault(row["image_sha256"], set()).add(split)
        counts[split] += 1
        valid.append((key, row))
    for key, splits_seen in owner_splits.items():
        if len(splits_seen) > 1:
            issues.append(f"份 {key!r} 同时出现在多个集合")
    for key, scene_set in scenes.items():
        if len(scene_set) > 1:
            issues.append(f"份 {key!r} 含多个 scene")
    for digest, splits_seen in image_splits.items():
        if len(splits_seen) > 1:
            issues.append(f"同一图片内容跨集合出现: {digest[:12]}")
    for group in sorted(set(group_splits) - present):
        issues.append(f"份 {group!r} 在 group_splits 中但清单中没有图片")
    for split in SPLITS:
        if isinstance(split_images.get(split), int) and split_images[split] != counts[split]:
            issues.append(f"{split} 划分的清单图片数为 {counts[split]}，summary.json 记录 {split_images[split]}")
    if schema == 3 and isinstance(summary.get("split_stats"), dict):
        stats = summary["split_stats"].get("splits", {})
        for split in SPLITS:
            recorded = stats.get(split, {}).get("images", 0) if isinstance(stats.get(split, {}), dict) else None
            if split in counts and recorded is not None and recorded != counts[split]:
                issues.append(f"split_stats 中 {split} 的图片数为 {recorded}，清单为 {counts[split]}")
    for _, row in valid:
        image_file, label_file = dataset_dir / row["image"], dataset_dir / row["label"]
        if not image_file.is_file() or not label_file.is_file():
            issues.append(f"{row['image']} 与标签不配对：缺少图片或标签文件")
            continue
        if sha256(image_file) != row["image_sha256"]:
            issues.append(f"{row['image']} 图片哈希与清单不一致")
        label_payload = label_file.read_bytes()
        if hashlib.sha256(label_payload).hexdigest() != row["label_sha256"]:
            issues.append(f"{row['label']} 标签哈希与清单不一致")
        width, height, bboxes = row["width"], row["height"], row["bboxes"]
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0 \
                or not isinstance(bboxes, list):
            issues.append(f"{row['image']} 的宽高或 bboxes 无效")
            continue
        if not all(_valid_box(box, width, height) for box in bboxes):
            issues.append(f"{row['image']} 含无效框（非有限、退化或越界）")
            continue
        if type(row["instances"]) is not int or row["instances"] != len(bboxes):
            issues.append(f"{row['image']} 的 instances 与 bboxes 数量不一致")
        actual = _label_boxes(label_payload, row["label"], issues)
        if actual is not None and not _same_boxes(actual, _expected_yolo(bboxes, width, height)):
            issues.append(f"{row['label']} 与清单中的框不一致")
    listed = {row["image"] for _, row in valid} | {row["label"] for _, row in valid}
    extra = sorted(set(_snapshot_files(dataset_dir)) - listed)
    if extra:
        issues.append(f"快照中有清单外的 PNG/TXT，例如 {extra[:5]}")
    if issues:
        raise SnapshotError(issues)
    return SnapshotIndex(
        root=dataset_dir, data=data, summary=summary, rows=[row for _, row in valid], hashes=hashes,
        image_paths=tuple(sorted(row["image"] for _, row in valid)),
        label_paths=tuple(sorted(row["label"] for _, row in valid)))


def _valid_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def verify_snapshot_unchanged(index):
    """使用快照前后复核：任一 PNG/TXT、清单或配置文件的摘要变化都拒绝继续。"""
    current = _hash_snapshot(index.root)
    changed = sorted(key for key in set(current) | set(index.hashes) if current.get(key) != index.hashes.get(key))
    if changed:
        raise SnapshotError([f"快照在使用期间发生变化，例如 {changed[:5]}"])
