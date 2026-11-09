"""light_v1 离线增强：读取已划分快照，只为 train 生成副本，发布完整派生快照。

本模块不导入 torch 或 ultralytics。Albumentations 只在依赖检查与 augment_one 中按需导入。
"""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import yaml

from .augmentation import (
    MAX_ATTEMPTS, RECIPE_NAME, CandidateRejected, augment_one, derive_sample_seed, recipe_params,
)
from .common import json_text, new_output_directory, project_path, sha256, utc_now
from .snapshot import SnapshotError, load_snapshot, verify_snapshot_unchanged

SPLITS = ("train", "val", "test")
PINNED = {"albumentations": "2.0.8", "albucore": "0.0.24"}
CV2_DISTRIBUTIONS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python")
LIBRARY_NAMES = ("albumentations", "albucore", "opencv-python", "numpy")
RESERVED_DIR = "__aug__"
MAX_COPIES = 9999


class AugmentationError(ValueError):
    """增强输入、依赖或发布失败；命令行返回 2。"""


@dataclass(frozen=True)
class AugmentationPlan:
    index: object
    copies: int
    seed: int
    expected_copies: int


def _installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _check_dependencies():
    present = [name for name in CV2_DISTRIBUTIONS if _installed(name)]
    if len(present) != 1:
        raise AugmentationError(
            f"必须且只能安装一个 OpenCV 发行包，当前为 {present or '无'}；参见 docs/augmentation_environment.md")
    for name, version in PINNED.items():
        found = _installed(name)
        if found != version:
            raise AugmentationError(f"{name} 版本必须是 {version}，当前为 {found}")
    try:
        import albumentations  # noqa: F401
    except ImportError as exc:
        raise AugmentationError(f"无法导入 albumentations: {exc}") from exc


def _check_int(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        bound = "整数" if minimum is None else f"不小于 {minimum} 的整数"
        raise AugmentationError(f"{name} 必须为{bound}")


def prepare_augmentation(data_yaml, *, copies=1, seed=42):
    """只读校验基础快照（schema 2/3）与依赖，计算预期副本数；不创建任何文件。"""
    _check_int(copies, "copies", 1)
    if copies > MAX_COPIES:
        raise AugmentationError(
            f"copies 不能大于 {MAX_COPIES}：副本文件名使用 4 位编号 __aug{{copy_index:04d}}，超过会破坏命名")
    _check_int(seed, "seed")
    _check_dependencies()
    data_path = project_path(data_yaml).resolve()
    if data_path.name != "data.yaml":
        raise AugmentationError(f"--data 必须指向 data.yaml 文件: {data_yaml}")
    try:
        index = load_snapshot(data_path)
    except SnapshotError as exc:
        raise AugmentationError("基础快照校验失败：" + "；".join(exc.issues)) from exc
    for folder in ("images", "labels"):
        if any(p.name == RESERVED_DIR for p in (index.root / folder).rglob("*") if p.is_dir()):
            raise AugmentationError(f"基础快照含保留目录 {RESERVED_DIR}，拒绝增强以避免递归")
    train = [row for row in index.rows if row["split"] == "train"]
    if not train:
        raise AugmentationError("基础快照没有 train 图片")
    return AugmentationPlan(index=index, copies=copies, seed=seed, expected_copies=len(train) * copies)


def _with_provenance_fields(row):
    group = row["group_id"]
    return {**row,
            "split_group_id": row.get("split_group_id", group),
            "scene": row.get("scene", group),
            "tags": list(row.get("tags", []))}


def _overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def _check_output(output, base, raw_root):
    if _overlaps(output, base.resolve()):
        raise ValueError("输出目录不能与输入快照目录重叠")
    if raw_root and _overlaps(output, Path(raw_root).resolve()):
        raise ValueError("输出目录不能与原图目录重叠")


def _copy_original(row, base, staging):
    for key, digest_key in (("image", "image_sha256"), ("label", "label_sha256")):
        destination = staging / row[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / row[key], destination)
        if sha256(destination) != row[digest_key]:
            raise AugmentationError(f"复制后哈希与清单不一致: {row[key]}")


def _corners(box):
    xc, yc, width, height = box
    x1, x2 = max(0.0, xc - width / 2), min(1.0, xc + width / 2)
    y1, y2 = max(0.0, yc - height / 2), min(1.0, yc + height / 2)
    if not (x1 < x2 and y1 < y2):
        return None
    return x1, y1, x2, y2


def _write_candidate(staging, parent, copy_index, attempt, seed, payload, corners):
    session = Path(parent["image"]).parent.name
    name = f"{Path(parent['image']).stem}__aug{copy_index:04d}"
    rel_image = f"images/train/{RESERVED_DIR}/{session}/{name}.png"
    rel_label = f"labels/train/{RESERVED_DIR}/{session}/{name}.txt"
    label_text = "".join(
        f"0 {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}\n"
        for x1, y1, x2, y2 in corners)
    label_bytes = label_text.encode("utf-8")
    image_path, label_path = staging / rel_image, staging / rel_label
    if image_path.exists() or label_path.exists():
        raise AugmentationError(f"增强输出路径冲突: {rel_image}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(payload)
    label_path.write_bytes(label_bytes)
    width, height = parent["width"], parent["height"]
    return {
        "kind": "augmentation", "status": "generated", "split": "train",
        "session": session, "image": rel_image, "label": rel_label,
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "width": width, "height": height,
        "bboxes": [[x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in corners],
        "instances": len(corners),
        "group_id": parent["group_id"], "split_group_id": parent["split_group_id"],
        "scene": parent["scene"], "tags": list(parent["tags"]),
        "parent_image": parent["image"], "parent_image_sha256": parent["image_sha256"],
        "parent_label_sha256": parent["label_sha256"],
        "copy_index": copy_index, "attempt": attempt, "sample_seed": seed,
    }


def _make_copy(parent, copy_index, plan, base, staging, eval_hashes, augment):
    width, height = parent["width"], parent["height"]
    image = cv2.imread(str(base / parent["image"]), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise AugmentationError(f"无法读取或尺寸与清单不符: {parent['image']}")
    boxes = [[(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
             for x1, y1, x2, y2 in parent["bboxes"]]
    reasons = []
    for attempt in range(MAX_ATTEMPTS):
        seed = derive_sample_seed(plan.seed, parent["image"], copy_index, attempt)
        try:
            sample = augment(image, boxes, sample_seed=seed)
        except CandidateRejected as exc:
            reasons.append(exc.code)
            continue
        if sample.image_bgr.shape != image.shape or sample.image_bgr.dtype != np.uint8:
            raise AugmentationError("增强输出的尺寸或类型与输入不一致")
        if len(sample.boxes_yolo) != len(boxes) or sorted(sample.instance_ids) != list(range(len(boxes))):
            reasons.append("dropped_box")
            continue
        if np.array_equal(sample.image_bgr, image):
            reasons.append("unchanged_image")
            continue
        corners = [_corners(box) for box in sample.boxes_yolo]
        if any(corner is None for corner in corners):
            reasons.append("invalid_box")
            continue
        ok, encoded = cv2.imencode(".png", sample.image_bgr)
        if not ok:
            raise AugmentationError(f"PNG 编码失败: {parent['image']}")
        payload = encoded.tobytes()
        if hashlib.sha256(payload).hexdigest() in eval_hashes:
            reasons.append("duplicate_eval_image")
            continue
        return _write_candidate(staging, parent, copy_index, attempt, seed, payload, corners), None
    return None, reasons


def _check_cross_split_duplicates(rows):
    seen = {}
    for row in rows:
        seen.setdefault(row["image_sha256"], set()).add(row["split"])
    conflicts = sorted(digest[:12] for digest, splits in seen.items() if len(splits) > 1)
    if conflicts:
        raise AugmentationError(f"同一图片内容跨集合出现: {conflicts[:5]}")


def _library_versions():
    return {name: _installed(name) for name in LIBRARY_NAMES}


def _summary(plan, rows, augmented, skipped):
    base = plan.index.summary
    index = plan.index
    counts = {split: 0 for split in SPLITS}
    for row in rows:
        counts[row["split"]] += 1
    split_images = {split: counts[split] for split in SPLITS if split in base["split_images"]}
    total = sum(base["split_images"].values())
    instances = sum(len(row["bboxes"]) for row in rows)
    positive = sum(1 for row in rows if row["bboxes"])
    return {
        "schema_version": 4,
        "created_at": utc_now(),
        "dataset_kind": "augmented",
        "class_names": {"0": "energy_unit"},
        "split_images": split_images,
        "base_split_images": dict(base["split_images"]),
        "base_ratios": {split: base["split_images"][split] / total for split in base["split_images"]},
        "instances": instances,
        "positive": positive,
        "negative": len(rows) - positive,
        "group_splits": dict(base["group_splits"]),
        "base_dataset": {
            "path": str(index.root),
            "data_yaml_sha256": index.hashes["data.yaml"],
            "summary_sha256": index.hashes["summary.json"],
            "manifest_sha256": index.hashes["manifest.jsonl"],
            "files_sha256": {key: value for key, value in index.hashes.items()
                             if key.startswith(("images/", "labels/"))},
        },
        "augmentation": {
            "recipe": RECIPE_NAME,
            "recipe_params": recipe_params(),
            "copies": plan.copies,
            "seed": plan.seed,
            "max_attempts": MAX_ATTEMPTS,
            "libraries": _library_versions(),
            "planned": plan.expected_copies,
            "generated": len(augmented),
            "skipped": len(skipped),
            "skipped_items": skipped,
        },
    }


def augment_dataset(plan, output, *, augment=augment_one):
    """生成完整派生快照到全新目录；任何失败都不发布。augment 只供测试注入。"""
    index = plan.index
    base = index.root
    output = project_path(output).resolve()
    _check_output(output, base, index.summary.get("raw_root"))
    if output.exists():
        raise FileExistsError(f"输出已存在，请使用新目录: {output}")
    originals = [_with_provenance_fields(row) for row in sorted(index.rows, key=lambda r: r["image"])]
    parents = [row for row in originals if row["split"] == "train"]
    eval_hashes = {row["image_sha256"] for row in originals if row["split"] in ("val", "test")}
    augmented, skipped = [], []
    with new_output_directory(output) as staging:
        for row in originals:
            _copy_original(row, base, staging)
        for parent in parents:
            for copy_index in range(1, plan.copies + 1):
                row, reasons = _make_copy(parent, copy_index, plan, base, staging, eval_hashes, augment)
                if row is None:
                    skipped.append({"parent_image": parent["image"], "copy_index": copy_index,
                                    "reasons": reasons})
                else:
                    augmented.append(row)
        if not augmented:
            raise AugmentationError("所有增强副本均失败，不发布只有原图的增强集")
        rows = [{**row, "kind": "original"} for row in originals] + augmented
        _check_cross_split_duplicates(rows)
        summary = _summary(plan, rows, augmented, skipped)
        present = [split for split in SPLITS if split in summary["split_images"]]
        data = {"path": str(output), "train": "images/train", "val": "images/val",
                "names": {0: "energy_unit"}}
        if "test" in present:
            data["test"] = "images/test"
        splits = {split: sorted(g for g, s in summary["group_splits"].items() if s == split)
                  for split in present}
        (staging / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")
        (staging / "splits.yaml").write_text(yaml.safe_dump(splits, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8")
        (staging / "summary.json").write_text(json_text(summary), encoding="utf-8")
        verify_snapshot_unchanged(index)
    return {**summary, "output": str(output)}


def _build_parser():
    parser = argparse.ArgumentParser(description="为已划分的快照生成 light_v1 离线增强副本（只增强 train）。")
    parser.add_argument("--data", required=True, help="基础快照的 data.yaml（schema 2/3）")
    parser.add_argument("--copies", type=int, default=1, help="每张 train 原图的副本数，默认 1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="只校验并报告计划，不生成任何文件")
    mode.add_argument("--output", help="输出的全新数据集目录，例如 datasets/energy_v2_aug1")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        plan = prepare_augmentation(args.data, copies=args.copies, seed=args.seed)
        if args.check_only:
            report = {
                "recipe": RECIPE_NAME,
                "copies": plan.copies,
                "seed": plan.seed,
                "train_images": plan.expected_copies // plan.copies,
                "planned_generated": plan.expected_copies,
                "note": "预计数量是上限，实际生成数以发布报告为准",
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        summary = augment_dataset(plan, args.output)
    except (AugmentationError, OSError, ValueError) as exc:
        print(f"增强失败: {exc}", file=sys.stderr)
        return 2
    report = {
        "output": summary["output"],
        "requested": summary["augmentation"]["planned"],
        "generated": summary["augmentation"]["generated"],
        "skipped": summary["augmentation"]["skipped"],
        "skipped_items": summary["augmentation"]["skipped_items"],
        "split_images": summary["split_images"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
