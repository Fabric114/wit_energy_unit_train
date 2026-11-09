"""读取 GUI 标注，按采集组构建独立、可追溯的 YOLO 数据集。"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .annotations import (
    annotation_status, check_output_location, ensure_session_unchanged,
    ensure_unchanged, load_instances, read_annotation_snapshot,
    session_folder, session_inputs,
)
from .common import json_text, new_output_directory, sha256, utc_now, valid_name
from .splitting import (
    assign_groups, check_duplicate_groups, load_yaml_snapshot, resolve_groups,
    summarize_groups, summarize_splits, verify_config_snapshots,
)

__all__ = ["DatasetValidationError", "PreparedDataset", "assign_groups", "build_dataset",
           "discover_sessions", "inspect_dataset", "prepare_dataset"]


class DatasetValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"校验发现 {len(report['issues'])} 项问题，请先在 X-AnyLabeling 处理")


def discover_sessions(raw_root, sessions=None):
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise ValueError(f"原图根目录不存在: {root}")
    names = list(sessions) if sessions is not None else [
        p.name for p in root.iterdir()
        if p.is_dir() and p.name != "_delete_"
        and ((p / "meta.json").exists() or any(p.glob("*.png")))
    ]
    if not names:
        raise ValueError("未找到采集 session；先采集图片或用 --session 指定")
    return sorted({valid_name(name) for name in names})


def inspect_dataset(raw_root, sessions=None):
    """只读扫描，集中返回问题；旧模板和 review.json 不参与默认读取。"""
    names = discover_sessions(raw_root, sessions)
    summary = dict.fromkeys(("images", "passed", "pending", "failed", "missing", "invalid",
                             "positive", "negative", "instances", "clipped_boxes"), 0)
    records, sources, issues, groups = [], {}, [], set()
    for session in names:
        try:
            meta, images, digest = session_inputs(raw_root, session)
        except (ValueError, OSError) as exc:
            summary["invalid"] += 1
            issues.append({"image": session, "kind": "invalid", "reason": str(exc)})
            continue
        group = meta["group_id"]
        groups.add(group)
        sources[session] = {"images": [p.name for p in images], "meta_sha256": digest, "group_id": group}
        folder = session_folder(raw_root, session)
        for path in images:
            summary["images"] += 1
            image_id = f"{session}/{path.name}"
            label = folder / (path.stem + ".json")
            if not label.exists():
                summary["missing"] += 1
                issues.append({"image": image_id, "kind": "missing", "reason": "缺少同名 JSON，请在 GUI 保存；缺标注不等于负样本"})
                continue
            try:
                image, doc, snapshot = read_annotation_snapshot(path, label, meta)
                status = annotation_status(doc)
                instances = load_instances(doc, image.shape, path.name)
            except (ValueError, OSError) as exc:
                summary["invalid"] += 1
                issues.append({"image": image_id, "kind": "invalid", "reason": str(exc)})
                continue
            summary[status] += 1
            if status != "passed":
                reason = "GUI 尚未检查通过" if status == "pending" else "GUI 标记为不通过，请修正或移出本次数据集"
                issues.append({"image": image_id, "kind": status, "reason": reason})
            else:
                summary["positive" if instances else "negative"] += 1
                summary["instances"] += len(instances)
            clipped = sum(item["clipped"] for item in instances)
            summary["clipped_boxes"] += clipped
            records.append({"session": session, "group_id": group, "status": status,
                            "source_image": str(path), "source_annotation": str(label), **snapshot,
                            "width": image.shape[1], "height": image.shape[0],
                            "bboxes": [i["bbox"] for i in instances], "clipped_boxes": clipped})
    summary.update(sessions=len(names), groups=len(groups))
    return {"summary": summary, "issues": issues, "records": records, "sources": sources,
            "sessions": names, "group_ids": sorted(groups)}


@dataclass(frozen=True)
class PreparedDataset:
    """check 与 build 共用的划分结果；检查或构建都只调用一次 prepare_dataset。"""
    report: dict
    records: list
    assignment: dict
    split_stats: dict
    warnings: list
    group_definition: dict
    split_config: "dict | None"
    config_snapshots: tuple
    algorithm: str
    seed: int
    val_ratio: "float | None"
    test_ratio: "float | None"


def prepare_dataset(raw_root, *, sessions=None, val_ratio=.2, test_ratio=0, seed=42,
                    splits=None, group_file=None, split_file=None):
    if splits is not None and split_file is not None:
        raise ValueError("固定划分只能使用 splits 或 split_file 之一")
    report = inspect_dataset(raw_root, sessions)
    if report["issues"]:
        raise DatasetValidationError(report)
    image_ids = {f"{session}/{name}" for session, source in report["sources"].items()
                 for name in source["images"]}
    snapshots, split_config = [], None
    if split_file is not None:
        splits, snapshot = load_yaml_snapshot(split_file)
        snapshots.append(snapshot)
        split_config = {"source": "file", "path": str(snapshot.path), "sha256": snapshot.sha256, "splits": splits}
    elif splits is not None:
        split_config = {"source": "inline", "path": None, "sha256": None, "splits": splits}
    resolved = resolve_groups(report["records"], image_ids, group_file)
    if resolved.snapshot is not None:
        snapshots.append(resolved.snapshot)
    duplicates = check_duplicate_groups(resolved.records)
    stats = summarize_groups(resolved.records)
    groups = sorted({row["split_group_id"] for row in resolved.records})
    assignment = assign_groups(groups, val_ratio, test_ratio, seed, splits, group_stats=stats)
    fixed = splits is not None
    ratios = None if fixed else {"train": 1 - val_ratio - test_ratio, "val": val_ratio,
                                 **({"test": test_ratio} if test_ratio > 0 else {})}
    split_stats, warnings = summarize_splits(resolved.records, assignment, ratios,
                                             excluded_count=len(resolved.excluded))
    warnings = duplicates + warnings
    if not resolved.confirmed:
        warnings.insert(0, {"code": "group_file_unconfirmed", "path": resolved.definition["path"]})
    return PreparedDataset(
        report=report, records=resolved.records, assignment=assignment, split_stats=split_stats,
        warnings=warnings, group_definition=resolved.definition, split_config=split_config,
        config_snapshots=tuple(snapshots), algorithm="fixed" if fixed else "shuffled_groups_v1",
        seed=seed, val_ratio=None if fixed else val_ratio, test_ratio=None if fixed else test_ratio)


def _verify_sources(raw_root, selected_sessions, report):
    if discover_sessions(raw_root, selected_sessions) != report["sessions"]:
        raise ValueError("构建期间 session 清单发生变化，请重新运行")
    for session, source in report["sources"].items():
        ensure_session_unchanged(raw_root, session, source["images"], source["meta_sha256"])
    for row in report["records"]:
        ensure_unchanged(Path(row["source_image"]), Path(row["source_annotation"]), row)


def build_dataset(raw_root, output, *, sessions=None,
                  val_ratio=.2, test_ratio=0, seed=42, splits=None,
                  group_file=None, split_file=None):
    check_output_location(output, raw_root)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在，请使用新目录: {output}")
    prepared = prepare_dataset(raw_root, sessions=sessions, val_ratio=val_ratio, test_ratio=test_ratio,
                               seed=seed, splits=splits, group_file=group_file, split_file=split_file)
    if not prepared.group_definition["confirmed"]:
        raise ValueError("group-file 的 confirmed 不是 true，分组尚未人工核对，拒绝构建")
    report, assignment, rows = prepared.report, prepared.assignment, []
    summary = {"schema_version": 3, "created_at": utc_now(), **report["summary"],
               "raw_root": str(Path(raw_root).resolve()),
               "class_names": {"0": "energy_unit"}, "group_splits": assignment,
               "algorithm": prepared.algorithm, "split_method": prepared.algorithm,
               "group_definition": prepared.group_definition, "split_config": prepared.split_config,
               "split_stats": prepared.split_stats, "warnings": prepared.warnings,
               "excluded_images": len(prepared.group_definition.get("exclude", [])),
               "seed": prepared.seed, "val_ratio": prepared.val_ratio, "test_ratio": prepared.test_ratio,
               "sources": report["sources"],
               "split_images": {split: 0 for split in sorted(set(assignment.values()))}}
    with new_output_directory(output) as staging:
        for row in prepared.records:
            split = assignment[row["split_group_id"]]
            source = Path(row["source_image"])
            image_relative = f"images/{split}/{row['session']}/{source.name}"
            label_relative = f"labels/{split}/{row['session']}/{source.stem}.txt"
            image_copy, label_file = staging / image_relative, staging / label_relative
            image_copy.parent.mkdir(parents=True, exist_ok=True)
            label_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, image_copy)
            if sha256(image_copy) != row["image_sha256"]:
                raise ValueError(f"复制期间原图发生变化: {source.name}")
            width, height = row["width"], row["height"]
            text = "".join(
                f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n"
                for x1, y1, x2, y2 in row["bboxes"]
            )
            label_file.write_text(text, encoding="utf-8")
            rows.append({**row, "split": split, "image": image_relative, "label": label_relative,
                         "instances": len(row["bboxes"]), "label_sha256": hashlib.sha256(text.encode()).hexdigest()})
            summary["split_images"][split] += 1
        data = {"path": str(output), "train": "images/train", "val": "images/val", "names": {0: "energy_unit"}}
        if "test" in assignment.values():
            data["test"] = "images/test"
        full_split = {split: sorted(g for g, s in assignment.items() if s == split)
                      for split in ("train", "val", "test") if split in set(assignment.values())}
        (staging / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staging / "splits.yaml").write_text(yaml.safe_dump(full_split, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staging / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        (staging / "summary.json").write_text(json_text(summary), encoding="utf-8")
        _verify_sources(raw_root, sessions, report)
        verify_config_snapshots(prepared.config_snapshots)
    return {**summary, "output": str(output)}
