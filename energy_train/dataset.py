"""读取 GUI 标注，按采集组构建独立、可追溯的 YOLO 数据集。"""

import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import yaml

from .annotations import (
    annotation_status, check_output_location, ensure_session_unchanged,
    ensure_unchanged, load_instances, read_annotation_snapshot,
    session_folder, session_inputs,
)
from .common import json_text, new_output_directory, sha256, utc_now, valid_name


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


def assign_groups(groups, val_ratio=.2, test_ratio=0, seed=42, splits=None):
    """确定性整组划分；比例按组数近似，每个启用的集合至少有一组。"""
    groups = sorted(set(groups))
    if splits is not None:
        if not isinstance(splits, dict) or set(splits) - {"train", "val", "test"}:
            raise ValueError("固定划分必须是仅含 train/val/test 的映射")
        if not splits.get("train") or not splits.get("val"):
            raise ValueError("固定划分的 train 和 val 都必须有采集组")
        mapping = {}
        for split, members in splits.items():
            if not isinstance(members, list) or any(not isinstance(g, str) for g in members):
                raise ValueError(f"{split} 必须是采集 group_id 的字符串列表")
            for group in members:
                if group in mapping:
                    raise ValueError(f"采集组重复分配: {group}")
                mapping[group] = split
        if set(mapping) != set(groups):
            raise ValueError(f"固定划分与当前采集组不一致；遗漏 {sorted(set(groups) - set(mapping))}，多余 {sorted(set(mapping) - set(groups))}")
        return mapping
    if not (math.isfinite(val_ratio) and math.isfinite(test_ratio)
            and 0 < val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val-ratio 必须大于 0，test-ratio 不小于 0，二者之和小于 1，且为有限数值")
    ratios = {"train": 1 - val_ratio - test_ratio, "val": val_ratio}
    if test_ratio > 0:
        ratios["test"] = test_ratio
    if len(groups) < len(ratios):
        raise ValueError(f"独立采集组不足：至少需要 {len(ratios)} 组，当前只有 {len(groups)} 组；请补采独立场景，不能按帧拆分同一组")
    counts = {split: 1 for split in ratios}
    for _ in range(len(groups) - len(ratios)):
        split = max(ratios, key=lambda key: len(groups) * ratios[key] - counts[key])
        counts[split] += 1
    random.Random(seed).shuffle(groups)
    mapping, offset = {}, 0
    for split, count in counts.items():
        mapping.update(dict.fromkeys(groups[offset:offset + count], split))
        offset += count
    return mapping


def _verify_sources(raw_root, selected_sessions, report):
    if discover_sessions(raw_root, selected_sessions) != report["sessions"]:
        raise ValueError("构建期间 session 清单发生变化，请重新运行")
    for session, source in report["sources"].items():
        ensure_session_unchanged(raw_root, session, source["images"], source["meta_sha256"])
    for row in report["records"]:
        ensure_unchanged(Path(row["source_image"]), Path(row["source_annotation"]), row)


def build_dataset(raw_root, output, *, sessions=None,
                  val_ratio=.2, test_ratio=0, seed=42, splits=None):
    check_output_location(output, raw_root)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在，请使用新目录: {output}")
    report = inspect_dataset(raw_root, sessions)
    if report["issues"]:
        raise DatasetValidationError(report)
    assignment = assign_groups(report["group_ids"], val_ratio, test_ratio, seed, splits)
    rows = []
    summary = {"schema_version": 2, "created_at": utc_now(), **report["summary"],
               "raw_root": str(Path(raw_root).resolve()),
               "class_names": {"0": "energy_unit"}, "group_splits": assignment,
               "split_method": "fixed" if splits is not None else "seeded_group_count", "seed": seed,
               "val_ratio": val_ratio if splits is None else None,
               "test_ratio": test_ratio if splits is None else None, "sources": report["sources"],
               "split_images": {split: 0 for split in sorted(set(assignment.values()))}}
    with new_output_directory(output) as staging:
        for row in report["records"]:
            split = assignment[row["group_id"]]
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
        (staging / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staging / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        (staging / "summary.json").write_text(json_text(summary), encoding="utf-8")
        _verify_sources(raw_root, sessions, report)
    return {**summary, "output": str(output)}
