"""场景分组：解析可选 group-file（份、范围展开、排除与标签）并校验完整性。

份（split group）是不可拆分的划分单位。group-file 只引用现有 PNG，不移动数据。
"""

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import valid_name

TOP_KEYS = {"schema_version", "confirmed", "groups", "exclude", "tags"}
REQUIRED_KEYS = {"schema_version", "confirmed", "groups"}
GROUP_KEYS = {"id", "scene", "reason", "images", "ranges"}
EXCLUDE_KEYS = {"reason", "images", "ranges"}
RANGE_KEYS = {"from", "to"}
TAG_VALUES = ("blur", "occlusion", "glare", "edge", "far")


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class ResolvedGroups:
    records: list
    excluded: tuple
    definition: dict
    snapshot: "FileSnapshot | None"
    confirmed: bool


class _StrictLoader(yaml.SafeLoader):
    """拒绝重复键；标准 SafeLoader 会静默保留最后一个值。"""


def _construct_unique_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"YAML 键重复: {key}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_snapshot(path):
    """一次读取字节并据此解析；之后只使用这份字节，避免读取与摘要之间被改写。"""
    path = Path(path).resolve()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取配置文件: {path}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"配置文件必须是 UTF-8 编码: {path}") from exc
    try:
        data = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"配置顶层必须是映射: {path}")
    return data, FileSnapshot(path, payload, hashlib.sha256(payload).hexdigest())


def image_id(record):
    return f"{record['session']}/{Path(record['source_image']).name}"


def _image_ref(value, image_ids, where):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: 图片路径必须是非空字符串")
    parts = value.split("/")
    if (value.startswith("/") or "\\" in value or len(parts) != 2
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError(f"{where}: 图片路径非法，需为 session/文件名（禁止绝对路径、反斜杠与 ..）: {value}")
    if value not in image_ids:
        raise ValueError(f"{where}: 图片不在本次所选 session 的现存 PNG 中: {value}")
    return value


def _range_members(item, image_ids, where):
    if not isinstance(item, dict) or set(item) != RANGE_KEYS:
        raise ValueError(f"{where}: range 只能包含 from 与 to 两个键")
    start = _image_ref(item["from"], image_ids, where)
    end = _image_ref(item["to"], image_ids, where)
    session = start.split("/")[0]
    if end.split("/")[0] != session:
        raise ValueError(f"{where}: range 不能跨 session: {start} → {end}")
    if start > end:
        raise ValueError(f"{where}: range 的 from 晚于 to: {start} > {end}")
    return {name for name in image_ids if name.split("/")[0] == session and start <= name <= end}


def _members(entry, image_ids, where):
    """展开 images 与 ranges，返回图片 ID 集合。"""
    for key in ("images", "ranges"):
        if key in entry and not isinstance(entry[key], list):
            raise ValueError(f"{where}: {key} 必须是列表")
    members = set()
    for value in entry.get("images", []):
        members.add(_image_ref(value, image_ids, where))
    for item in entry.get("ranges", []):
        members |= _range_members(item, image_ids, where)
    return members


def _reason(entry, where):
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{where}: reason 必须是非空字符串")
    return reason


def _parse_group_file(data, image_ids):
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise ValueError(f"group-file 含未知顶层键: {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"group-file 缺少必需键: {sorted(missing)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("group-file 的 schema_version 必须是整数 1")
    if not isinstance(data["confirmed"], bool):
        raise ValueError("group-file 的 confirmed 必须是 true 或 false")
    if not isinstance(data["groups"], list) or not data["groups"]:
        raise ValueError("group-file 的 groups 必须是非空列表")

    owner, groups, seen_ids = {}, [], set()
    for index, entry in enumerate(data["groups"], 1):
        if not isinstance(entry, dict) or set(entry) - GROUP_KEYS:
            raise ValueError(f"groups[{index}]: 必须是映射，且只能包含 {sorted(GROUP_KEYS)}")
        group_id = valid_name(entry.get("id"))
        where = f"份 {group_id}"
        if group_id in seen_ids:
            raise ValueError(f"份 ID 重复: {group_id}")
        seen_ids.add(group_id)
        scene = valid_name(entry.get("scene", group_id))
        reason = _reason(entry, where)
        members = _members(entry, image_ids, where)
        if not members:
            raise ValueError(f"{where} 展开后没有图片")
        for image in members:
            if image in owner:
                raise ValueError(f"图片同时属于多份: {image}（{owner[image]} 与 {group_id}）")
            owner[image] = group_id
        groups.append({"id": group_id, "scene": scene, "reason": reason, "images": sorted(members)})

    excluded = {}
    raw_exclude = data.get("exclude", [])
    if not isinstance(raw_exclude, list):
        raise ValueError("group-file 的 exclude 必须是列表")
    for index, entry in enumerate(raw_exclude, 1):
        where = f"exclude[{index}]"
        if not isinstance(entry, dict) or set(entry) - EXCLUDE_KEYS:
            raise ValueError(f"{where}: 只能包含 reason、images、ranges")
        reason = _reason(entry, where)
        members = _members(entry, image_ids, where)
        if not members:
            raise ValueError(f"{where} 展开后没有图片")
        for image in members:
            if image in owner:
                raise ValueError(f"图片同时出现在份与排除项中: {image}")
            if image in excluded:
                raise ValueError(f"图片重复排除: {image}")
            excluded[image] = reason

    raw_tags = data.get("tags", {})
    if not isinstance(raw_tags, dict):
        raise ValueError("group-file 的 tags 必须是映射")
    tags = {}
    for image, values in raw_tags.items():
        _image_ref(image, image_ids, "tags")
        if image in excluded:
            raise ValueError(f"tags 不能指向被排除的图片: {image}")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"tags 的值必须是字符串列表: {image}")
        if len(set(values)) != len(values):
            raise ValueError(f"tags 含重复标签: {image}")
        unknown_tags = set(values) - set(TAG_VALUES)
        if unknown_tags:
            raise ValueError(f"未知标签 {sorted(unknown_tags)}: {image}，允许值 {list(TAG_VALUES)}")
        if values:
            tags[image] = sorted(values)

    missing = sorted(image_ids - set(owner) - set(excluded))
    if missing:
        raise ValueError(f"group-file 遗漏 {len(missing)} 张图片，例如 {missing[:5]}；"
                         "每张现存 PNG 必须恰好出现在一份或 exclude 中")
    return {"confirmed": data["confirmed"], "groups": groups, "excluded": excluded, "tags": tags}


def resolve_groups(records, image_ids, group_file=None):
    """返回不修改输入的新行，并附加 split_group_id、scene 与 tags；exclude 的行不返回。"""
    records = list(records)
    image_ids = set(image_ids)
    ids = [image_id(row) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("审核记录中同一图片重复")
    unknown = sorted(set(ids) - image_ids)
    if unknown:
        raise ValueError(f"审核记录含未知图片: {unknown[:5]}")
    missing = sorted(image_ids - set(ids))
    if missing:
        raise ValueError(f"缺少审核记录的图片: {missing[:5]}")

    if group_file is None:
        members = {}
        for row in records:
            members.setdefault(row["group_id"], []).append(image_id(row))
        out = [{**row, "split_group_id": row["group_id"], "scene": row["group_id"], "tags": []}
               for row in records]
        definition = {"mode": "meta", "path": None, "sha256": None, "confirmed": True,
                      "groups": [{"id": g, "scene": g, "images": sorted(m)} for g, m in sorted(members.items())],
                      "exclude": [], "tags": {}}
        return ResolvedGroups(out, (), definition, None, True)

    data, snapshot = load_yaml_snapshot(group_file)
    parsed = _parse_group_file(data, image_ids)
    owner = {image: group["id"] for group in parsed["groups"] for image in group["images"]}
    scenes = {group["id"]: group["scene"] for group in parsed["groups"]}
    out = []
    for row in records:
        key = image_id(row)
        if key in parsed["excluded"]:
            continue
        group = owner[key]
        out.append({**row, "split_group_id": group, "scene": scenes[group],
                    "tags": parsed["tags"].get(key, [])})
    definition = {"mode": "file", "path": str(snapshot.path), "sha256": snapshot.sha256,
                  "confirmed": parsed["confirmed"], "groups": parsed["groups"],
                  "exclude": [{"image": image, "reason": reason}
                              for image, reason in sorted(parsed["excluded"].items())],
                  "tags": parsed["tags"]}
    return ResolvedGroups(out, tuple(sorted(parsed["excluded"])), definition, snapshot,
                          parsed["confirmed"])


def check_duplicate_groups(records):
    """同一内容跨份直接拒绝；同一份内的重复只报告 warning，不静默去重。"""
    by_digest = {}
    for row in records:
        by_digest.setdefault(row["image_sha256"], []).append(row)
    warnings = []
    for digest, rows in sorted(by_digest.items()):
        if len(rows) < 2:
            continue
        names = sorted(image_id(row) for row in rows)
        if len({row["split_group_id"] for row in rows}) > 1:
            raise ValueError(f"同一图片内容出现在不同份，拒绝构建: {names}")
        warnings.append({"code": "duplicate_image", "image_sha256": digest,
                         "images": names, "count": len(rows)})
    return warnings


def verify_config_snapshots(snapshots):
    """复核配置文件字节未变；构建完成前调用，防止复制期间的配置改动被忽略。"""
    for snapshot in snapshots:
        try:
            current = snapshot.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"构建期间配置文件消失: {snapshot.path}") from exc
        if current != snapshot.payload:
            raise ValueError(f"构建期间配置文件被修改: {snapshot.path}")


RATIO_TOLERANCE = .05


@dataclass(frozen=True)
class GroupStats:
    """一份的统计。multi 指含至少 3 个实例的图片，用于检查多目标覆盖。"""
    images: int
    instances: int
    negative: int
    multi: int
    tags: dict
    scene: str


def summarize_groups(records):
    acc = {}
    for row in records:
        group = row["split_group_id"]
        item = acc.setdefault(group, {"images": 0, "instances": 0, "negative": 0, "multi": 0,
                                      "tags": {}, "scene": row["scene"]})
        if item["scene"] != row["scene"]:
            raise ValueError(f"份 {group} 包含多个场景标签，请统一 scene 或拆成不同份")
        count = len(row["bboxes"])
        item["images"] += 1
        item["instances"] += count
        item["negative"] += count == 0
        item["multi"] += count >= 3
        for tag in row.get("tags", []):
            item["tags"][tag] = item["tags"].get(tag, 0) + 1
    return {group: GroupStats(v["images"], v["instances"], v["negative"], v["multi"],
                              dict(sorted(v["tags"].items())), v["scene"])
            for group, v in acc.items()}


def _fixed_mapping(groups, splits):
    if not isinstance(splits, dict) or set(splits) - {"train", "val", "test"}:
        raise ValueError("固定划分必须是仅含 train/val/test 的映射")
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("固定划分的 train 和 val 都必须有采集组")
    mapping = {}
    for split, members in splits.items():
        if not isinstance(members, list) or any(not isinstance(g, str) for g in members):
            raise ValueError(f"{split} 必须是份 ID 的字符串列表")
        for group in members:
            if group in mapping:
                raise ValueError(f"份重复分配: {group}")
            mapping[group] = split
    if set(mapping) != set(groups):
        raise ValueError(f"固定划分与当前份不一致；遗漏 {sorted(set(groups) - set(mapping))}，"
                         f"多余 {sorted(set(mapping) - set(groups))}")
    return mapping


def assign_groups(groups, val_ratio=.2, test_ratio=0, seed=42, splits=None, *, group_stats=None):
    """shuffled_groups_v1：按 seed 打乱整份，依次按图片数取 val、test，剩余为 train。"""
    groups = sorted(set(groups))
    if splits is not None:
        return _fixed_mapping(groups, splits)
    if not (math.isfinite(val_ratio) and math.isfinite(test_ratio)
            and 0 < val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val-ratio 必须大于 0，test-ratio 不小于 0，二者之和小于 1，且为有限数值")
    ratios = {"train": 1 - val_ratio - test_ratio, "val": val_ratio}
    if test_ratio > 0:
        ratios["test"] = test_ratio
    if len(groups) < len(ratios):
        raise ValueError(f"独立采集组不足：至少需要 {len(ratios)} 组，当前只有 {len(groups)} 组；请补采独立场景，不能按帧拆分同一组")
    images = {g: group_stats[g].images if group_stats else 1 for g in groups}
    total = sum(images.values())
    order = list(groups)
    random.Random(seed).shuffle(order)
    mapping, index = {}, 0
    for split in [name for name in ("val", "test") if name in ratios]:
        target, size, chosen = ratios[split] * total, 0, []
        while index < len(order) - 1:
            group = order[index]
            if chosen and not abs(size + images[group] - target) < abs(size - target):
                break
            chosen.append(group)
            size += images[group]
            index += 1
        if not chosen:
            raise ValueError(f"无法为 {split} 留出采集组，请补充独立组")
        mapping.update(dict.fromkeys(chosen, split))
    mapping.update(dict.fromkeys(order[index:], "train"))
    return mapping


def summarize_splits(records, assignment, ratios, *, excluded_count=0):
    """返回 (统计, warnings)。ratios 为 None 时（固定映射）不报告比例偏差。"""
    records = list(records)
    total = len(records)
    per, scenes = {}, {}
    for row in records:
        split = assignment[row["split_group_id"]]
        item = per.setdefault(split, {"images": 0, "groups": set(), "instances": 0,
                                      "negative": 0, "multi": 0, "tags": {}})
        count = len(row["bboxes"])
        item["images"] += 1
        item["groups"].add(row["split_group_id"])
        item["instances"] += count
        item["negative"] += count == 0
        item["multi"] += count >= 3
        for tag in row.get("tags", []):
            item["tags"][tag] = item["tags"].get(tag, 0) + 1
        scenes.setdefault(row["scene"], set()).add(split)
    empty = {"images": 0, "groups": set(), "instances": 0, "negative": 0, "multi": 0, "tags": {}}
    stats = {"total_images": total, "excluded_images": excluded_count, "splits": {}}
    for split, item in sorted(per.items()):
        stats["splits"][split] = {**item, "groups": len(item["groups"]), "tags": dict(sorted(item["tags"].items())),
                                  "ratio": item["images"] / total if total else 0}
    warnings = []
    if ratios is not None:
        for split, target in ratios.items():
            actual = per.get(split, empty)["images"] / total
            if abs(actual - target) > RATIO_TOLERANCE:
                warnings.append({"code": "ratio_deviation", "split": split,
                                 "target": target, "actual": actual})
    global_negative = sum(item["negative"] for item in per.values())
    global_multi = sum(item["multi"] for item in per.values())
    for split in ("train", "val"):
        item = per.get(split, empty)
        if global_negative and item["negative"] == 0:
            warnings.append({"code": "missing_feature", "split": split, "feature": "negative"})
        if global_multi and item["multi"] == 0:
            warnings.append({"code": "missing_feature", "split": split, "feature": "multi"})
    for scene, members in sorted(scenes.items()):
        if len(members) > 1:
            warnings.append({"code": "shared_scene", "scene": scene, "splits": sorted(members)})
    untagged = sum(1 for row in records if not row.get("tags"))
    if untagged:
        warnings.append({"code": "tags_unknown", "count": untagged})
    if excluded_count:
        warnings.append({"code": "excluded_images", "count": excluded_count})
    return stats, warnings
