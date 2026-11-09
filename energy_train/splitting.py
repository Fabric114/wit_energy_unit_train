"""场景分组：解析可选 group-file（份、范围展开、排除与标签）并校验完整性。

份（split group）是不可拆分的划分单位。group-file 只引用现有 PNG，不移动数据。
"""

import hashlib
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
