"""按时间断点生成分组草稿。只读 raw；输出 confirmed: false，必须人工核对后才能使用。

采集启动边界用 frames.jsonl 的 saved_at 与 meta.json 的 capture_runs[].started_at 比较（同为 UTC 墙钟）；
间隔与时长用 timestamp_ms。两者不在同一时间域，不能混用。
"""

import argparse
import bisect
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .common import project_path
from .dataset import discover_sessions

SHORT_GAP_FACTOR = 5


@dataclass(frozen=True)
class ProposedGroup:
    id: str
    scene: str
    images: tuple
    start: str
    end: str
    duration_s: float
    gap_before_s: "float | None"
    short_gap: bool


@dataclass(frozen=True)
class GroupProposal:
    groups: tuple
    excluded: tuple
    sessions: tuple
    params: dict


def _check_params(gap_seconds, max_group_size, buffer):
    if isinstance(gap_seconds, bool) or not isinstance(gap_seconds, (int, float)) \
            or not math.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ValueError("gap-seconds 必须是大于 0 的有限数值")
    if isinstance(buffer, bool) or not isinstance(buffer, int) or buffer < 0:
        raise ValueError("buffer 必须是不小于 0 的整数")
    if max_group_size is not None and (isinstance(max_group_size, bool)
                                       or not isinstance(max_group_size, int)
                                       or max_group_size <= 2 * buffer):
        raise ValueError("max-group-size 必须是整数，且大于 2×buffer")


def _session_frames(folder):
    """返回 [(timestamp_ms, 文件名, 启动序号)]，按时间排序；只包含磁盘上现存的 PNG。"""
    pngs = sorted(p.name for p in folder.glob("*.png"))
    records = {}
    log = folder / "frames.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("image") in records:
                raise ValueError(f"{folder.name}: frames.jsonl 中图片记录重复: {record.get('image')}")
            records[record.get("image")] = record
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"{folder.name} 缺少 meta.json，无法读取采集启动记录")
    runs = sorted(datetime.fromisoformat(run["started_at"])
                  for run in json.loads(meta_path.read_text(encoding="utf-8")).get("capture_runs", []))
    rows = []
    for name in pngs:
        if name not in records:
            raise ValueError(f"{folder.name}: 图片 {name} 在 frames.jsonl 中缺少记录，不猜测时间")
        record = records[name]
        stamp = record.get("timestamp_ms")
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise ValueError(f"{name} 的 timestamp_ms 不是有限数值")
        saved = datetime.fromisoformat(record["saved_at"])
        rows.append((float(stamp), name, bisect.bisect_right(runs, saved)))
    return sorted(rows)


def _segments(rows, gap_seconds):
    """相邻间隔超过 gap_seconds，或跨越一次新的采集启动，即开始新段。"""
    if not rows:
        return []
    segments, current = [], [rows[0]]
    for previous, row in zip(rows, rows[1:]):
        if (row[0] - previous[0]) / 1000 > gap_seconds or row[2] != previous[2]:
            segments.append(current)
            current = []
        current.append(row)
    segments.append(current)
    return segments


def _subdivide(rows, max_size, buffer):
    """长场景按时间均分为不超过 max_size 的块；切点两侧各 buffer 张移入缓冲（不参与分组）。"""
    if max_size is None or len(rows) <= max_size:
        return [rows], []
    count = math.ceil(len(rows) / max_size)
    base, extra = divmod(len(rows), count)
    pieces, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        pieces.append(rows[start:start + size])
        start += size
    kept, dropped = [], []
    for index, piece in enumerate(pieces):
        left = buffer if index > 0 else 0
        right = buffer if index < len(pieces) - 1 else 0
        dropped += piece[:left] + piece[len(piece) - right:]
        kept.append(piece[left:len(piece) - right])
    return kept, dropped


def propose_groups(raw_root, *, sessions=None, gap_seconds=60.0, max_group_size=None, buffer=3):
    _check_params(gap_seconds, max_group_size, buffer)
    root = Path(raw_root)
    names = discover_sessions(root, sessions)
    groups, excluded = [], []
    for session in names:
        previous_ts = None
        for number, segment in enumerate(_segments(_session_frames(root / session), gap_seconds), 1):
            scene = f"{session}_g{number:02d}"
            pieces, dropped = _subdivide(segment, max_group_size, buffer)
            excluded += [f"{session}/{name}" for _, name, _ in dropped]
            for index, piece in enumerate(pieces, 1):
                if not piece:
                    raise ValueError(f"{scene} 细分后出现空块，请增大 max-group-size 或减小 buffer")
                first, last = piece[0], piece[-1]
                gap = None if previous_ts is None else (first[0] - previous_ts) / 1000
                groups.append(ProposedGroup(
                    id=scene if len(pieces) == 1 else f"{scene}_p{index}", scene=scene,
                    images=tuple(f"{session}/{name}" for _, name, _ in piece),
                    start=f"{session}/{first[1]}", end=f"{session}/{last[1]}",
                    duration_s=(last[0] - first[0]) / 1000, gap_before_s=gap,
                    short_gap=gap is not None and gap_seconds <= gap <= SHORT_GAP_FACTOR * gap_seconds))
                previous_ts = last[0]
    params = {"gap_seconds": gap_seconds, "max_group_size": max_group_size, "buffer": buffer}
    return GroupProposal(groups=tuple(groups), excluded=tuple(excluded), sessions=tuple(names), params=params)


def _report_lines(group):
    line = f"# {group.id}: 起 {group.start} 止 {group.end}，{len(group.images)} 张，时长 {group.duration_s:.1f} 秒"
    if group.gap_before_s is None:
        return [line]
    note = "  【短断点，请核对是否同一场景】" if group.short_gap else ""
    return [line, f"#   与上一份间隔 {group.gap_before_s:.1f} 秒{note}"]


def render_group_file(proposal):
    """生成 confirmed: false 的 group-file 文本；报告写在 YAML 注释中。"""
    header = ["# 分组草稿：由 script/propose_groups.py 按时间断点生成，尚未人工核对。",
              "# 看图核对、合并或调整后把 confirmed 改为 true；confirmed 不是 true 时无法构建。"]
    report = [line for group in proposal.groups for line in _report_lines(group)]
    groups = []
    for group in proposal.groups:
        entry = {"id": group.id, "reason": f"时间断点草稿（{group.images[0].split('/')[0]}）",
                 "ranges": [{"from": group.images[0], "to": group.images[-1]}]}
        if group.scene != group.id:
            entry["scene"] = group.scene
        groups.append(entry)
    body = {"schema_version": 1, "confirmed": False, "groups": groups,
            "exclude": [{"reason": "细分缓冲", "images": list(proposal.excluded)}] if proposal.excluded else []}
    return "\n".join(header + report) + "\n" + yaml.safe_dump(body, allow_unicode=True, sort_keys=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="按时间断点生成分组草稿（只读 raw，输出 confirmed: false）")
    parser.add_argument("--raw-root", default="raw")
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    parser.add_argument("--max-group-size", type=int)
    parser.add_argument("--buffer", type=int, default=3)
    parser.add_argument("--output", required=True, help="新文件路径；已存在则拒绝覆盖")
    args = parser.parse_args(argv)
    try:
        output = project_path(args.output)
        if output.exists():
            raise ValueError(f"输出已存在，拒绝覆盖: {output}")
        proposal = propose_groups(project_path(args.raw_root), sessions=args.sessions,
                                  gap_seconds=args.gap_seconds, max_group_size=args.max_group_size,
                                  buffer=args.buffer)
        text = render_group_file(proposal)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (ValueError, OSError, KeyError) as exc:
        print(f"操作失败: {exc}", file=sys.stderr)
        return 2
    print(text)
    return 0
