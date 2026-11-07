"""Shared argument parsing; paths resolve from the project, not the caller's cwd."""

import argparse
import json

import yaml

from .annotations import render_session
from .common import project_path
from .dataset import DatasetValidationError, assign_groups, build_dataset, inspect_dataset


def main(action):
    descriptions = {"render": "可选：输出检测框检查图，原始 PNG/JSON 不变",
                    "build": "直接读取 X-AnyLabeling JSON：校验、按采集组划分并构建 YOLO 数据集"}
    parser = argparse.ArgumentParser(description=descriptions[action])
    parser.add_argument("--session", nargs="+" if action == "build" else None,
                        required=action == "render", help="采集 session；构建时省略则自动发现所有 session")
    parser.add_argument("--raw-root", default="raw")
    parser.add_argument("--output", required=action == "render", help="全新输出目录，不能与源目录重叠")
    if action == "build":
        parser.add_argument("--check-only", action="store_true", help="只读校验和划分可行性检查，不生成文件")
        parser.add_argument("--val-ratio", type=float, default=.2, help="验证组占比，默认 0.2（按组数近似）")
        parser.add_argument("--test-ratio", type=float, default=0, help="测试组占比，默认 0；大于 0 时至少需要 3 个独立组")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--split-file", help="可选 YAML：train/val/test 对应 group_id 列表，固定分配优先于比例")
    args = parser.parse_args()
    if action == "build" and (args.check_only == bool(args.output)):
        parser.error("请二选一：--check-only 或 --output <新目录>")
    try:
        raw_root = project_path(args.raw_root)
        if action == "render":
            result = render_session(raw_root, args.session, project_path(args.output))
        else:
            splits = None
            if args.split_file:
                splits = yaml.safe_load(project_path(args.split_file).read_text(encoding="utf-8"))
                if splits is None:
                    raise ValueError("固定划分 YAML 不能为空")
            options = {"sessions": args.session,
                       "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "seed": args.seed, "splits": splits}
            if args.check_only:
                report = inspect_dataset(raw_root, args.session)
                result = compact_report(report)
                try:
                    result["group_splits"] = assign_groups(report["group_ids"], args.val_ratio, args.test_ratio, args.seed, splits)
                except ValueError as exc:
                    result["split_error"] = str(exc)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if report["issues"] or "split_error" in result:
                    parser.exit(2)
                return
            result = build_dataset(raw_root, project_path(args.output), **options)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except DatasetValidationError as exc:
        print(json.dumps(compact_report(exc.report), ensure_ascii=False, indent=2))
        parser.exit(2, f"操作失败: {exc}\n")
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, yaml.YAMLError) as exc:
        parser.exit(2, f"操作失败: {exc}\n")


def compact_report(report):
    return {"summary": report["summary"], "group_ids": report["group_ids"],
            "issues": report["issues"][:20], "remaining_issues": max(0, len(report["issues"]) - 20)}


if __name__ == "__main__":
    raise SystemExit("请使用 script/ 中的入口脚本")
