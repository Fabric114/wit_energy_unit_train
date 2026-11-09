"""Shared argument parsing; paths resolve from the project, not the caller's cwd."""

import argparse
import json

import yaml

from .annotations import render_session
from .common import project_path
from .dataset import DatasetValidationError, build_dataset, prepare_dataset


def main(action):
    descriptions = {"render": "可选：输出检测框检查图，原始 PNG/JSON 不变",
                    "build": "直接读取 X-AnyLabeling JSON：校验、按场景份划分并构建 YOLO 数据集"}
    parser = argparse.ArgumentParser(description=descriptions[action])
    parser.add_argument("--session", nargs="+" if action == "build" else None,
                        required=action == "render", help="采集 session；构建时省略则自动发现所有 session")
    parser.add_argument("--raw-root", default="raw")
    parser.add_argument("--output", required=action == "render", help="全新输出目录，不能与源目录重叠")
    if action == "build":
        parser.add_argument("--check-only", action="store_true", help="只读校验和划分检查，不生成文件")
        parser.add_argument("--val-ratio", type=float, default=.2, help="验证集占比（按图片数整份分配），默认 0.2")
        parser.add_argument("--test-ratio", type=float, default=0, help="测试集占比，默认 0；大于 0 时至少需要 3 份")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--split-file", help="可选 YAML：train/val/test 对应份 ID 的完整映射，优先于比例")
        parser.add_argument("--group-file", help="可选 YAML：显式定义场景份；须 confirmed: true 才能构建")
    args = parser.parse_args()
    if action == "build" and (args.check_only == bool(args.output)):
        parser.error("请二选一：--check-only 或 --output <新目录>")
    try:
        raw_root = project_path(args.raw_root)
        if action == "render":
            result = render_session(raw_root, args.session, project_path(args.output))
        else:
            options = {"sessions": args.session, "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
                       "seed": args.seed,
                       "split_file": project_path(args.split_file) if args.split_file else None,
                       "group_file": project_path(args.group_file) if args.group_file else None}
            if args.check_only:
                prepared = prepare_dataset(raw_root, **options)
                publishable = prepared.group_definition["confirmed"]
                result = {**compact_report(prepared.report), "group_splits": prepared.assignment,
                          "split_stats": prepared.split_stats, "warnings": prepared.warnings,
                          "publishable": publishable}
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if not publishable:
                    parser.exit(2, "分组文件尚未 confirmed: true，结论为不可发布\n")
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
