"""Shared command line for the camera capture scripts.

configs/capture.yaml holds one section per camera (d435, hik). Each script
reads only its own section; CLI options override that section.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .common import project_path

FIXED_ARGS = {"session", "group_id"}


def load_section(path, section, config_cls):
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("配置必须为 YAML 键值表")
    values = document.get(section)
    if not isinstance(values, dict):
        raise ValueError(f"配置缺少相机段 {section}:，现有段: {sorted(document)}")
    unknown = set(values) - {field.name for field in fields(config_cls)} - FIXED_ARGS
    if unknown:
        raise ValueError(f"{section} 段有未知配置项: {sorted(unknown)}")
    return dict(values)


def build_parser(description, config_cls):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--session", required=True, help="场景名，例如 s01_near_front")
    parser.add_argument("--group-id", required=True, help="同一连续拍摄过程共享一个来源组")
    for field in fields(config_cls):
        if field.name in FIXED_ARGS:
            continue
        option = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        elif field.type in (int, float):
            parser.add_argument(option, type=field.type)
        else:
            parser.add_argument(option)
    return parser


def main(section, config_cls, make_source, description, argv=None):
    parser = build_parser(description, config_cls)
    args = vars(parser.parse_args(argv))
    try:
        from .capture import run_capture

        values = load_section(project_path(args.pop("config")), section, config_cls)
        values.update({k: v for k, v in args.items() if v is not None})
        values["output_root"] = project_path(values.get("output_root", "raw"))
        config = config_cls(**values)
        source = make_source(config) if make_source else None
        print(json.dumps(run_capture(config, source=source), ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"采集失败: {exc}\n")
