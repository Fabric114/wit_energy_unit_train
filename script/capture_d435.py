#!/usr/bin/env python3
"""Collect raw color frames without changing other camera processes."""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_train.common import project_path


def main():
    parser = argparse.ArgumentParser(description="D435 原始彩色采集；空格保存，a 暂停/自动，q 退出")
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--session", required=True, help="场景名，例如 s01_near_front")
    parser.add_argument("--group-id", required=True, help="同一连续拍摄过程共享一个来源组")
    for name in ("output-root", "serial", "note"):
        parser.add_argument("--" + name)
    for name in ("width", "height", "fps", "warmup-frames", "max-images"):
        parser.add_argument("--" + name, type=int)
    for name in ("interval", "duration", "blur-threshold", "duplicate-threshold"):
        parser.add_argument("--" + name, type=float)
    for name in ("preview", "auto", "reject-blur"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    try:
        import yaml
        from energy_train.capture import CaptureConfig, run_capture
        config_path = project_path(args.pop("config"))
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("配置必须为 YAML 键值表")
        unknown = set(values) - {field.name for field in fields(CaptureConfig)}
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        values.update({k: v for k, v in args.items() if v is not None})
        values["output_root"] = project_path(values.get("output_root", "raw"))
        print(json.dumps(run_capture(CaptureConfig(**values)), ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"采集失败: {exc}\n")


if __name__ == "__main__":
    main()
