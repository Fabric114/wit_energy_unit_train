#!/usr/bin/env python3
"""Read-only dependency and optional camera inventory; never starts streams."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description="采集/标注环境检查；不加载训练框架")
    parser.add_argument("--camera", action="store_true", help="枚举相机，不启动数据流")
    args = parser.parse_args()
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": {}}
    missing = False
    for package, module in (("numpy", "numpy"), ("opencv-python", "cv2"), ("PyYAML", "yaml"), ("pyrealsense2", "pyrealsense2")):
        try:
            version = importlib.metadata.version(package)
            spec = importlib.util.find_spec(module)
            report["packages"][package] = {"version": version, "path": spec.origin if spec else None}
            missing |= spec is None
        except (ImportError, ValueError):
            report["packages"][package] = {"error": "not installed"}
            missing = True
    if args.camera:
        try:
            import pyrealsense2 as rs
            report["cameras"] = [{"name": d.get_info(rs.camera_info.name), "serial": d.get_info(rs.camera_info.serial_number)}
                                 for d in rs.context().query_devices()]
            if not report["cameras"]:
                report["camera_note"] = "未发现设备；检查连接/USB 权限。没有启动相机。"
                missing = True
        except (ImportError, RuntimeError) as exc:
            report["camera_error"] = str(exc)
            missing = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
