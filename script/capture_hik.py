#!/usr/bin/env python3
"""Collect raw color frames from a Hikrobot MVS camera (reads the hik section of capture.yaml)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_train.capture_cli import main
from energy_train.capture_hik import HikCaptureConfig, HikColorSource

if __name__ == "__main__":
    main("hik", HikCaptureConfig, HikColorSource, "海康相机原始彩色采集；空格保存，a 暂停/自动，q 退出")
