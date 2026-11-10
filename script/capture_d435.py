#!/usr/bin/env python3
"""Collect raw color frames without changing other camera processes (reads the d435 section of capture.yaml)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_train.capture import CaptureConfig
from energy_train.capture_cli import main

if __name__ == "__main__":
    main("d435", CaptureConfig, None, "D435 原始彩色采集；空格保存，a 暂停/自动，q 退出")
