#!/usr/bin/env python3
"""校验 GUI 标注并一次构建 YOLO 数据集。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_train.annotation_cli import main

if __name__ == "__main__":
    main("build")
