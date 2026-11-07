#!/usr/bin/env python3
"""用校验过的 build_dataset 数据集训练 YOLO 检测模型，并记录 provenance。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_train.training import main

if __name__ == "__main__":
    sys.exit(main())
