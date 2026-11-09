#!/usr/bin/env python3
"""为已划分的快照生成 light_v1 离线增强副本（只增强 train），并发布完整派生快照。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_train.augmentation_dataset import main

if __name__ == "__main__":
    sys.exit(main())
