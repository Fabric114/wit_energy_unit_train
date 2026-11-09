#!/usr/bin/env python3
"""按时间断点生成分组草稿（confirmed: false），只读 raw。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_train.group_proposal import main

if __name__ == "__main__":
    raise SystemExit(main())
