"""用 OpenVINO YOLO 模型为 raw/<session>/ 中的 PNG 生成 X-AnyLabeling 预标注 JSON。

规则：
- 只处理 raw/<session>/ 直接包含的 PNG，跳过 _delete_ 目录。
- 已存在同名 JSON 的图片一律跳过，不覆盖任何已有标注（包括人工标注）。
- 预标注 JSON 写入 checked=false，review_status=pending。人工检查后再在 X-AnyLabeling 中标记通过。
- 框为矩形，标签统一为 energy_unit。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "raw"
DEFAULT_MODEL = PROJECT_ROOT / "reference_model" / "best_openvino_model" / "best.xml"
LABEL = "energy_unit"
INPUT_SIZE = 640
PAD_VALUE = 114


def letterbox(image_rgb: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """等比缩放并居中填充到 640x640，返回 (输入张量, 缩放比, 左填充, 上填充)。"""
    h, w = image_rgb.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale, pad_x, pad_y


def detect(compiled, image_bgr: np.ndarray, conf: float, iou: float) -> list[dict]:
    """返回原图坐标系中的检测框列表：[{x1,y1,x2,y2,score}]。"""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor, scale, pad_x, pad_y = letterbox(rgb)
    pred = compiled(tensor)[compiled.output(0)][0]  # 形状 (5, 8400)：cx, cy, w, h, score
    scores = pred[4]
    keep = scores >= conf
    if not keep.any():
        return []
    cx, cy, bw, bh, sc = (pred[i][keep] for i in range(5))
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale
    x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
    y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)

    boxes_xywh = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, sc.tolist(), conf, iou)
    results = []
    for i in np.array(indices).flatten():
        a, b, c, d = float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])
        if c - a < 1 or d - b < 1:  # 跳过退化框
            continue
        results.append({"x1": a, "y1": b, "x2": c, "y2": d, "score": float(sc[i])})
    return results


def build_json(png: Path, image_bgr: np.ndarray, dets: list[dict]) -> dict:
    h, w = image_bgr.shape[:2]
    shapes = []
    for d in dets:
        shapes.append({
            "review_status": "pending",
            "label": LABEL,
            "score": round(d["score"], 4),
            "points": [[round(d["x1"], 2), round(d["y1"], 2)], [round(d["x2"], 2), round(d["y2"], 2)]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })
    return {
        "version": "4.3.1",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": png.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "description": "",
    }


def process_session(compiled, session_dir: Path, conf: float, iou: float) -> dict:
    stats = {"written": 0, "skipped_existing": 0, "unreadable": 0, "boxes": 0}
    for png in sorted(session_dir.glob("*.png")):
        out = png.with_suffix(".json")
        if out.exists():
            stats["skipped_existing"] += 1
            continue
        image = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if image is None:
            stats["unreadable"] += 1
            continue
        dets = detect(compiled, image, conf, iou)
        data = build_json(png, image, dets)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)  # 原子替换，避免写到一半留下坏文件
        stats["written"] += 1
        stats["boxes"] += len(dets)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 X-AnyLabeling 预标注 JSON（不覆盖已有标注）")
    parser.add_argument("--session", nargs="*", help="要处理的 session，如 s02；默认处理 raw/ 下全部 session")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    args = parser.parse_args()

    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(args.model)), "CPU")

    if args.session:
        sessions = [args.raw_root / s for s in args.session]
    else:
        sessions = sorted(p for p in args.raw_root.iterdir()
                          if p.is_dir() and p.name != "_delete_" and not p.name.startswith("."))

    for session_dir in sessions:
        if not session_dir.is_dir():
            print(f"[跳过] 目录不存在：{session_dir}")
            continue
        s = process_session(compiled, session_dir, args.conf, args.iou)
        print(f"{session_dir.name}: 新写入 {s['written']} 个 JSON，共 {s['boxes']} 个框；"
              f"跳过已有 {s['skipped_existing']} 个；无法读取 {s['unreadable']} 个")


if __name__ == "__main__":
    main()
