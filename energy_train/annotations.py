"""X-AnyLabeling 检测框解析、原生审核和来源快照。"""

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common import json_text, new_output_directory, sha256, valid_name


def session_inputs(raw_root, session):
    folder = Path(raw_root).resolve() / valid_name(session)
    payload = (folder / "meta.json").read_bytes()
    meta = json.loads(payload)
    if not isinstance(meta, dict) or meta.get("session") != session:
        raise ValueError("meta.json 的 session 不匹配")
    valid_name(meta.get("group_id"))
    camera = meta.get("camera", {})
    if not isinstance(camera, dict) or any(
        type(camera.get(key)) is not int or camera[key] <= 0 for key in ("width", "height")
    ):
        raise ValueError("meta.json 缺少有效的相机宽高")
    images = sorted(folder.glob("*.png"))
    if not images:
        raise ValueError(f"session 没有 PNG 图片: {folder}")
    return meta, images, hashlib.sha256(payload).hexdigest()


def annotation_status(document):
    """使用 X-AnyLabeling 的整图状态，不把 truthy 值当成人工通过。"""
    value = document.get("checked", False)
    if value is True:
        return "passed"
    if value is False:
        return "pending"
    if value == "failed":
        return "failed"
    raise ValueError('checked 必须为 true、false 或 "failed"')


def read_image_snapshot(path, meta):
    payload = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED) if payload else None
    expected = (meta["camera"]["height"], meta["camera"]["width"], 3)
    if image is None or image.dtype != np.uint8 or image.shape != expected:
        raise ValueError(f"图片损坏或尺寸/通道不是 {expected} uint8 BGR: {path}")
    return image, hashlib.sha256(payload).hexdigest()


def read_annotation_snapshot(path, label, meta):
    """Hash precisely the bytes we decode, not a subsequent GUI save."""
    image, image_digest = read_image_snapshot(path, meta)
    payload = label.read_bytes()
    doc = json.loads(payload)
    if not isinstance(doc, dict) or not isinstance(doc.get("imagePath"), str):
        raise ValueError(f"imagePath 缺失或格式无效: {label}")
    linked = Path(doc["imagePath"])
    # A basename uses the explicitly supplied raw session; all other paths must resolve there.
    resolved = path.parent / linked if linked.parent == Path(".") else label.parent / linked
    if resolved.resolve() != path.resolve():
        raise ValueError(f"imagePath 指向另一张图片: {label}")
    if doc.get("imageData") is not None:
        try:
            embedded = base64.b64decode(doc["imageData"], validate=True)
            pixels = cv2.imdecode(np.frombuffer(embedded, np.uint8), cv2.IMREAD_UNCHANGED) if embedded else None
        except (TypeError, ValueError, binascii.Error, cv2.error) as exc:
            raise ValueError(f"无效的 imageData 内嵌图像: {label}") from exc
        if pixels is None or pixels.dtype != image.dtype or not np.array_equal(pixels, image):
            raise ValueError(f"imageData 内嵌图像与 raw 原图不一致: {label}")
    return image, doc, {"image_sha256": image_digest,
                        "annotation_sha256": hashlib.sha256(payload).hexdigest()}


def ensure_unchanged(path, label, snapshot):
    annotation_digest = sha256(label) if label.exists() else None
    if sha256(path) != snapshot["image_sha256"] or annotation_digest != snapshot["annotation_sha256"]:
        raise ValueError(f"图片或 JSON 在操作期间被修改，请保存后重新运行: {path.name}")


def ensure_session_unchanged(raw_root, session, image_names, meta_digest):
    _, images, current_digest = session_inputs(raw_root, session)
    if [p.name for p in images] != image_names or current_digest != meta_digest:
        raise ValueError(f"session 图片清单或 meta.json 在操作期间发生变化: {session}")


def session_folder(raw_root, session):
    """GUI 标注与 PNG 同目录，不存在独立标注根。"""
    return Path(raw_root).resolve() / valid_name(session)


def check_output_location(output, raw_root):
    output = Path(output).resolve()
    root = Path(raw_root).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("输出目录不能与原图目录重叠")


def load_instances(document, image_shape, image_name):
    if not isinstance(document, dict):
        raise ValueError("标注 JSON 必须是对象")
    height, width = image_shape[:2]
    if document.get("imageWidth") != width or document.get("imageHeight") != height:
        raise ValueError(f"标注尺寸和图片不一致: {image_name}")
    linked_name = str(document.get("imagePath", "")).replace("\\", "/").split("/")[-1]
    if linked_name != image_name:
        raise ValueError(f"imagePath 对应的图片名不匹配: {image_name}")
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("shapes 必须是数组")
    instances, seen = [], set()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict) or shape.get("label") != "energy_unit":
            raise ValueError(f"shape {index}: 类别必须是 energy_unit")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"shape {index}: 只接受 rectangle 检测框")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) not in (2, 4):
            raise ValueError(f"shape {index}: 矩形必须有 2 个对角点或 4 个角点")
        clean, clipped = [], False
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"shape {index}: 每个点必须为 [x,y]")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError(f"shape {index}: 坐标必须是有限数值")
            x, y = point
            if not 0 <= x <= width or not 0 <= y <= height:
                if not (-1 <= x <= width + 1 and -1 <= y <= height + 1):
                    raise ValueError(f"shape {index}: 坐标超出图片边界（矩形仅容许 1 像素以内偏差）")
                x, y = max(0, min(x, width)), max(0, min(y, height))
                clipped = True
            clean.append([float(x), float(y)])
        x1, x2 = min(p[0] for p in clean), max(p[0] for p in clean)
        y1, y2 = min(p[1] for p in clean), max(p[1] for p in clean)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"shape {index}: 矩形宽高必须大于零")
        if len(clean) == 4 and set(map(tuple, clean)) != {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}:
            raise ValueError(f"shape {index}: 4 个角点必须构成水平矩形")
        if (x2 - x1) * (y2 - y1) <= 1e-6:
            raise ValueError(f"shape {index}: 标注面积为零或过小")
        if (x1, y1, x2, y2) in seen:
            raise ValueError(f"shape {index}: 重复标注，请删除重复的框")
        seen.add((x1, y1, x2, y2))
        # Each rectangle is one object; the GUI group_id field is ignored.
        instances.append({"class_id": 0, "bbox": [x1, y1, x2, y2], "clipped": clipped})
    return instances


def render_session(raw_root, session, output):
    """可选叠图；允许预览待检查标注，不创建或更新审核文件。"""
    check_output_location(output, raw_root)
    meta, paths, meta_digest = session_inputs(raw_root, session)
    folder = session_folder(raw_root, session)
    rows, snapshots = [], []
    with new_output_directory(output) as staging:
        for path in paths:
            label = folder / (path.stem + ".json")
            image, doc, snapshot = read_annotation_snapshot(path, label, meta)
            instances = load_instances(doc, image.shape, path.name)
            status = annotation_status(doc)
            for index, instance in enumerate(instances):
                x1, y1, x2, y2 = [int(round(v)) for v in instance["bbox"]]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 160, 255), 2)
                cv2.putText(image, f"unit {index}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 160, 255), 1)
            cv2.putText(image, f"{status} | units={len(instances)}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            if not cv2.imwrite(str(staging / path.name), image):
                raise OSError(f"叠图保存失败: {path.name}")
            rows.append({"image": path.name, "status": status, "instances": len(instances),
                         "clipped_boxes": sum(i["clipped"] for i in instances)})
            snapshots.append((path, label, snapshot))
        for path, label, snapshot in snapshots:
            ensure_unchanged(path, label, snapshot)
        ensure_session_unchanged(raw_root, session, [p.name for p in paths], meta_digest)
        (staging / "index.json").write_text(json_text(rows), encoding="utf-8")
    return {"images": len(rows), "output": str(output)}


