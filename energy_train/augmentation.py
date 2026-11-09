"""light_v1 离线增强：固定参数、单次变换与候选检查。

本模块不读写文件。Albumentations 只在函数体内导入，导入本模块不加载 albumentations、torch 或 ultralytics。
"""

import hashlib
import json
import math
import types

import cv2
import numpy as np

RECIPE_NAME = "light_v1"
MAX_ATTEMPTS = 10
CLASS_ID = 0
EDGE_EPSILON = 1e-6


def _freeze(value):
    if isinstance(value, dict):
        return types.MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, (dict, types.MappingProxyType)):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


# 设计第 6.3 节的全部参数。只读；修改任何参数必须同时更换 RECIPE_NAME。
LIGHT_V1 = _freeze({
    "HorizontalFlip": {"p": 0.5},
    "Affine": {
        "p": 0.7,
        "scale": (0.9, 1.1),
        "keep_ratio": True,
        "translate_percent": (-0.05, 0.05),
        "rotate": (-5, 5),
        "shear": (0, 0),
        "fill": 114,
        "border_mode": cv2.BORDER_CONSTANT,
    },
    "RandomBrightnessContrast": {
        "p": 0.5,
        "brightness_limit": (-0.15, 0.15),
        "contrast_limit": (-0.15, 0.15),
    },
    "GaussNoise": {"p": 0.15, "std_range": (0.01, 0.03), "mean_range": (0, 0)},
    "Blur": {"p": 0.2, "blur_limit": (3, 5), "sigma_limit": (0.5, 1.0)},
    "BboxParams": {"format": "yolo", "min_visibility": 0.7, "min_area": 16, "clip": True},
})


def recipe_params():
    """返回 LIGHT_V1 的可修改普通副本，可直接 JSON 序列化写入 summary。"""
    return _thaw(LIGHT_V1)


class CandidateRejected(Exception):
    """预期的候选不合格：code 为 dropped_box、invalid_box、unchanged_image 之一。"""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class AugmentSample:
    def __init__(self, image_bgr, boxes_yolo, class_ids, instance_ids):
        self.image_bgr = image_bgr
        self.boxes_yolo = boxes_yolo
        self.class_ids = class_ids
        self.instance_ids = instance_ids


def derive_sample_seed(seed, parent_image, copy_index, attempt):
    """对 (seed, 父图相对路径, copy_index, attempt) 的 UTF-8 紧凑 JSON 数组取 SHA256 前 4 字节（big-endian）。"""
    payload = json.dumps([seed, parent_image, copy_index, attempt],
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_transform(sample_seed, *, recipe=None):
    """构建 Compose；所有参数显式传入，recipe 仅供测试注入。"""
    import albumentations as A

    params = LIGHT_V1 if recipe is None else recipe
    affine, blur, bbox = params["Affine"], params["Blur"], params["BboxParams"]
    brightness = params["RandomBrightnessContrast"]
    noise = params["GaussNoise"]
    transforms = [
        A.HorizontalFlip(p=params["HorizontalFlip"]["p"]),
        A.Affine(
            scale=affine["scale"], keep_ratio=affine["keep_ratio"],
            translate_percent=affine["translate_percent"], rotate=affine["rotate"],
            shear=affine["shear"], fill=affine["fill"], fill_mask=0,
            border_mode=affine["border_mode"], interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST, fit_output=False,
            rotate_method="largest_box", balanced_scale=False, p=affine["p"],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness["brightness_limit"],
            contrast_limit=brightness["contrast_limit"],
            brightness_by_max=True, ensure_safe_range=False, p=brightness["p"],
        ),
        A.GaussNoise(
            std_range=noise["std_range"], mean_range=noise["mean_range"],
            per_channel=True, noise_scale_factor=1, p=noise["p"],
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=blur["blur_limit"], allow_shifted=True,
                         angle_range=(0, 360), direction_range=(-1.0, 1.0), p=1.0),
            A.GaussianBlur(blur_limit=blur["blur_limit"], sigma_limit=blur["sigma_limit"], p=1.0),
        ], p=blur["p"]),
    ]
    bbox_params = A.BboxParams(
        format=bbox["format"], label_fields=["class_ids", "instance_ids"],
        min_visibility=bbox["min_visibility"], min_area=bbox["min_area"], clip=bbox["clip"],
    )
    return A.Compose(transforms, bbox_params=bbox_params, seed=sample_seed)


def _check_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("增强输入必须是 BGR uint8 三通道数组")
    return image


def _valid_yolo_box(values):
    xc, yc, width, height = values
    if not all(math.isfinite(v) for v in values) or width <= 0 or height <= 0:
        return False
    return (xc - width / 2 >= -EDGE_EPSILON and xc + width / 2 <= 1 + EDGE_EPSILON
            and yc - height / 2 >= -EDGE_EPSILON and yc + height / 2 <= 1 + EDGE_EPSILON)


def _check_input_boxes(boxes):
    checked = []
    for box in boxes:
        if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
            raise CandidateRejected("invalid_box")
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            raise CandidateRejected("invalid_box")
        if not _valid_yolo_box(values):
            raise CandidateRejected("invalid_box")
        checked.append(values)
    return checked


def augment_one(image_bgr, boxes_yolo, *, sample_seed, recipe=None):
    """单次尝试：BGR→RGB 变换后转回 BGR，检查框完整性、几何有效性与像素是否变化。不重试、不做文件 I/O。"""
    image = _check_image(image_bgr)
    boxes = _check_input_boxes(boxes_yolo)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ids = list(range(len(boxes)))
    transform = make_transform(sample_seed, recipe=recipe)
    out = transform(image=rgb, bboxes=boxes, class_ids=[CLASS_ID] * len(boxes), instance_ids=ids)

    returned_ids = [int(value) for value in out["instance_ids"]]
    if len(out["bboxes"]) != len(boxes) or sorted(returned_ids) != ids:
        raise CandidateRejected("dropped_box")
    pairs = sorted(zip(returned_ids, out["bboxes"], out["class_ids"]), key=lambda item: item[0])
    out_boxes, out_classes = [], []
    for _, box, class_id in pairs:
        values = [float(v) for v in box]
        if int(class_id) != CLASS_ID or not _valid_yolo_box(values):
            raise CandidateRejected("invalid_box")
        out_boxes.append(values)
        out_classes.append(CLASS_ID)
    out_rgb = out["image"]
    if out_rgb.shape != rgb.shape or np.array_equal(out_rgb, rgb):
        raise CandidateRejected("unchanged_image")
    return AugmentSample(
        image_bgr=cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR),
        boxes_yolo=out_boxes,
        class_ids=out_classes,
        instance_ids=[item[0] for item in pairs],
    )
