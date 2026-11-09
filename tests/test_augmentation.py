import hashlib
import json
import subprocess
import sys
import unittest

import albumentations as A  # noqa: F401  真实库必须可导入；缺失时本模块报错，而不是 skip
import cv2
import numpy as np

from energy_train.augmentation import (
    LIGHT_V1, MAX_ATTEMPTS, RECIPE_NAME, CandidateRejected, augment_one,
    derive_sample_seed, make_transform, recipe_params,
)
from energy_train.common import PROJECT_ROOT

TRANSFORMS = ("HorizontalFlip", "Affine", "RandomBrightnessContrast", "GaussNoise", "Blur")


def asymmetric_image():
    image = np.zeros((80, 100, 3), np.uint8)
    image[:, :, 0] = np.arange(100, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(80, dtype=np.uint8)[:, None]
    image[10:30, 60:90, 2] = 255
    return image


def only(name, **params):
    """复制 LIGHT_V1，仅保留 name（p=1），其余变换 p=0；params 覆盖该变换参数。"""
    recipe = recipe_params()
    for key in TRANSFORMS:
        recipe[key]["p"] = 0.0
    recipe[name]["p"] = 1.0
    recipe[name].update(params)
    return recipe


def none():
    return only("HorizontalFlip", p=0.0)


def run(sample_seed, image_bgr, boxes, recipe=LIGHT_V1):
    """直接调用单次 Compose，返回 RGB 输出，用于验证随机状态。"""
    transform = make_transform(sample_seed, recipe=recipe)
    out = transform(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), bboxes=boxes,
                    class_ids=[0] * len(boxes), instance_ids=list(range(len(boxes))))
    return out["image"], out["bboxes"]


class RecipeTests(unittest.TestCase):
    def test_recipe_name_and_attempts(self):
        self.assertEqual(RECIPE_NAME, "light_v1")
        self.assertEqual(MAX_ATTEMPTS, 10)

    def test_light_v1_values_match_design_6_3(self):
        recipe = recipe_params()
        self.assertEqual(recipe["HorizontalFlip"]["p"], 0.5)
        affine = recipe["Affine"]
        self.assertEqual(affine["p"], 0.7)
        self.assertEqual(affine["scale"], (0.9, 1.1))
        self.assertTrue(affine["keep_ratio"])
        self.assertEqual(affine["translate_percent"], (-0.05, 0.05))
        self.assertEqual(affine["rotate"], (-5, 5))
        self.assertEqual(affine["shear"], (0, 0))
        self.assertEqual(affine["fill"], 114)
        self.assertEqual(affine["border_mode"], cv2.BORDER_CONSTANT)
        self.assertEqual(recipe["RandomBrightnessContrast"]["p"], 0.5)
        self.assertEqual(recipe["RandomBrightnessContrast"]["brightness_limit"], (-0.15, 0.15))
        self.assertEqual(recipe["RandomBrightnessContrast"]["contrast_limit"], (-0.15, 0.15))
        self.assertEqual(recipe["GaussNoise"]["p"], 0.15)
        self.assertEqual(recipe["GaussNoise"]["std_range"], (0.01, 0.03))
        self.assertEqual(recipe["GaussNoise"]["mean_range"], (0, 0))
        self.assertEqual(recipe["Blur"]["p"], 0.2)
        self.assertEqual(recipe["Blur"]["blur_limit"], (3, 5))
        self.assertEqual(recipe["Blur"]["sigma_limit"], (0.5, 1.0))
        self.assertEqual(recipe["BboxParams"], {"format": "yolo", "min_visibility": 0.7,
                                              "min_area": 16, "clip": True})

    def test_recipe_params_is_a_copy(self):
        first = recipe_params()
        first["HorizontalFlip"]["p"] = 0.0
        first["Affine"]["fill"] = 0
        self.assertEqual(recipe_params()["HorizontalFlip"]["p"], 0.5)
        self.assertEqual(recipe_params()["Affine"]["fill"], 114)
        self.assertEqual(LIGHT_V1["HorizontalFlip"]["p"], 0.5)

    def test_recipe_params_is_json_serializable(self):
        json.dumps(recipe_params(), allow_nan=False)


class DeriveSeedTests(unittest.TestCase):
    def test_matches_documented_encoding(self):
        payload = json.dumps([42, "s01/a.png", 1, 3], separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        self.assertEqual(derive_sample_seed(42, "s01/a.png", 1, 3), expected)

    def test_each_component_changes_the_seed(self):
        base = derive_sample_seed(42, "s01/a.png", 1, 3)
        self.assertNotEqual(base, derive_sample_seed(43, "s01/a.png", 1, 3))
        self.assertNotEqual(base, derive_sample_seed(42, "s02/a.png", 1, 3))
        self.assertNotEqual(base, derive_sample_seed(42, "s01/a.png", 2, 3))
        self.assertNotEqual(base, derive_sample_seed(42, "s01/a.png", 1, 4))


class CompositionTests(unittest.TestCase):
    def test_horizontal_flip_mirrors_pixels_and_boxes(self):
        image = asymmetric_image()
        sample = augment_one(image, [[.2, .4, .2, .2]], sample_seed=1, recipe=only("HorizontalFlip"))
        np.testing.assert_allclose(sample.boxes_yolo, [[.8, .4, .2, .2]], atol=1e-6)
        self.assertEqual(sample.class_ids, [0])
        self.assertEqual(sample.instance_ids, [0])
        np.testing.assert_array_equal(sample.image_bgr, image[:, ::-1])

    def test_pure_red_and_blue_blocks_keep_channel_order(self):
        image = np.zeros((80, 100, 3), np.uint8)
        image[:, :50] = (0, 0, 255)  # BGR: 红
        image[:, 50:] = (255, 0, 0)  # BGR: 蓝
        sample = augment_one(image, [], sample_seed=1, recipe=only("HorizontalFlip"))
        np.testing.assert_array_equal(sample.image_bgr[:, :50], np.broadcast_to((255, 0, 0), (80, 50, 3)))
        np.testing.assert_array_equal(sample.image_bgr[:, 50:], np.broadcast_to((0, 0, 255), (80, 50, 3)))

    def test_same_seed_gives_same_output_after_other_parents(self):
        image = asymmetric_image()
        other = np.full((80, 100, 3), 128, np.uint8)
        first_img, first_boxes = run(99, image, [[.5, .5, .2, .2]])
        run(5, other, [[.3, .3, .1, .1]])
        second_img, second_boxes = run(99, image, [[.5, .5, .2, .2]])
        np.testing.assert_array_equal(first_img, second_img)
        np.testing.assert_allclose(first_boxes, second_boxes)

    def test_default_recipe_outputs_valid_boxes_or_expected_rejection(self):
        image = asymmetric_image()
        accepted = 0
        for seed in range(30):
            try:
                sample = augment_one(image, [[.2, .4, .2, .2]], sample_seed=seed)
            except CandidateRejected as exc:
                self.assertEqual(exc.code, "unchanged_image")
                continue
            accepted += 1
            (x, y, w, h), = sample.boxes_yolo
            self.assertTrue(0 <= x - w / 2 and x + w / 2 <= 1 + 1e-6)
            self.assertTrue(0 <= y - h / 2 and y + h / 2 <= 1 + 1e-6)
            self.assertEqual(sample.image_bgr.shape, image.shape)
            self.assertEqual(sample.image_bgr.dtype, np.uint8)
        self.assertGreater(accepted, 15)

    def test_negative_sample_stays_empty(self):
        image = asymmetric_image()
        sample = augment_one(image, [], sample_seed=1, recipe=only("HorizontalFlip"))
        self.assertEqual(sample.boxes_yolo, [])
        self.assertEqual(sample.class_ids, [])
        self.assertEqual(sample.instance_ids, [])

    def test_default_recipe_negative_sample_is_empty_when_accepted(self):
        image = asymmetric_image()
        for seed in range(20):
            try:
                sample = augment_one(image, [], sample_seed=seed)
            except CandidateRejected as exc:
                self.assertEqual(exc.code, "unchanged_image")
                continue
            self.assertEqual(sample.boxes_yolo, [])


class RejectionTests(unittest.TestCase):
    def test_one_lost_instance_rejects_candidate(self):
        # 只沿 x 平移 50%：第一个框完整保留，第二个框仅约 50% 可见（<0.7），应被丢弃。
        recipe = only("Affine", scale=(1.0, 1.0), rotate=(0.0, 0.0),
                      translate_percent={"x": (0.5, 0.5), "y": (0.0, 0.0)})
        with self.assertRaises(CandidateRejected) as ctx:
            augment_one(asymmetric_image(), [[.2, .5, .2, .2], [.5, .5, .2, .2]],
                        sample_seed=1, recipe=recipe)
        self.assertEqual(ctx.exception.code, "dropped_box")

    def test_single_box_survives_same_translation(self):
        # 同一平移下仅保留第一个框时必须成功，证明上一用例的拒绝来自丢框。
        recipe = only("Affine", scale=(1.0, 1.0), rotate=(0.0, 0.0),
                      translate_percent={"x": (0.5, 0.5), "y": (0.0, 0.0)})
        sample = augment_one(asymmetric_image(), [[.2, .5, .2, .2]], sample_seed=1, recipe=recipe)
        self.assertEqual(len(sample.boxes_yolo), 1)

    def test_all_instances_lost_rejects_candidate(self):
        recipe = only("Affine", scale=(1.0, 1.0), rotate=(0.0, 0.0),
                      translate_percent={"x": (0.5, 0.5), "y": (0.0, 0.0)})
        with self.assertRaises(CandidateRejected) as ctx:
            augment_one(asymmetric_image(), [[.8, .5, .2, .2], [.9, .5, .2, .2]],
                        sample_seed=1, recipe=recipe)
        self.assertEqual(ctx.exception.code, "dropped_box")

    def test_invalid_input_boxes_are_rejected(self):
        bad = [
            [[.2, .4, 0.0, .2]],
            [[.2, .4, -.2, .2]],
            [[float("nan"), .4, .2, .2]],
            [[1.2, .4, .2, .2]],
            [[.2, .4, .2]],
        ]
        for boxes in bad:
            with self.subTest(boxes=boxes), self.assertRaises(CandidateRejected) as ctx:
                augment_one(asymmetric_image(), boxes, sample_seed=1, recipe=only("HorizontalFlip"))
            self.assertEqual(ctx.exception.code, "invalid_box")

    def test_unchanged_image_is_rejected(self):
        with self.assertRaises(CandidateRejected) as ctx:
            augment_one(asymmetric_image(), [[.2, .4, .2, .2]], sample_seed=1, recipe=none())
        self.assertEqual(ctx.exception.code, "unchanged_image")

    def test_non_bgr_uint8_input_is_programming_error(self):
        with self.assertRaises(ValueError):
            augment_one(np.zeros((80, 100), np.uint8), [], sample_seed=1)
        with self.assertRaises(ValueError):
            augment_one(np.zeros((80, 100, 3), np.float32), [], sample_seed=1)


class ImportTests(unittest.TestCase):
    def test_import_does_not_load_heavy_libraries(self):
        code = ("import sys; import energy_train.augmentation; "
                "bad = [n for n in ('torch', 'ultralytics', 'albumentations') if n in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
