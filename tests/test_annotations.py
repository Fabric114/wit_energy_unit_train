import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from energy_train.capture import CaptureSession
from energy_train.annotations import (
    annotation_status, load_instances, read_annotation_snapshot,
    render_session, session_folder,
)


def polygon(points=None, group=None, **overrides):
    return {"label": "energy_unit", "shape_type": "polygon", "group_id": group,
            "points": points or [[1, 1], [4, 1], [4, 4], [1, 4]], "flags": {}, **overrides}


def document(shapes):
    return {"imageWidth": 10, "imageHeight": 8, "imagePath": "s01_00000.png",
            "checked": True, "shapes": shapes}


def rectangle(points, **overrides):
    return {"label": "energy_unit", "shape_type": "rectangle", "group_id": None,
            "points": points, **overrides}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.frame = np.full((8, 10, 3), 60, np.uint8)
        with CaptureSession(self.raw, "s01", "g1", {"width": 10, "height": 8, "serial": "test"}, {}) as s:
            self.image = s.save(self.frame, 0, "manual", {})
        self.name = self.image.name
        self.label = self.image.with_suffix(".json")
        self.label.write_text(json.dumps(document([rectangle([[1, 2], [9, 7]])])))
        self.meta = json.loads((self.image.parent / "meta.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_rectangle_accepts_either_diagonal_and_drag_direction(self):
        for points in ([[1, 2], [9, 7]], [[9, 7], [1, 2]],
                       [[1, 7], [9, 2]], [[9, 2], [1, 7]], [[0, 0], [10, 8]]):
            with self.subTest(points=points):
                instances = load_instances(document([rectangle(points)]), self.frame.shape, self.name)
                expected = [0, 0, 10, 8] if points == [[0, 0], [10, 8]] else [1, 2, 9, 7]
                self.assertEqual(len(instances), 1)
                self.assertEqual(instances[0]["bbox"], expected)


    def test_rectangle_accepts_xanylabeling_four_corners(self):
        points = [[1.5, 2], [9, 2], [9, 7], [1.5, 7]]
        instances = load_instances(document([rectangle(points)]), self.frame.shape, self.name)
        self.assertEqual([instance["bbox"] for instance in instances], [[1.5, 2, 9, 7]])


    def test_rectangle_rejects_four_points_that_are_not_axis_aligned_corners(self):
        points = [[1, 2], [9, 2], [8, 7], [1, 7]]
        with self.assertRaises(ValueError):
            load_instances(document([rectangle(points)]), self.frame.shape, self.name)


    def test_rectangles_are_independent_even_with_shared_group(self):
        shapes = [rectangle([[1, 1], [3, 4]], group_id=1),
                  rectangle([[6, 2], [9, 5]], group_id=1),
                  rectangle([[1, 5], [3, 7]], group_id=1)]
        instances = load_instances(document(shapes), self.frame.shape, self.name)
        self.assertEqual([i["bbox"] for i in instances], [[1, 1, 3, 4], [6, 2, 9, 5], [1, 5, 3, 7]])


    def test_invalid_rectangles_rejected(self):
        for points in ([], [[1, 1]], [[1, 1], [4, 1], [4, 4]],
                       [[1, 1], [1, 4]], [[1, 1], [4, 1]],
                       [[-1.01, 1], [4, 4]], [[1, 1], [11.01, 4]],
                       [[1, 1], [4, 9.01]], [[1, 1], [float("inf"), 4]],
                       [[1, 1], [float("nan"), 4]], [[True, 1], [4, 4]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                load_instances(document([rectangle(points)]), self.frame.shape, self.name)


    def test_duplicate_rectangles_rejected_for_either_diagonal(self):
        first = rectangle([[1, 1], [4, 4]])
        for second in (rectangle([[4, 4], [1, 1]]), rectangle([[1, 4], [4, 1]]),
                       rectangle([[1, 1], [4, 1], [4, 4], [1, 4]])):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "重复"):
                load_instances(document([first, second]), self.frame.shape, self.name)


    def test_invalid_shapes_rejected(self):
        cases = [{**rectangle([[1, 1], [4, 4]]), "label": "armor"},
                 polygon(), polygon([[1, 1], [4, 1], [4, 4]]),
                 rectangle([[-1, 1], [2, 4]], shape_type="polygon")]
        for shape in cases:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                load_instances(document([shape]), self.frame.shape, self.name)


    def test_size_filename_and_duplicate_shape_rejected(self):
        for changes in ({"imageWidth": 11}, {"imagePath": "wrong.png"},
                        {"shapes": [rectangle([[1, 1], [4, 4]]), rectangle([[1, 1], [4, 4]])]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_instances({**document([]), **changes}, self.frame.shape, self.name)


    def test_native_status_accepts_only_boolean_true_as_passed(self):
        for value, expected in ((True, "passed"), (False, "pending"), ("failed", "failed")):
            self.assertEqual(annotation_status({"checked": value}), expected)
        self.assertEqual(annotation_status({}), "pending")
        for value in (1, 0, "true", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                annotation_status({"checked": value})

    def test_annotation_source_is_next_to_image(self):
        self.assertEqual(session_folder(self.raw, "s01"), self.raw / "s01")

    def test_small_rectangle_overflow_is_clipped_without_editing_document(self):
        doc = document([rectangle([[-0.7, 2], [9, 2], [9, 8.2], [-0.7, 8.2]])])
        before = json.dumps(doc)
        instances = load_instances(doc, self.frame.shape, self.name)
        self.assertEqual(instances[0]["bbox"], [0, 2, 9, 8])
        self.assertTrue(instances[0]["clipped"])
        self.assertEqual(json.dumps(doc), before)
        with self.assertRaises(ValueError):
            load_instances(document([rectangle([[-1, 1], [-0.5, 4]])]), self.frame.shape, self.name)

    def test_render_pending_is_optional_and_never_changes_source_files(self):
        doc = document([rectangle([[1, 2], [9, 7]])]); doc["checked"] = False
        self.label.write_text(json.dumps(doc))
        before = {p.name: p.read_bytes() for p in self.image.parent.iterdir() if p.is_file()}
        output = self.root / "preview"
        result = render_session(self.raw, "s01", output)
        self.assertEqual(result["images"], 1)
        self.assertEqual(json.loads((output / "index.json").read_text())[0]["status"], "pending")
        self.assertFalse(np.array_equal(cv2.imread(str(output / self.name)), self.frame))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.image.parent.iterdir() if p.is_file()})

    def test_embedded_image_must_match_raw_pixels(self):
        doc = document([])
        doc["imageData"] = base64.b64encode(cv2.imencode(".png", np.zeros_like(self.frame))[1]).decode()
        self.label.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "imageData|内嵌"):
            read_annotation_snapshot(self.image, self.label, self.meta)
        doc["imageData"] = base64.b64encode(self.image.read_bytes()).decode()
        self.label.write_text(json.dumps(doc))
        read_annotation_snapshot(self.image, self.label, self.meta)

    def test_image_path_cannot_point_at_different_same_named_image(self):
        other = self.root / "other" / self.name; other.parent.mkdir(); other.write_bytes(self.image.read_bytes())
        doc = document([]); doc["imagePath"] = str(other)
        self.label.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "imagePath"):
            read_annotation_snapshot(self.image, self.label, self.meta)


if __name__ == "__main__":
    unittest.main()
