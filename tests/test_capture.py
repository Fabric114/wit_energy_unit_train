import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from energy_train.capture import (
    CaptureConfig, CaptureSession, assess_frame, rejection_reason, run_capture,
)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.camera = {"width": 4, "height": 3, "serial": "test"}
        self.frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def writer(self, group="g1", camera=None):
        return CaptureSession(self.root, "s01", group, camera or self.camera, {})

    def test_png_preserves_bgr_and_records_provenance(self):
        with self.writer() as session:
            p = session.save(self.frame, 1234.5, "manual", assess_frame(self.frame))
        np.testing.assert_array_equal(cv2.imread(str(p)), self.frame)
        line = json.loads((p.parent / "frames.jsonl").read_text())
        self.assertEqual(line["group_id"], "g1")
        self.assertEqual(line["timestamp_ms"], 1234.5)
        self.assertEqual(line["reason"], "manual")
        self.assertEqual(len(line["image_sha256"]), 64)

    def test_resume_uses_max_number_not_count_and_preserves_meta(self):
        with self.writer() as session:
            p = session.save(self.frame, 1, "auto", {})
        (p.parent / "s01_00008.png").write_bytes(p.read_bytes())
        with self.writer() as session:
            q = session.save(self.frame, 2, "auto", {})
        self.assertEqual(q.name, "s01_00009.png")
        self.assertTrue(p.exists())
        meta = json.loads((p.parent / "meta.json").read_text())
        self.assertEqual(len(meta["capture_runs"]), 2)

    def test_reject_group_or_camera_change(self):
        with self.writer():
            pass
        with self.assertRaisesRegex(ValueError, "group"):
            with self.writer(group="different"):
                pass
        with self.assertRaisesRegex(ValueError, "camera"):
            with self.writer(camera={**self.camera, "serial": "other"}):
                pass

    def test_reject_path_traversal_and_competing_writer(self):
        with self.assertRaises(ValueError):
            CaptureSession(self.root, "../bad", "g1", self.camera, {})
        with self.writer():
            with self.assertRaisesRegex(RuntimeError, "使用|locked"):
                with self.writer():
                    pass

    def test_write_failure_leaves_no_png_or_success_log(self):
        with self.writer() as session:
            with patch("energy_train.capture.cv2.imencode", return_value=(False, None)):
                with self.assertRaises(OSError):
                    session.save(self.frame, 1, "auto", {})
        folder = self.root / "s01"
        self.assertEqual(list(folder.glob("*.png")), [])
        self.assertFalse((folder / "frames.jsonl").exists())

    def test_reject_wrong_frame_shape_and_dtype(self):
        with self.writer() as session:
            for frame in (self.frame[:2], self.frame.astype(float)):
                with self.assertRaises(ValueError):
                    session.save(frame, 1, "auto", {})

    def test_quality_policy_preserves_difficult_frames_and_manual_override(self):
        q = assess_frame(np.zeros((3, 4, 3), np.uint8), self.frame)
        self.assertEqual(q["blur_score"], 0)
        cfg = CaptureConfig(session="s01", group_id="g1", reject_blur=False)
        self.assertIsNone(rejection_reason(q, cfg, manual=False))
        cfg.reject_blur = True
        self.assertEqual(rejection_reason(q, cfg, manual=False), "blur")
        self.assertIsNone(rejection_reason(q, cfg, manual=True))
        cfg.reject_blur = False
        q["difference"] = 0.0
        self.assertEqual(rejection_reason(q, cfg, manual=False), "duplicate")

    def test_invalid_config(self):
        for changes in ({"interval": 0}, {"max_images": -1}, {"interval": float("nan")},
                        {"width": 2.5}, {"preview": False, "auto": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CaptureConfig(session="s", group_id="g", **changes).validate()

    def test_loop_warmup_interval_limit_and_release(self):
        class Source:
            def __init__(self, camera):
                self.t = 0
                self.stopped = False
                self.camera = camera

            def start(self):
                return self.camera

            def read(self):
                self.t += 0.1
                return np.full((3, 4, 3), int(self.t * 30), np.uint8), self.t * 1000

            def stop(self):
                self.stopped = True

        source = Source(self.camera)
        cfg = CaptureConfig(output_root=self.root, session="s01", group_id="g1",
                            width=4, height=3, warmup_frames=2, interval=0.5,
                            max_images=3, preview=False, duplicate_threshold=0)
        result = run_capture(cfg, source=source, clock=lambda: source.t)
        self.assertTrue(source.stopped)
        self.assertEqual(result["saved"], 3)
        records = [json.loads(s) for s in (self.root / "s01/frames.jsonl").read_text().splitlines()]
        times = [r["timestamp_ms"] for r in records]
        self.assertGreaterEqual(times[0], 300)
        self.assertTrue(all(b - a >= 499 for a, b in zip(times, times[1:])))

    def test_read_failure_releases_source(self):
        class Broken:
            stopped = False

            def start(self):
                return {"width": 4, "height": 3, "serial": "test"}

            def read(self):
                raise RuntimeError("disconnected")

            def stop(self):
                self.stopped = True

        source = Broken()
        cfg = CaptureConfig(output_root=self.root, session="s01", group_id="g1",
                            width=4, height=3, preview=False)
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            run_capture(cfg, source=source)
        self.assertTrue(source.stopped)


if __name__ == "__main__":
    unittest.main()
