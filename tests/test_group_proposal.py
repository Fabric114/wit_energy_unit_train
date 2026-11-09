import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from energy_train.group_proposal import propose_groups, render_group_file
from energy_train.splitting import resolve_groups

ROOT = Path(__file__).resolve().parents[1]
BASE = datetime(2026, 11, 9, tzinfo=timezone.utc)


def at(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat()


def make_session(raw, session, stamps, runs=(0,), extra_records=()):
    """stamps 为每张图的 timestamp_ms；runs 为各次启动的秒数（UTC 墙钟）。"""
    folder = raw / session
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps(
        {"session": session, "group_id": "g", "capture_runs": [{"started_at": at(s), "settings": {}} for s in runs]}))
    lines = []
    for number, ts in enumerate(stamps):
        name = f"{session}_{number:05d}.png"
        cv2.imwrite(str(folder / name), np.full((2, 2, 3), number, np.uint8))
        lines.append({"image": name, "session": session, "group_id": "g", "saved_at": at(ts / 1000),
                      "timestamp_ms": ts})
    lines.extend(extra_records)
    (folder / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    return folder


def ids(group):
    return [Path(p).name for p in group.images]


class GroupProposalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "raw"
        self.raw.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_time_gap_starts_a_new_group_and_small_gaps_do_not(self):
        make_session(self.raw, "s01", [0, 2000, 4000, 70000])
        proposal = propose_groups(self.raw, gap_seconds=60)
        self.assertEqual([len(g.images) for g in proposal.groups], [3, 1])

    def test_capture_run_boundary_splits_even_with_small_gap(self):
        make_session(self.raw, "s01", [0, 1000, 2000, 3000], runs=(0, 1.5))
        proposal = propose_groups(self.raw, gap_seconds=60)
        self.assertEqual([len(g.images) for g in proposal.groups], [2, 2])

    def test_short_gap_is_flagged_and_long_gap_is_not(self):
        make_session(self.raw, "s01", [0, 100_000, 500_000])
        groups = propose_groups(self.raw, gap_seconds=60).groups
        self.assertTrue(groups[1].short_gap)
        self.assertFalse(groups[2].short_gap)

    def test_sessions_never_merge(self):
        make_session(self.raw, "s01", [0, 1000])
        make_session(self.raw, "s02", [2000, 3000])
        proposal = propose_groups(self.raw, gap_seconds=60)
        self.assertEqual({g.id.split("_")[0] for g in proposal.groups}, {"s01", "s02"})

    def test_frames_deleted_from_disk_are_ignored(self):
        make_session(self.raw, "s01", [0, 1000], extra_records=[
            {"image": "s01_00099.png", "session": "s01", "group_id": "g", "saved_at": at(9),
             "timestamp_ms": 9000}])
        self.assertEqual(sum(len(g.images) for g in propose_groups(self.raw).groups), 2)

    def test_png_without_frame_record_is_rejected(self):
        folder = make_session(self.raw, "s01", [0, 1000])
        (folder / "frames.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frames.jsonl"):
            propose_groups(self.raw)

    def test_duplicate_frame_record_is_rejected(self):
        make_session(self.raw, "s01", [0], extra_records=[
            {"image": "s01_00000.png", "session": "s01", "group_id": "g", "saved_at": at(0),
             "timestamp_ms": 5}])
        with self.assertRaisesRegex(ValueError, "重复"):
            propose_groups(self.raw)

    def test_non_finite_timestamp_is_rejected(self):
        folder = make_session(self.raw, "s01", [0])
        record = json.loads((folder / "frames.jsonl").read_text())
        record["timestamp_ms"] = None
        (folder / "frames.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "timestamp_ms"):
            propose_groups(self.raw)

    def test_long_scene_is_subdivided_with_buffer_excluded_at_cuts(self):
        make_session(self.raw, "s01", [i * 1000 for i in range(25)])
        proposal = propose_groups(self.raw, gap_seconds=60, max_group_size=10, buffer=3)
        self.assertEqual(len(proposal.groups), 3)
        self.assertTrue(all(len(g.images) <= 10 for g in proposal.groups))
        self.assertEqual({g.scene for g in proposal.groups}, {"s01_g01"})
        self.assertEqual(len(proposal.excluded), 12)

    def test_invalid_subdivision_parameters_are_rejected(self):
        make_session(self.raw, "s01", [i * 1000 for i in range(25)])
        for kwargs in ({"max_group_size": True}, {"buffer": -1}, {"max_group_size": 6, "buffer": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                propose_groups(self.raw, max_group_size=kwargs.get("max_group_size"),
                               buffer=kwargs.get("buffer", 3))

    def test_rendered_draft_round_trips_through_group_file_parser(self):
        make_session(self.raw, "s01", [0, 1000, 2000, 90_000, 91_000])
        proposal = propose_groups(self.raw, gap_seconds=60)
        draft = self.raw.parent / "groups.yaml"
        draft.write_text(render_group_file(proposal), encoding="utf-8")
        names = sorted(p.name for p in (self.raw / "s01").glob("*.png"))
        records = [{"session": "s01", "source_image": f"/raw/s01/{name}", "group_id": "g",
                    "image_sha256": name} for name in names]
        image_ids = {f"s01/{name}" for name in names}
        resolved = resolve_groups(records, image_ids, draft)
        self.assertFalse(resolved.confirmed)
        self.assertEqual(len(resolved.records), 5)

    def test_cli_refuses_existing_output_and_never_touches_raw(self):
        make_session(self.raw, "s01", [0, 1000])
        before = {p: p.read_bytes() for p in self.raw.rglob("*") if p.is_file()}
        output = self.raw.parent / "groups.yaml"
        output.write_text("keep", encoding="utf-8")
        tool = ROOT / "script" / "propose_groups.py"
        result = subprocess.run([sys.executable, str(tool), "--raw-root", str(self.raw),
                                 "--output", str(output)], cwd=self.raw.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")
        self.assertEqual(before, {p: p.read_bytes() for p in self.raw.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
