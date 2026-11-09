import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

from energy_train.splitting import (
    check_duplicate_groups, load_yaml_snapshot, resolve_groups, verify_config_snapshots,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_IDS = {
    "s01/s01_00001.png", "s01/s01_00002.png", "s01/s01_00003.png", "s01/s01_00004.png",
    "s01/s01_00007.png", "s02/s02_00001.png", "s02/s02_00002.png",
}
VALID = """schema_version: 1
confirmed: true
groups:
  - id: s01_a
    reason: 第一段
    ranges:
      - {from: s01/s01_00001.png, to: s01/s01_00007.png}
  - id: s02_a
    reason: 第二段
    images: [s02/s02_00001.png, s02/s02_00002.png]
"""


def make_row(image_id, group, digest=None):
    session, name = image_id.split("/")
    return {
        "session": session, "source_image": f"/raw/{session}/{name}", "group_id": group,
        "image_sha256": digest or hashlib.sha256(image_id.encode()).hexdigest(),
        "bboxes": [], "width": 640, "height": 480,
    }


def meta_rows():
    return [make_row(i, "g1" if i.startswith("s01") else "g2") for i in sorted(IMAGE_IDS)]


def ids_of(resolved, group):
    return sorted(f"{r['session']}/{Path(r['source_image']).name}"
                  for r in resolved.records if r["split_group_id"] == group)


class SplittingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="groups.yaml"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_meta_fallback_and_no_mutation(self):
        rows = meta_rows()
        original = copy.deepcopy(rows)
        resolved = resolve_groups(rows, IMAGE_IDS)
        self.assertEqual(resolved.definition["mode"], "meta")
        self.assertTrue(resolved.confirmed)
        self.assertEqual({r["split_group_id"] for r in resolved.records}, {"g1", "g2"})
        self.assertEqual([r["scene"] for r in resolved.records], [r["group_id"] for r in resolved.records])
        self.assertEqual(rows, original)
        self.assertNotIn("split_group_id", rows[0])

    def test_range_expands_existing_pngs_inclusive(self):
        resolved = resolve_groups(meta_rows(), IMAGE_IDS, self.write(VALID))
        self.assertEqual(ids_of(resolved, "s01_a"), [
            "s01/s01_00001.png", "s01/s01_00002.png", "s01/s01_00003.png",
            "s01/s01_00004.png", "s01/s01_00007.png",
        ])
        self.assertEqual(ids_of(resolved, "s02_a"), ["s02/s02_00001.png", "s02/s02_00002.png"])
        self.assertTrue(resolved.confirmed)
        self.assertEqual(resolved.definition["mode"], "file")
        self.assertEqual(resolved.snapshot.sha256, hashlib.sha256(VALID.encode()).hexdigest())

    def test_explicit_scene_can_span_sessions(self):
        text = VALID.replace("  - id: s02_a\n    reason: 第二段\n    images: [s02/s02_00001.png, s02/s02_00002.png]",
                             "  - id: s02_a\n    scene: shared_scene\n    reason: 第二段\n    images: [s02/s02_00001.png, s02/s02_00002.png]")
        text = text.replace("  - id: s01_a\n    reason: 第一段\n    ranges:\n      - {from: s01/s01_00001.png, to: s01/s01_00007.png}",
                            "  - id: s01_a\n    scene: shared_scene\n    reason: 第一段\n    ranges:\n      - {from: s01/s01_00001.png, to: s01/s01_00007.png}")
        resolved = resolve_groups(meta_rows(), IMAGE_IDS, self.write(text))
        self.assertEqual({r["scene"] for r in resolved.records}, {"shared_scene"})
        self.assertEqual({r["split_group_id"] for r in resolved.records}, {"s01_a", "s02_a"})

    def test_excluded_images_leave_records_and_tags_attach(self):
        text = VALID.replace("      - {from: s01/s01_00001.png, to: s01/s01_00007.png}",
                             "      - {from: s01/s01_00001.png, to: s01/s01_00003.png}\n"
                             "      - {from: s01/s01_00007.png, to: s01/s01_00007.png}") + (
            "exclude:\n  - reason: 严重模糊\n    images: [s01/s01_00004.png]\n"
            "tags:\n  s01/s01_00001.png: [occlusion, blur]\n")
        resolved = resolve_groups(meta_rows(), IMAGE_IDS, self.write(text))
        self.assertEqual(resolved.excluded, ("s01/s01_00004.png",))
        self.assertNotIn("s01/s01_00004.png", ids_of(resolved, "s01_a"))
        first = next(r for r in resolved.records if r["source_image"].endswith("s01_00001.png"))
        self.assertEqual(first["tags"], ["blur", "occlusion"])

    def test_omitted_image_is_rejected(self):
        text = VALID.replace("images: [s02/s02_00001.png, s02/s02_00002.png]", "images: [s02/s02_00001.png]")
        with self.assertRaisesRegex(ValueError, "遗漏"):
            resolve_groups(meta_rows(), IMAGE_IDS, self.write(text))

    def test_invalid_group_files_are_rejected(self):
        variants = {
            "重复 YAML 键": VALID + "schema_version: 1\n",
            "未知顶层键": VALID + "extra: 1\n",
            "schema_version 为布尔": VALID.replace("schema_version: 1", "schema_version: true"),
            "confirmed 非布尔": VALID.replace("confirmed: true", "confirmed: 'yes'"),
            "份 ID 重复": VALID.replace("id: s02_a", "id: s01_a"),
            "非法 scene": VALID.replace("    reason: 第二段", "    reason: 第二段\n    scene: bad scene"),
            "reason 为空": VALID.replace("reason: 第一段", "reason: ' '"),
            "range 跨 session": VALID.replace("to: s01/s01_00007.png", "to: s02/s02_00002.png"),
            "range from 晚于 to": VALID.replace("from: s01/s01_00001.png, to: s01/s01_00007.png",
                                               "from: s01/s01_00007.png, to: s01/s01_00001.png"),
            "绝对路径": VALID.replace("images: [s02/s02_00001.png", "images: [/s02/s02_00001.png"),
            "反斜杠路径": VALID.replace("images: [s02/s02_00001.png", "images: [s02\\\\s02_00001.png"),
            "路径穿越": VALID.replace("images: [s02/s02_00001.png", "images: [s02/../s02/s02_00001.png"),
            "未知图片": VALID.replace("s02/s02_00002.png]", "s02/s02_09999.png]"),
            "图片属于两份": VALID.replace("images: [s02/s02_00001.png, s02/s02_00002.png]",
                                        "images: [s02/s02_00001.png, s02/s02_00002.png, s01/s01_00001.png]"),
            "份与排除项重叠": VALID + "exclude:\n  - reason: x\n    images: [s02/s02_00001.png]\n",
            "排除项缺 reason": VALID + "exclude:\n  - images: [s01/s01_00004.png]\n",
            "未知标签": VALID + "tags:\n  s02/s02_00001.png: [bad]\n",
            "重复标签": VALID + "tags:\n  s02/s02_00001.png: [blur, blur]\n",
            "空份": VALID.replace("images: [s02/s02_00001.png, s02/s02_00002.png]", "images: []"),
            "groups 为空": "schema_version: 1\nconfirmed: true\ngroups: []\n",
        }
        for label, text in variants.items():
            with self.subTest(label), self.assertRaises(ValueError):
                resolve_groups(meta_rows(), IMAGE_IDS, self.write(text))

    def test_duplicate_content_across_groups_rejected(self):
        rows = [make_row("s01/s01_00001.png", "g1", "same"), make_row("s02/s02_00001.png", "g2", "same")]
        resolved = resolve_groups(rows, {"s01/s01_00001.png", "s02/s02_00001.png"})
        with self.assertRaisesRegex(ValueError, "不同份"):
            check_duplicate_groups(resolved.records)

    def test_duplicate_content_within_group_is_reported(self):
        rows = [make_row("s01/s01_00001.png", "g1", "same"), make_row("s01/s01_00002.png", "g1", "same")]
        resolved = resolve_groups(rows, {"s01/s01_00001.png", "s01/s01_00002.png"})
        warnings = check_duplicate_groups(resolved.records)
        self.assertEqual([w["code"] for w in warnings], ["duplicate_image"])
        self.assertEqual(warnings[0]["count"], 2)

    def test_config_snapshot_detects_change_and_removal(self):
        path = self.write(VALID)
        _, snapshot = load_yaml_snapshot(path)
        verify_config_snapshots([snapshot])
        path.write_text(VALID + "# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "被修改"):
            verify_config_snapshots([snapshot])
        path.unlink()
        with self.assertRaisesRegex(ValueError, "消失"):
            verify_config_snapshots([snapshot])

    def test_example_file_is_marked_as_unconfirmed_example(self):
        text = (ROOT / "configs" / "dataset" / "groups.example.yaml").read_text(encoding="utf-8")
        data, _ = load_yaml_snapshot(ROOT / "configs" / "dataset" / "groups.example.yaml")
        self.assertIn("示例", text.splitlines()[0])
        self.assertIs(data["confirmed"], False)
        self.assertEqual(yaml.safe_load(text)["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
