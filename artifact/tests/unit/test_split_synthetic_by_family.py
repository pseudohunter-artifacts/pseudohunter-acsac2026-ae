"""Unit tests for scripts/split_synthetic_by_family.py (L45 fix)."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _import_split_module():
    """Import the script by path; it isn't an installed package."""

    here = Path(__file__).resolve().parent.parent.parent
    path = here / "scripts" / "split_synthetic_by_family.py"
    spec = importlib.util.spec_from_file_location(
        "_test_split_synthetic_by_family", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class LOFOSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_split_module()

    # ------------------------------------------------------- grouping

    def test_group_by_family_drops_non_ok(self):
        manifest = {
            "tasks": [
                {"task_name": "a", "transform_family": "xor", "status": "ok"},
                {"task_name": "b", "transform_family": "xor", "status": "ok"},
                {"task_name": "c", "transform_family": "xor", "status": "failed"},
                {"task_name": "d", "transform_family": "base64", "status": "ok"},
                # Task missing transform_family field; must also be dropped.
                {"task_name": "e", "status": "ok"},
            ]
        }
        groups = self.mod._group_by_family(manifest)
        self.assertEqual(sorted(groups), ["base64", "xor"])
        self.assertEqual([t["task_name"] for t in groups["xor"]], ["a", "b"])
        self.assertEqual([t["task_name"] for t in groups["base64"]], ["d"])

    # ------------------------------------------------------- split  building

    def test_build_split_partitions_correctly(self):
        fam_to_tasks = {
            "xor": [{"task_name": "x1"}, {"task_name": "x2"}],
            "base64": [{"task_name": "b1"}],
            "split_xor": [{"task_name": "s1"}, {"task_name": "s2"}, {"task_name": "s3"}],
        }
        s = self.mod._build_split(fam_to_tasks, held_out_family="xor")
        self.assertEqual(s["held_out_family"], "xor")
        self.assertEqual(s["train_count"], 4)   # 1 + 3
        self.assertEqual(s["val_count"], 2)
        self.assertEqual(s["train_task_names"], ["b1", "s1", "s2", "s3"])
        self.assertEqual(s["val_task_names"], ["x1", "x2"])
        # Sanity on the per-family counts that ship in the split doc.
        self.assertEqual(s["train_family_counts"], {"base64": 1, "split_xor": 3})
        self.assertEqual(s["val_family_counts"], {"xor": 2})

    def test_every_task_lands_in_exactly_one_of_train_or_val(self):
        fam_to_tasks = {
            "fa": [{"task_name": f"fa_{i}"} for i in range(5)],
            "fb": [{"task_name": f"fb_{i}"} for i in range(3)],
            "fc": [{"task_name": f"fc_{i}"} for i in range(2)],
        }
        for held_out in fam_to_tasks:
            s = self.mod._build_split(fam_to_tasks, held_out_family=held_out)
            all_tasks = set(s["train_task_names"]) | set(s["val_task_names"])
            expected = {
                t["task_name"] for ts in fam_to_tasks.values() for t in ts
            }
            self.assertEqual(
                all_tasks, expected,
                f"held_out={held_out}: train u val must cover every task"
            )
            self.assertEqual(
                set(s["train_task_names"]) & set(s["val_task_names"]),
                set(),
                f"held_out={held_out}: train and val must not overlap",
            )

    def test_val_set_equals_held_out_family_tasks(self):
        fam_to_tasks = {
            "fa": [{"task_name": f"fa_{i}"} for i in range(3)],
            "fb": [{"task_name": f"fb_{i}"} for i in range(2)],
        }
        s = self.mod._build_split(fam_to_tasks, held_out_family="fb")
        self.assertEqual(s["val_task_names"], ["fb_0", "fb_1"])

    def test_json_roundtrip_preserves_shape(self):
        import tempfile

        fam_to_tasks = {
            "xor": [{"task_name": "x1"}],
            "base64": [{"task_name": "b1"}],
        }
        s = self.mod._build_split(fam_to_tasks, held_out_family="xor")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            self.mod._write_json(p, s)
            back = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(back, s)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
