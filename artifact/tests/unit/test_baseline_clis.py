"""Tests for the APKiD / sanity-rules CLIs.

The APKiD CLI cannot be exercised end-to-end without a real APKiD
installation, but we can still verify:

- argument parsing shape,
- the missing-apkid path produces a clean non-zero exit,
- the manifest-shaped and explicit-shaped JSONL entries are coerced
  into the correct baseline input.

The sanity-rules CLI is exercised end-to-end because it only depends
on stdlib + evaluation metrics.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from android_packer.cli import run_apkid_baseline as apkid_cli
from android_packer.cli import run_sanity_rules_baseline as sanity_cli


class ApkidCliEntryCoercionTests(unittest.TestCase):
    def test_explicit_entry_shape_passes_through(self):
        row = {"apk_id": "a", "apk_path": "/tmp/a.apk", "true_label_id": 1}
        self.assertEqual(apkid_cli._coerce_entry(row), row)

    def test_manifest_row_is_coerced(self):
        row = {
            "generated_apk_id": "pack-0001",
            "generated_apk_path": "/out/pack-0001.apk",
            "seed_apk_id": "seed-1",
            "transform_family": "xor",
        }
        coerced = apkid_cli._coerce_entry(row)
        self.assertEqual(coerced["apk_id"], "pack-0001")
        self.assertEqual(coerced["apk_path"], "/out/pack-0001.apk")
        self.assertEqual(coerced["true_label_id"], 1)

    def test_manifest_row_missing_id_keeps_stringifiable(self):
        # Defensive: if somebody fed a malformed manifest we at least
        # don't blow up in the CLI; the empty id will just produce a
        # scan failure downstream.
        row = {"generated_apk_path": "/out/x.apk"}
        coerced = apkid_cli._coerce_entry(row)
        self.assertEqual(coerced["apk_path"], "/out/x.apk")
        self.assertEqual(coerced["true_label_id"], 1)


class ApkidCliFailurePathTests(unittest.TestCase):
    def test_missing_apkid_produces_exit_code_2(self):
        # Force the library-level call to raise ApkidNotInstalledError
        # so we can see the CLI translate it into a controlled exit.
        import android_packer.cli.run_apkid_baseline as module
        from android_packer.baselines import ApkidNotInstalledError

        def _fake_run_apkid_baseline(*args, **kwargs):
            raise ApkidNotInstalledError("fake missing apkid")

        original = module.run_apkid_baseline
        module.run_apkid_baseline = _fake_run_apkid_baseline  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                entries = Path(td) / "entries.jsonl"
                entries.write_text(
                    json.dumps({"apk_id": "a", "apk_path": "/x.apk", "true_label_id": 0}) + "\n",
                    encoding="utf-8",
                )
                rc = module.main(
                    [
                        "--apk-entries",
                        str(entries),
                        "--apk-predictions-out",
                        str(Path(td) / "pred.jsonl"),
                        "--report-out",
                        str(Path(td) / "report.json"),
                    ]
                )
                self.assertEqual(rc, 2)
        finally:
            module.run_apkid_baseline = original  # type: ignore[assignment]


class SanityRulesCliEndToEndTests(unittest.TestCase):
    def test_end_to_end_writes_predictions_and_report(self):
        rows = [
            {
                "apk_id": "apk1",
                "object_id": "assets/secret.bin",
                "region_id": "r0",
                "object_path": "assets/secret.bin",
                "object_type": "blob",
                "offset_start": 0,
                "offset_end": 8192,
                "size": 8192,
                "sha256": "0" * 64,
                "entropy": 7.0,
                "printable_ratio": 0.01,
                "label": "hidden_executable_payload",
                "label_id": 1,
                "overlap_bytes": 0,
                "overlap_ratio": 0.0,
                "max_iou": 0.0,
                "matched_label_count": 0,
                "transform_families": [],
                "payload_sha256s": [],
            },
            {
                "apk_id": "apk1",
                "object_id": "assets/icon.png",
                "region_id": "r1",
                "object_path": "assets/icon.png",
                "object_type": "blob",
                "offset_start": 0,
                "offset_end": 1024,
                "size": 1024,
                "sha256": "1" * 64,
                "entropy": 4.0,
                "printable_ratio": 0.5,
                "label": "benign",
                "label_id": 0,
                "overlap_bytes": 0,
                "overlap_ratio": 0.0,
                "max_iou": 0.0,
                "matched_label_count": 0,
                "transform_families": [],
                "payload_sha256s": [],
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            labels_path = td / "labels.jsonl"
            labels_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )
            region_pred = td / "region.jsonl"
            object_pred = td / "object.jsonl"
            apk_pred = td / "apk.jsonl"
            report = td / "report.json"
            rc = sanity_cli.main(
                [
                    "--region-labels",
                    str(labels_path),
                    "--region-predictions-out",
                    str(region_pred),
                    "--object-predictions-out",
                    str(object_pred),
                    "--apk-predictions-out",
                    str(apk_pred),
                    "--report-out",
                    str(report),
                ]
            )
            self.assertEqual(rc, 0)
            # All four output files must exist.
            self.assertTrue(region_pred.exists())
            self.assertTrue(object_pred.exists())
            self.assertTrue(apk_pred.exists())
            self.assertTrue(report.exists())
            # Report must identify itself and include region/object/apk metrics.
            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_data["baseline"], "sanity_rules")
            self.assertIn("region", report_data["metrics"])
            self.assertIn("object", report_data["metrics"])
            self.assertIn("apk", report_data["metrics"])
            # The hidden payload object must have been flagged positive.
            object_rows = [
                json.loads(line)
                for line in object_pred.read_text(encoding="utf-8").splitlines()
            ]
            payload_row = next(
                r for r in object_rows if r["object_path"] == "assets/secret.bin"
            )
            self.assertEqual(payload_row["predicted_label_id"], 1)


if __name__ == "__main__":
    unittest.main()
