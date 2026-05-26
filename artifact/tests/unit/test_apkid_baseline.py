"""Tests for the APKiD-backed APK-level reference baseline.

These tests never import :mod:`apkid`; they inject a fake ``scan_fn`` so
that the behaviour can be exercised on a machine without the optional
APKiD runtime.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Mapping

from android_packer.baselines import (
    ApkidBaselineConfig,
    ApkidNotInstalledError,
    run_apkid_baseline,
)


def _apk_entry(apk_id: str, true_label_id: int = 0) -> dict:
    # Path content does not matter; fake scan_fn never touches the disk.
    return {
        "apk_id": apk_id,
        "apk_path": f"/virtual/{apk_id}.apk",
        "true_label_id": true_label_id,
    }


def _fake_scan(result: Mapping):
    """Build a scan_fn that always returns ``result``."""

    def _scan(apk_path: Path, timeout: float) -> Mapping:  # noqa: ARG001
        return result

    return _scan


def _scan_by_apk_id(results_by_id: Mapping[str, Mapping]):
    """Scan function that returns a different result per APK id."""

    def _scan(apk_path: Path, timeout: float) -> Mapping:  # noqa: ARG001
        apk_id = Path(apk_path).stem
        return results_by_id.get(apk_id, {"files": []})

    return _scan


class ApkidDecisionTests(unittest.TestCase):
    def test_empty_matches_yields_negative(self):
        result = run_apkid_baseline(
            [_apk_entry("benign")],
            scan_fn=_fake_scan({"files": [{"filename": "benign.apk", "matches": {}}]}),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 0)
        self.assertEqual(pred.predicted_label_id, 0)
        self.assertTrue(pred.scan_ok)
        self.assertEqual(pred.detected_families, [])

    def test_single_packer_hit_yields_positive(self):
        # APKiD-shaped JSON: one file, one packer family match.
        result = run_apkid_baseline(
            [_apk_entry("packed", true_label_id=1)],
            scan_fn=_fake_scan(
                {
                    "files": [
                        {
                            "filename": "packed.apk!classes.dex",
                            "matches": {"packer": ["Bangcle"]},
                        }
                    ]
                }
            ),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 1)
        self.assertEqual(pred.predicted_label_id, 1)
        self.assertEqual(pred.detected_families, ["Bangcle"])
        self.assertEqual(pred.matches[0].category, "packer")
        self.assertEqual(pred.matches[0].family, "Bangcle")

    def test_multiple_categories_and_files_are_all_counted(self):
        payload = {
            "files": [
                {
                    "filename": "a.apk!classes.dex",
                    "matches": {
                        "packer": ["Jiagu"],
                        "protector": ["Tencent Legu"],
                    },
                },
                {
                    "filename": "a.apk!classes2.dex",
                    "matches": {"obfuscator": ["DashO"]},
                },
            ]
        }
        result = run_apkid_baseline(
            [_apk_entry("a", true_label_id=1)],
            scan_fn=_fake_scan(payload),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 3)  # three distinct matches
        self.assertEqual(pred.predicted_label_id, 1)
        # detected_families is sorted + deduplicated across categories.
        self.assertEqual(
            pred.detected_families,
            sorted(["Jiagu", "Tencent Legu", "DashO"]),
        )

    def test_aux_categories_ignored_by_default(self):
        payload = {
            "files": [
                {
                    "filename": "a.apk!classes.dex",
                    "matches": {"anti_vm": ["Emulator Detector"]},
                }
            ]
        }
        result = run_apkid_baseline(
            [_apk_entry("a", true_label_id=0)],
            scan_fn=_fake_scan(payload),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 0)
        self.assertEqual(pred.predicted_label_id, 0)

    def test_aux_categories_promote_positive_when_enabled(self):
        payload = {
            "files": [
                {
                    "filename": "a.apk!classes.dex",
                    "matches": {"anti_disassembly": ["OLLVM"]},
                }
            ]
        }
        result = run_apkid_baseline(
            [_apk_entry("a", true_label_id=1)],
            config=ApkidBaselineConfig(include_aux_categories=True),
            scan_fn=_fake_scan(payload),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 1)
        self.assertEqual(pred.predicted_label_id, 1)
        self.assertEqual(pred.matches[0].category, "anti_disassembly")

    def test_min_hits_threshold(self):
        # With min_hits=2 a single match is no longer enough to flag.
        payload = {
            "files": [
                {
                    "filename": "x.apk!classes.dex",
                    "matches": {"packer": ["Bangcle"]},
                }
            ]
        }
        result = run_apkid_baseline(
            [_apk_entry("x", true_label_id=1)],
            config=ApkidBaselineConfig(min_hits=2),
            scan_fn=_fake_scan(payload),
        )
        pred = result.apk_predictions[0]
        self.assertEqual(pred.score, 1)
        self.assertEqual(pred.predicted_label_id, 0)

    def test_scan_failure_is_recorded_but_keeps_pipeline_moving(self):
        def _failing_scan(apk_path: Path, timeout: float) -> Mapping:  # noqa: ARG001
            raise TimeoutError("simulated apkid hang")

        result = run_apkid_baseline(
            [_apk_entry("broken", true_label_id=1)],
            scan_fn=_failing_scan,
        )
        pred = result.apk_predictions[0]
        self.assertFalse(pred.scan_ok)
        # Even with true_label_id=1, a failed scan must never be flipped
        # to positive: we have no evidence to report.
        self.assertEqual(pred.predicted_label_id, 0)
        self.assertIn("TimeoutError", pred.scan_error or "")
        self.assertEqual(result.report["counts"]["scan_failures"], 1)


class ApkidReportTests(unittest.TestCase):
    def test_report_advertises_apk_only_granularity(self):
        result = run_apkid_baseline(
            [_apk_entry("a", true_label_id=0)],
            scan_fn=_fake_scan({"files": []}),
        )
        self.assertEqual(result.report["baseline"], "apkid")
        self.assertEqual(result.report["localization_granularity"], "apk_only")
        # Crucially no 'region' or 'object' keys appear under metrics.
        self.assertEqual(list(result.report["metrics"].keys()), ["apk"])

    def test_family_histogram_is_sorted_and_counts_apks_not_matches(self):
        payload_a = {
            "files": [
                {
                    "filename": "a.apk!classes.dex",
                    "matches": {"packer": ["Bangcle", "Bangcle"]},
                }
            ]
        }
        payload_b = {
            "files": [
                {
                    "filename": "b.apk!classes.dex",
                    "matches": {"packer": ["Jiagu"]},
                }
            ]
        }
        result = run_apkid_baseline(
            [
                _apk_entry("a", true_label_id=1),
                _apk_entry("b", true_label_id=1),
            ],
            scan_fn=_scan_by_apk_id({"a": payload_a, "b": payload_b}),
        )
        histo = result.report["detected_family_histogram"]
        # Histogram counts the number of APKs on which a family was
        # detected, not the raw match count. Deduplication happens at
        # the APK level, so Bangcle shows up once for 'a'.
        self.assertEqual(histo, {"Bangcle": 1, "Jiagu": 1})

    def test_apk_metrics_reflect_ground_truth(self):
        # 2 TP, 1 FN, 0 FP.
        scans = {
            "pos1": {"files": [{"filename": "pos1.apk", "matches": {"packer": ["Jiagu"]}}]},
            "pos2": {"files": [{"filename": "pos2.apk", "matches": {"packer": ["Bangcle"]}}]},
            "pos3": {"files": [{"filename": "pos3.apk", "matches": {}}]},
            "neg1": {"files": [{"filename": "neg1.apk", "matches": {}}]},
        }
        entries = [
            _apk_entry("pos1", true_label_id=1),
            _apk_entry("pos2", true_label_id=1),
            _apk_entry("pos3", true_label_id=1),
            _apk_entry("neg1", true_label_id=0),
        ]
        result = run_apkid_baseline(entries, scan_fn=_scan_by_apk_id(scans))
        m = result.report["metrics"]["apk"]
        self.assertEqual(m["true_positives"], 2)
        self.assertEqual(m["false_negatives"], 1)
        self.assertEqual(m["false_positives"], 0)
        self.assertEqual(m["true_negatives"], 1)


class ApkidConfigTests(unittest.TestCase):
    def test_invalid_min_hits_is_rejected(self):
        with self.assertRaises(ValueError):
            run_apkid_baseline(
                [_apk_entry("a")],
                config=ApkidBaselineConfig(min_hits=0),
                scan_fn=_fake_scan({"files": []}),
            )

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            run_apkid_baseline(
                [_apk_entry("a")],
                config=ApkidBaselineConfig(timeout_seconds=0.0),
                scan_fn=_fake_scan({"files": []}),
            )


class ApkidImportFallbackTests(unittest.TestCase):
    def test_missing_apkid_raises_actionable_error(self):
        # When ``scan_fn`` is not provided and the real apkid module
        # is unavailable, a friendly error with install instructions
        # must bubble up on first invocation.
        #
        # We simulate "apkid not installed" by monkeypatching
        # ``_default_scan_fn`` so we do not rely on the environment.
        from android_packer.baselines import apkid as apkid_module

        def _raise_missing() -> None:
            raise ApkidNotInstalledError(
                "apkid not installed; install with "
                "'pip install android-packer[apkid]'"
            )

        original = apkid_module._default_scan_fn
        apkid_module._default_scan_fn = _raise_missing  # type: ignore[assignment]
        try:
            with self.assertRaises(ApkidNotInstalledError) as cm:
                run_apkid_baseline([_apk_entry("a")])
            self.assertIn("apkid", str(cm.exception).lower())
        finally:
            apkid_module._default_scan_fn = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
