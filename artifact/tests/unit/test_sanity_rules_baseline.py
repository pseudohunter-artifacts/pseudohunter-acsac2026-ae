"""Tests for the sanity-check heuristic baseline (:mod:`baselines.sanity_rules`)."""

import unittest

from android_packer.baselines import (
    SanityRulesConfig,
    run_sanity_rules_baseline,
)


def _region_row(
    *,
    apk_id: str,
    object_id: str,
    region_id: str,
    object_path: str,
    offset_start: int,
    offset_end: int,
    label_id: int,
    printable_ratio: float = 0.4,
    object_type: str = "blob",
    label: str = "benign",
) -> dict:
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "region_id": region_id,
        "object_path": object_path,
        "object_type": object_type,
        "offset_start": offset_start,
        "offset_end": offset_end,
        "size": offset_end - offset_start,
        "sha256": "0" * 64,
        "entropy": 4.0,
        "printable_ratio": printable_ratio,
        "label": label,
        "label_id": label_id,
        "overlap_bytes": 0,
        "overlap_ratio": 0.0,
        "max_iou": 0.0,
        "matched_label_count": 0,
        "transform_families": [],
        "payload_sha256s": [],
    }


class SanityRulesSignalTests(unittest.TestCase):
    def test_classes_dex_is_never_flagged_even_when_large(self):
        # Code path: two 4KB regions of the real classes.dex.
        rows = [
            _region_row(
                apk_id="apk",
                object_id="classes.dex",
                region_id=f"r{i}",
                object_path="classes.dex",
                object_type="dex",
                offset_start=i * 4096,
                offset_end=(i + 1) * 4096,
                label_id=0,
                printable_ratio=0.02,
            )
            for i in range(2)
        ]
        result = run_sanity_rules_baseline(rows)
        obj = result.object_predictions[0]
        self.assertEqual(obj.predicted_label_id, 0)
        self.assertEqual(obj.triggered_rules, ())

    def test_asset_payload_triggers_multiple_rules(self):
        # Hidden payload: assets/secret.bin, large, unknown extension,
        # suspicious path prefix, low printable ratio.
        rows = [
            _region_row(
                apk_id="apk",
                object_id="assets/secret.bin",
                region_id="r0",
                object_path="assets/secret.bin",
                offset_start=0,
                offset_end=8192,
                printable_ratio=0.02,
                label_id=1,
                label="hidden_executable_payload",
            ),
        ]
        result = run_sanity_rules_baseline(rows)
        obj = result.object_predictions[0]
        self.assertEqual(obj.predicted_label_id, 1)
        self.assertEqual(
            set(obj.triggered_rules),
            {
                "suspicious_path",
                "unknown_extension",
                "large_object",
                "low_printable_region",
            },
        )
        # The APK-level prediction must also fire because the score was
        # broadcast from the object's decision.
        self.assertEqual(result.apk_predictions[0].predicted_label_id, 1)

    def test_benign_asset_image_does_not_fire(self):
        # assets/icon.png with legitimate printable ratio and small size.
        # The suspicious_path rule still fires (anything under assets/
        # counts as "resource-like"), but a single 0.6-weight signal is
        # below the default 1.0 threshold, so the object stays negative.
        rows = [
            _region_row(
                apk_id="apk",
                object_id="assets/icon.png",
                region_id="r0",
                object_path="assets/icon.png",
                offset_start=0,
                offset_end=1024,
                printable_ratio=0.4,
                label_id=0,
            ),
        ]
        result = run_sanity_rules_baseline(rows)
        obj = result.object_predictions[0]
        # Path-based rule fires on its own, but unknown_extension /
        # large_object / low_printable must stay silent for a .png asset.
        self.assertEqual(obj.triggered_rules, ("suspicious_path",))
        self.assertEqual(obj.predicted_label_id, 0)

    def test_path_randomized_payload_is_flagged(self):
        # ``path_randomized`` turns the object path into something random
        # without an extension; must trigger suspicious_path (under
        # assets/) + unknown_extension + large_object.
        rows = [
            _region_row(
                apk_id="apk",
                object_id="assets/a0b9c1d2e3f4",
                region_id="r0",
                object_path="assets/a0b9c1d2e3f4",
                offset_start=0,
                offset_end=8192,
                printable_ratio=0.5,
                label_id=1,
                label="hidden_executable_payload",
            ),
        ]
        result = run_sanity_rules_baseline(rows)
        obj = result.object_predictions[0]
        triggered = set(obj.triggered_rules)
        self.assertIn("suspicious_path", triggered)
        self.assertIn("unknown_extension", triggered)
        self.assertIn("large_object", triggered)
        self.assertEqual(obj.predicted_label_id, 1)


class SanityRulesReportTests(unittest.TestCase):
    def test_report_structure_matches_entropy_contract(self):
        # A mixed APK with one benign image and one hidden payload.
        rows = [
            _region_row(
                apk_id="apk1",
                object_id="assets/icon.png",
                region_id="r0",
                object_path="assets/icon.png",
                offset_start=0,
                offset_end=1024,
                label_id=0,
            ),
            _region_row(
                apk_id="apk1",
                object_id="assets/secret.bin",
                region_id="r1",
                object_path="assets/secret.bin",
                offset_start=0,
                offset_end=8192,
                printable_ratio=0.01,
                label_id=1,
                label="hidden_executable_payload",
            ),
        ]
        result = run_sanity_rules_baseline(rows)
        report = result.report

        self.assertEqual(report["baseline"], "sanity_rules")
        self.assertEqual(report["counts"]["regions"], 2)
        self.assertEqual(report["counts"]["objects"], 2)
        self.assertEqual(report["counts"]["apks"], 1)

        for level in ("region", "object", "apk"):
            metrics = report["metrics"][level]
            for key in ("precision", "recall", "f1", "auroc", "auprc"):
                self.assertIn(key, metrics)

        self.assertIn("ranking", report)
        self.assertIn("mrr", report["ranking"]["object"])
        self.assertIn("localization", report)
        self.assertIn("mean_iou", report["localization"]["object"])
        self.assertIn("rule_trigger_counts", report)

    def test_rule_trigger_counts_reflect_multi_rule_hits(self):
        rows = [
            _region_row(
                apk_id="apk",
                object_id="assets/secret.bin",
                region_id="r0",
                object_path="assets/secret.bin",
                offset_start=0,
                offset_end=8192,
                printable_ratio=0.01,
                label_id=1,
                label="hidden_executable_payload",
            ),
        ]
        result = run_sanity_rules_baseline(rows)
        counts = result.report["rule_trigger_counts"]
        self.assertEqual(counts["suspicious_path"], 1)
        self.assertEqual(counts["unknown_extension"], 1)
        self.assertEqual(counts["large_object"], 1)
        self.assertEqual(counts["low_printable_region"], 1)


class SanityRulesConfigTests(unittest.TestCase):
    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            run_sanity_rules_baseline(
                [],
                config=SanityRulesConfig(suspicious_path_weight=-0.1),
            )

    def test_high_threshold_suppresses_single_signal_hits(self):
        # A single low-weight signal shouldn't cross a high threshold.
        rows = [
            _region_row(
                apk_id="apk",
                object_id="assets/icon.unknownext",
                region_id="r0",
                object_path="assets/icon.unknownext",
                offset_start=0,
                offset_end=100,  # small, so large_object won't fire
                printable_ratio=0.5,  # won't fire low_printable
                label_id=0,
            ),
        ]
        result = run_sanity_rules_baseline(
            rows,
            config=SanityRulesConfig(threshold=10.0),
        )
        obj = result.object_predictions[0]
        # Even though multiple rules fire (suspicious_path + unknown_ext),
        # the absurdly high threshold keeps it negative.
        self.assertEqual(obj.predicted_label_id, 0)


if __name__ == "__main__":
    unittest.main()
