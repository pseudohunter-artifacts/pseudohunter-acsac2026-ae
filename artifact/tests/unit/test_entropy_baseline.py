import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from android_packer.baselines import EntropyBaselineConfig, run_entropy_baseline, score_region
from android_packer.utils.jsonl import write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env() -> dict:
    """Ensure the subprocess can import ``android_packer`` without editable install."""

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + existing if existing else "")
    return env


class EntropyBaselineTests(unittest.TestCase):
    def test_scores_regions_and_aggregates_predictions(self):
        rows = [
            _region("apk1", "obj1", "r0", "assets/payload.bin", 0, 4, 7.5, 0.1, 1),
            _region("apk1", "obj1", "r1", "assets/payload.bin", 4, 8, 2.0, 0.8, 1),
            _region("apk1", "obj2", "r2", "res/raw/data.bin", 0, 4, 3.0, 0.6, 0),
        ]

        result = run_entropy_baseline(
            rows,
            config=EntropyBaselineConfig(entropy_threshold=7.0),
        )

        self.assertEqual([row.predicted_label_id for row in result.region_predictions], [1, 0, 0])
        self.assertEqual([row.true_label_id for row in result.object_predictions], [1, 0])
        self.assertEqual([row.predicted_label_id for row in result.object_predictions], [1, 0])
        self.assertEqual(result.object_predictions[0].predicted_offset_start, 0)
        self.assertEqual(result.object_predictions[0].predicted_offset_end, 4)
        self.assertEqual(result.apk_predictions[0].predicted_label_id, 1)
        self.assertEqual(result.report["metrics"]["region"]["precision"], 1.0)
        self.assertEqual(result.report["metrics"]["region"]["recall"], 0.5)

    def test_nonprintable_weight_can_raise_score(self):
        score = score_region(
            6.0,
            0.25,
            EntropyBaselineConfig(
                entropy_threshold=6.5,
                entropy_weight=1.0,
                nonprintable_weight=1.0,
            ),
        )

        self.assertEqual(score, 6.75)

    def test_report_includes_auroc_and_object_level_ranking(self):
        # Two APKs, each with a high-entropy positive object and a low-entropy
        # benign object. Scores should produce perfect ranking -> MRR=1.0,
        # Top-1 hit=1.0, and AUROC=1.0 on each level that has both classes.
        rows = [
            _region("apk1", "obj_p", "r0", "assets/payload.bin", 0, 4, 7.8, 0.1, 1),
            _region("apk1", "obj_b", "r1", "assets/readme.txt", 0, 4, 2.0, 0.9, 0),
            _region("apk2", "obj_p", "r2", "assets/payload.bin", 0, 4, 7.5, 0.1, 1),
            _region("apk2", "obj_b", "r3", "assets/readme.txt", 0, 4, 2.5, 0.8, 0),
        ]

        result = run_entropy_baseline(
            rows, config=EntropyBaselineConfig(entropy_threshold=7.0)
        )

        object_metrics = result.report["metrics"]["object"]
        self.assertAlmostEqual(object_metrics["auroc"], 1.0)
        self.assertAlmostEqual(object_metrics["auprc"], 1.0)

        ranking = result.report["ranking"]["object"]
        self.assertEqual(ranking["evaluated_group_count"], 2)
        self.assertAlmostEqual(ranking["mrr"], 1.0)
        # Both APKs rank their positive object first.
        self.assertEqual(ranking["top_k_hit"]["1"], 1.0)

    def test_apk_level_hard_label_follows_score_threshold(self):
        # Single APK with only low-entropy regions: score stays below the
        # threshold so APK-level predicted_label_id must be 0 even if the
        # object-level aggregation previously treated any positive object
        # count as ``predicted=1``.
        rows = [
            _region("apkX", "obj", "r0", "assets/x.bin", 0, 4, 2.0, 0.9, 1),
        ]

        result = run_entropy_baseline(
            rows, config=EntropyBaselineConfig(entropy_threshold=7.0)
        )

        self.assertEqual(result.apk_predictions[0].predicted_label_id, 0)
        self.assertLess(result.apk_predictions[0].score, 7.0)

    def test_report_includes_object_level_localization(self):
        # Positive object with two contiguous positive regions [0, 8); both
        # also score above threshold so the prediction bounding box is [0, 8).
        # Expected IoU = 1.0, boundary_error = 0.
        rows = [
            _region("apkL", "obj_p", "r0", "assets/p.bin", 0, 4, 7.8, 0.1, 1),
            _region("apkL", "obj_p", "r1", "assets/p.bin", 4, 8, 7.9, 0.1, 1),
            _region("apkL", "obj_b", "r2", "assets/b.bin", 0, 4, 2.0, 0.9, 0),
        ]

        result = run_entropy_baseline(
            rows, config=EntropyBaselineConfig(entropy_threshold=7.0)
        )

        localization = result.report["localization"]["object"]
        self.assertEqual(localization["evaluated"], 1)
        self.assertAlmostEqual(localization["mean_iou"], 1.0)
        self.assertAlmostEqual(localization["mean_boundary_error"], 0.0)
        self.assertAlmostEqual(localization["offset_hit_rate"], 1.0)

    def test_cli_writes_predictions_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "region_labels.jsonl"
            region_out = tmp_path / "region_predictions.jsonl"
            object_out = tmp_path / "object_predictions.jsonl"
            apk_out = tmp_path / "apk_predictions.jsonl"
            report_out = tmp_path / "report.json"
            write_jsonl(
                labels_path,
                [
                    _region("apk1", "obj1", "r0", "assets/payload.bin", 0, 4, 7.5, 0.1, 1),
                    _region("apk1", "obj2", "r1", "res/raw/data.bin", 0, 4, 2.0, 0.9, 0),
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/baselines/run_entropy_baseline.py",
                    "--region-labels",
                    str(labels_path),
                    "--region-predictions-out",
                    str(region_out),
                    "--object-predictions-out",
                    str(object_out),
                    "--apk-predictions-out",
                    str(apk_out),
                    "--report-out",
                    str(report_out),
                    "--entropy-threshold",
                    "7.0",
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            self.assertIn("region_predictions=2", completed.stdout)
            self.assertTrue(region_out.exists())
            self.assertTrue(object_out.exists())
            self.assertTrue(apk_out.exists())
            report = json.loads(report_out.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["apk"]["f1"], 1.0)


def _region(
    apk_id: str,
    object_id: str,
    region_id: str,
    object_path: str,
    start: int,
    end: int,
    entropy: float,
    printable_ratio: float,
    label_id: int,
) -> dict:
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "region_id": region_id,
        "object_path": object_path,
        "object_type": "asset_blob",
        "offset_start": start,
        "offset_end": end,
        "size": end - start,
        "sha256": "f" * 64,
        "entropy": entropy,
        "printable_ratio": printable_ratio,
        "label": "hidden_executable_payload" if label_id else "benign",
        "label_id": label_id,
        "overlap_bytes": end - start if label_id else 0,
        "overlap_ratio": 1.0 if label_id else 0.0,
        "max_iou": 1.0 if label_id else 0.0,
        "matched_label_count": 1 if label_id else 0,
        "transform_families": ["xor"] if label_id else [],
        "payload_sha256s": ["a" * 64] if label_id else [],
    }


if __name__ == "__main__":
    unittest.main()
