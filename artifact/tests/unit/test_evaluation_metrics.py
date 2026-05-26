import importlib
import math
import unittest

from android_packer.evaluation import metrics as metrics_module
from android_packer.evaluation.metrics import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)


class BinaryClassificationMetricsTests(unittest.TestCase):
    def test_hard_labels_compute_standard_counts(self):
        result = binary_classification_metrics(
            truth=[1, 1, 0, 0, 1],
            predictions=[1, 0, 0, 1, 1],
        )

        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.true_negatives, 1)
        # precision = 2/3, recall = 2/3, f1 = 2*2/(2*2+1+1) = 4/6
        self.assertAlmostEqual(result.precision, round(2 / 3, 6))
        self.assertAlmostEqual(result.recall, round(2 / 3, 6))
        self.assertAlmostEqual(result.f1, round(4 / 6, 6))
        self.assertIsNone(result.auroc)
        self.assertIsNone(result.auprc)

    def test_scores_with_threshold_compute_auroc_and_auprc(self):
        result = binary_classification_metrics(
            truth=[0, 0, 1, 1],
            scores=[0.1, 0.4, 0.35, 0.8],
            threshold=0.5,
        )

        # Only score >= 0.5 predicts positive: single TP for the 0.8 sample.
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 0)
        # AUROC = 0.75 for this classic textbook case.
        self.assertAlmostEqual(result.auroc, 0.75)
        self.assertIsNotNone(result.auprc)
        self.assertGreater(result.auprc, 0.0)

    def test_auroc_undefined_when_only_one_class_present(self):
        result = binary_classification_metrics(
            truth=[1, 1, 1],
            scores=[0.1, 0.5, 0.9],
            threshold=0.5,
        )

        self.assertIsNone(result.auroc)
        self.assertIsNone(result.auprc)

    def test_perfect_and_inverse_rankings(self):
        perfect = binary_classification_metrics(
            truth=[0, 0, 1, 1],
            scores=[0.1, 0.2, 0.9, 0.95],
            threshold=0.5,
        )
        inverse = binary_classification_metrics(
            truth=[0, 0, 1, 1],
            scores=[0.9, 0.95, 0.1, 0.2],
            threshold=0.5,
        )

        self.assertEqual(perfect.auroc, 1.0)
        self.assertEqual(inverse.auroc, 0.0)

    def test_requires_predictions_or_scores(self):
        with self.assertRaises(ValueError):
            binary_classification_metrics(truth=[0, 1])

    def test_scores_without_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            binary_classification_metrics(
                truth=[0, 1],
                scores=[0.2, 0.8],
            )

    def test_pure_python_fallback_matches_sklearn(self):
        truth = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.45, 0.7, 0.6, 0.25]

        with_sklearn = binary_classification_metrics(
            truth=truth,
            scores=scores,
            threshold=0.5,
        )

        # Force the fallback path and recompute.
        original = metrics_module._SKLEARN_AVAILABLE
        metrics_module._SKLEARN_AVAILABLE = False
        try:
            fallback = binary_classification_metrics(
                truth=truth,
                scores=scores,
                threshold=0.5,
            )
        finally:
            metrics_module._SKLEARN_AVAILABLE = original

        # Guard against accidentally skipping the contrast (the fixture must
        # have sklearn installed for the test to be meaningful).
        self.assertTrue(original, "scikit-learn is required to contrast the fallback")
        self.assertAlmostEqual(with_sklearn.auroc, fallback.auroc, places=6)
        self.assertAlmostEqual(with_sklearn.auprc, fallback.auprc, places=6)

    def test_module_reimports_cleanly_without_sklearn(self):
        # Simulate an install without the optional dependency by reloading the
        # module with its sklearn symbols hidden.
        import sys

        saved = {
            name: sys.modules[name]
            for name in list(sys.modules)
            if name == "sklearn" or name.startswith("sklearn.")
        }
        for name in saved:
            sys.modules[name] = None  # type: ignore[assignment]
        try:
            reloaded = importlib.reload(metrics_module)
            self.assertFalse(reloaded._SKLEARN_AVAILABLE)
            result = reloaded.binary_classification_metrics(
                truth=[0, 0, 1, 1],
                scores=[0.1, 0.4, 0.35, 0.8],
                threshold=0.5,
            )
            self.assertAlmostEqual(result.auroc, 0.75, places=6)
        finally:
            for name, module in saved.items():
                sys.modules[name] = module
            importlib.reload(metrics_module)


class RankingMetricsTests(unittest.TestCase):
    def test_mrr_and_top_k_aggregate_across_groups(self):
        # Group A: positive ranked 2nd (rr=0.5)
        # Group B: positive ranked 1st (rr=1.0)
        # Group C: no positive, skipped.
        result = ranking_metrics(
            groups=["A", "A", "A", "B", "B", "C", "C"],
            truth=[0, 1, 0, 1, 0, 0, 0],
            scores=[0.9, 0.5, 0.1, 0.8, 0.2, 0.7, 0.6],
            ks=(1, 3),
        )

        self.assertEqual(result.group_count, 3)
        self.assertEqual(result.evaluated_group_count, 2)
        self.assertAlmostEqual(result.mrr, round((0.5 + 1.0) / 2, 6))
        self.assertEqual(result.top_k_hit[1], 0.5)
        self.assertEqual(result.top_k_hit[3], 1.0)

    def test_groups_without_any_positive_are_skipped(self):
        result = ranking_metrics(
            groups=["solo", "solo", "solo"],
            truth=[0, 0, 0],
            scores=[0.1, 0.2, 0.3],
        )

        self.assertEqual(result.evaluated_group_count, 0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.top_k_hit[1], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ranking_metrics(groups=["A"], truth=[1, 0], scores=[0.1, 0.2])


class LocalizationMetricsTests(unittest.TestCase):
    def test_perfect_alignment_yields_unit_iou_and_zero_error(self):
        result = localization_metrics(
            [
                {"truth": [(10, 20)], "prediction": [(10, 20)]},
                {"truth": [(0, 5), (10, 15)], "prediction": [(0, 5), (10, 15)]},
            ]
        )

        self.assertEqual(result.evaluated, 2)
        self.assertAlmostEqual(result.mean_iou, 1.0)
        self.assertAlmostEqual(result.mean_boundary_error, 0.0)
        self.assertAlmostEqual(result.offset_hit_rate, 1.0)

    def test_partial_overlap_is_measured_on_unions(self):
        result = localization_metrics(
            [{"truth": [(0, 10)], "prediction": [(5, 15)]}],
        )

        # Intersection 5, union 15 -> IoU 1/3; boundary error |5-0|+|15-10|=10
        self.assertAlmostEqual(result.mean_iou, round(5 / 15, 6))
        self.assertAlmostEqual(result.mean_boundary_error, 10.0)
        self.assertAlmostEqual(result.offset_hit_rate, 1.0)

    def test_missing_prediction_reports_zero_iou_and_extent_penalty(self):
        result = localization_metrics(
            [{"truth": [(100, 150)], "prediction": []}],
        )

        self.assertAlmostEqual(result.mean_iou, 0.0)
        # Fallback penalty: 2 * truth extent = 2 * (150 - 100) = 100.
        self.assertAlmostEqual(result.mean_boundary_error, 100.0)
        self.assertAlmostEqual(result.offset_hit_rate, 0.0)

    def test_samples_without_truth_do_not_contribute(self):
        result = localization_metrics(
            [
                {"truth": [], "prediction": [(0, 10)]},
                {"truth": [(0, 10)], "prediction": [(0, 10)]},
            ]
        )

        self.assertEqual(result.support, 2)
        self.assertEqual(result.evaluated, 1)
        self.assertAlmostEqual(result.mean_iou, 1.0)


if __name__ == "__main__":
    unittest.main()
