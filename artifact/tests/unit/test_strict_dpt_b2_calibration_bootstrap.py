from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

from scripts.experiments.run_strict_dpt_b2_calibration_bootstrap import (
    best_f1_threshold,
    bootstrap_cis,
    brier_score,
    expected_calibration_error,
    paired_bootstrap_delta,
    threshold_for_target_tpr,
)


def test_brier_score() -> None:
    assert brier_score([0, 1], [0.25, 0.75]) == pytest.approx(0.0625)


def test_expected_calibration_error_bins_scores() -> None:
    labels = [0, 1, 1, 0]
    scores = [0.1, 0.2, 0.8, 0.9]

    ece, bins = expected_calibration_error(labels, scores, n_bins=2)

    assert ece == pytest.approx(0.35)
    assert bins[0]["count"] == 2
    assert bins[0]["confidence"] == pytest.approx(0.15)
    assert bins[0]["accuracy"] == pytest.approx(0.5)
    assert bins[1]["count"] == 2
    assert bins[1]["confidence"] == pytest.approx(0.85)
    assert bins[1]["accuracy"] == pytest.approx(0.5)


def test_best_f1_threshold_prefers_highest_f1() -> None:
    result = best_f1_threshold([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2])

    assert result["threshold"] == pytest.approx(0.7)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_threshold_for_target_tpr_reports_min_fpr() -> None:
    result = threshold_for_target_tpr(
        [0, 0, 1, 1],
        [0.2, 0.4, 0.8, 0.9],
        target_tpr=1.0,
    )

    assert result is not None
    assert result["threshold"] == pytest.approx(0.8)
    assert result["tpr"] == pytest.approx(1.0)
    assert result["fpr"] == pytest.approx(0.0)


def test_bootstrap_cis_is_deterministic_and_reports_metric_keys() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    first = bootstrap_cis(
        labels,
        scores,
        n_bootstrap=50,
        seed=7,
        metrics=("auroc", "auprc", "brier", "ece"),
    )
    second = bootstrap_cis(
        labels,
        scores,
        n_bootstrap=50,
        seed=7,
        metrics=("auroc", "auprc", "brier", "ece"),
    )

    assert first == second
    assert set(first) == {"auroc", "auprc", "brier", "ece"}
    assert first["auroc"] is not None
    assert first["auroc"]["p2_5"] <= first["auroc"]["p97_5"]


def test_paired_bootstrap_delta_uses_b_minus_a() -> None:
    labels = [0, 0, 1, 1, 0, 1]
    scores_a = [0.1, 0.7, 0.6, 0.8, 0.4, 0.5]
    scores_b = [0.1, 0.2, 0.6, 0.9, 0.3, 0.8]

    result = paired_bootstrap_delta(
        labels,
        scores_a,
        scores_b,
        metric="auroc",
        n_bootstrap=100,
        seed=11,
    )

    assert result["metric"] == "auroc"
    assert result["delta_b_minus_a"] is not None
    assert result["delta_b_minus_a"]["mean"] > 0
    assert 0 <= result["p_two_sided"] <= 1
