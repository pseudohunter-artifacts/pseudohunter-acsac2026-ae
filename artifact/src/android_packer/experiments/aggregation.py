"""Shared summary-aggregation helpers for baseline experiment runners.

Extracted from the original :mod:`android_packer.cli.run_synthetic_entropy_baseline`
module so that the new multi-baseline runner can reuse the exact same
macro / micro averaging rules without copy-paste divergence.

The helpers operate on task reports shaped like the per-baseline
report dicts produced by :mod:`android_packer.baselines.*`, i.e.

    {
        "metrics": {
            "region": {...}, "object": {...}, "apk": {...},
        },
        "ranking": {"object": {...}},
        "localization": {"object": {...}},
    }

Tasks whose report does not carry one of those keys are simply
skipped, which matches the way the APKiD baseline (APK-only) degrades.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence


def aggregate_reports(reports: Sequence[Mapping]) -> dict:
    """Aggregate a list of per-task baseline reports.

    Returns the structure previously produced by
    ``_aggregate_reports`` inside the synthetic entropy runner,
    generalised so that absent ``region`` / ``object`` levels do not
    raise (APKiD only emits ``apk``).
    """

    return {
        "task_count": len(reports),
        "metrics": {
            level: _aggregate_level_metrics(reports, level)
            for level in ("region", "object", "apk")
        },
        "ranking": {
            "object": _aggregate_ranking_metrics(reports),
        },
        "localization": {
            "object": _aggregate_localization_metrics(reports),
        },
    }


def macro_average(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------


def _aggregate_level_metrics(reports: Sequence[Mapping], level: str) -> dict:
    totals = {
        "support": 0,
        "positives": 0,
        "predicted_positives": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
    }
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    contributing = 0
    for report in reports:
        metrics = report.get("metrics", {}).get(level)
        if metrics is None:
            # Baseline does not report this level (e.g. APKiD). Skip.
            continue
        contributing += 1
        for key in totals:
            totals[key] += int(metrics[key])
        auroc = metrics.get("auroc")
        if auroc is not None:
            auroc_values.append(float(auroc))
        auprc = metrics.get("auprc")
        if auprc is not None:
            auprc_values.append(float(auprc))

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    tn = totals["true_negatives"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_div(tp + tn, totals["support"])
    return {
        **totals,
        "contributing_task_count": contributing,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "auroc": macro_average(auroc_values),
        "auprc": macro_average(auprc_values),
        "auroc_task_count": len(auroc_values),
        "auprc_task_count": len(auprc_values),
    }


def _aggregate_ranking_metrics(reports: Sequence[Mapping]) -> dict:
    mrr_values: list[float] = []
    evaluated_groups_total = 0
    group_count_total = 0
    top_k_accumulator: dict[str, list[float]] = {}
    for report in reports:
        ranking = report.get("ranking", {}).get("object")
        if not ranking:
            continue
        mrr_values.append(float(ranking["mrr"]))
        evaluated_groups_total += int(ranking["evaluated_group_count"])
        group_count_total += int(ranking["group_count"])
        for k, hit in ranking.get("top_k_hit", {}).items():
            top_k_accumulator.setdefault(str(k), []).append(float(hit))
    return {
        "task_count": len(mrr_values),
        "group_count": group_count_total,
        "evaluated_group_count": evaluated_groups_total,
        "mrr": macro_average(mrr_values),
        "top_k_hit": {
            k: macro_average(values)
            for k, values in sorted(
                top_k_accumulator.items(), key=lambda item: int(item[0])
            )
        },
    }


def _aggregate_localization_metrics(reports: Sequence[Mapping]) -> dict:
    iou_values: list[float] = []
    boundary_values: list[float] = []
    hit_values: list[float] = []
    support_total = 0
    evaluated_total = 0
    for report in reports:
        localization = report.get("localization", {}).get("object")
        if not localization:
            continue
        iou_values.append(float(localization["mean_iou"]))
        boundary_values.append(float(localization["mean_boundary_error"]))
        hit_values.append(float(localization["offset_hit_rate"]))
        support_total += int(localization["support"])
        evaluated_total += int(localization["evaluated"])
    return {
        "task_count": len(iou_values),
        "support": support_total,
        "evaluated": evaluated_total,
        "mean_iou": macro_average(iou_values),
        "mean_boundary_error": macro_average(boundary_values),
        "offset_hit_rate": macro_average(hit_values),
    }


__all__ = [
    "aggregate_reports",
    "macro_average",
    "safe_div",
]
