"""Reusable evaluation metrics.

The goal of this module is to centralise the metrics listed in
``docs/method/experiment_goals.md`` so every baseline and future model shares
the same implementation:

- Binary classification: precision, recall, F1, accuracy, AUROC, AUPRC.
- Ranking: Mean Reciprocal Rank, Top-k hit.
- Localization: IoU, boundary error, offset hit rate.

``scikit-learn`` is consulted via the optional ``[metrics]`` extra for AUROC
and AUPRC; we transparently fall back to a pure-Python implementation when it
is not installed so the core pipeline stays dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised indirectly via the ``[metrics]`` extra.
    from sklearn.metrics import (  # type: ignore[import-not-found]
        average_precision_score as _sk_average_precision_score,
        roc_auc_score as _sk_roc_auc_score,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - kept as pure-Python fallback.
    _sk_roc_auc_score = None
    _sk_average_precision_score = None
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    support: int
    positives: int
    predicted_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    # ``None`` when the metric is undefined (e.g. single-class ground truth)
    # or when scores were not provided.
    auroc: Optional[float] = None
    auprc: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RankingMetrics:
    group_count: int
    # Groups that contained at least one positive; only these contribute to
    # MRR and Top-k computations.
    evaluated_group_count: int
    mrr: float
    top_k_hit: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # JSON requires string keys.
        payload["top_k_hit"] = {str(k): v for k, v in self.top_k_hit.items()}
        return payload


@dataclass(frozen=True)
class LocalizationMetrics:
    support: int
    evaluated: int
    mean_iou: float
    mean_boundary_error: float
    offset_hit_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


def binary_classification_metrics(
    truth: Sequence[int],
    predictions: Optional[Sequence[int]] = None,
    scores: Optional[Sequence[float]] = None,
    *,
    threshold: Optional[float] = None,
    round_digits: int = 6,
) -> BinaryClassificationMetrics:
    """Compute binary classification metrics from truth + predictions / scores.

    Either ``predictions`` (hard labels 0/1) or ``scores`` must be provided.
    When only ``scores`` is given, ``threshold`` selects positive predictions.
    AUROC/AUPRC are returned when ``scores`` is available and both classes are
    present in ``truth``; otherwise they are ``None``.
    """

    if predictions is None and scores is None:
        raise ValueError("Either predictions or scores must be provided.")

    truth_list = [int(v) for v in truth]
    if predictions is not None:
        hard = [int(v) for v in predictions]
    else:
        assert scores is not None
        if threshold is None:
            raise ValueError("threshold is required when only scores are provided")
        hard = [1 if float(s) >= threshold else 0 for s in scores]

    if len(truth_list) != len(hard):
        raise ValueError("truth and predictions/scores must have the same length")

    tp = fp = fn = tn = 0
    for actual, predicted in zip(truth_list, hard):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tn += 1
    total = len(truth_list)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # Derive F1 from raw counts for numeric consistency with the rounded
    # precision / recall we expose.
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    auroc: Optional[float] = None
    auprc: Optional[float] = None
    if scores is not None:
        score_list = [float(s) for s in scores]
        if len(score_list) != len(truth_list):
            raise ValueError("truth and scores must have the same length")
        if _has_both_classes(truth_list):
            auroc = _roc_auc(truth_list, score_list)
            auprc = _average_precision(truth_list, score_list)

    return BinaryClassificationMetrics(
        support=total,
        positives=sum(1 for v in truth_list if v == 1),
        predicted_positives=sum(1 for v in hard if v == 1),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=round(precision, round_digits),
        recall=round(recall, round_digits),
        f1=round(f1, round_digits),
        accuracy=round(accuracy, round_digits),
        auroc=None if auroc is None else round(auroc, round_digits),
        auprc=None if auprc is None else round(auprc, round_digits),
    )


def _has_both_classes(truth: Sequence[int]) -> bool:
    seen_pos = False
    seen_neg = False
    for value in truth:
        if value == 1:
            seen_pos = True
        elif value == 0:
            seen_neg = True
        if seen_pos and seen_neg:
            return True
    return False


def _roc_auc(truth: Sequence[int], scores: Sequence[float]) -> float:
    if _SKLEARN_AVAILABLE:
        return float(_sk_roc_auc_score(truth, scores))  # type: ignore[misc]
    return _roc_auc_pure_python(truth, scores)


def _average_precision(truth: Sequence[int], scores: Sequence[float]) -> float:
    if _SKLEARN_AVAILABLE:
        return float(_sk_average_precision_score(truth, scores))  # type: ignore[misc]
    return _average_precision_pure_python(truth, scores)


def _roc_auc_pure_python(truth: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware ROC AUC equivalent to scikit-learn's ``roc_auc_score``.

    Uses the Mann-Whitney U formulation: average rank of positives minus
    ``(n_pos + 1) / 2``, divided by ``n_neg``.
    """

    positives = sum(1 for y in truth if y == 1)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC is undefined when only one class is present")

    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, y in zip(ranks, truth) if y == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def _average_precision_pure_python(
    truth: Sequence[int], scores: Sequence[float]
) -> float:
    """Average precision using the scikit-learn discretisation.

    ``AP = sum_n (R_n - R_{n-1}) * P_n`` where ``R_n`` / ``P_n`` are the recall
    / precision at the nth threshold (scores sorted in descending order).
    Ties are handled by grouping equal scores into the same threshold step.
    """

    if not truth:
        return 0.0
    positives = sum(1 for y in truth if y == 1)
    if positives == 0:
        raise ValueError("AUPRC is undefined when no positive samples exist")

    paired = sorted(
        zip(scores, truth),
        key=lambda pair: (-float(pair[0]), -int(pair[1])),
    )

    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    total = len(paired)
    while index < total:
        current_score = paired[index][0]
        group_tp = 0
        group_fp = 0
        while index < total and paired[index][0] == current_score:
            if paired[index][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        tp += group_tp
        fp += group_fp
        precision = tp / (tp + fp)
        recall = tp / positives
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Return 1-based ranks with ties averaged, preserving input order."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while (
            end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]
        ):
            end += 1
        # Ranks are 1-based; average over the tied group.
        average_rank = (index + end) / 2 + 1
        for pos in range(index, end + 1):
            ranks[indexed[pos][0]] = average_rank
        index = end + 1
    return ranks


# ---------------------------------------------------------------------------
# Ranking (object-level Top-k / MRR)
# ---------------------------------------------------------------------------


def ranking_metrics(
    groups: Sequence,
    truth: Sequence[int],
    scores: Sequence[float],
    *,
    ks: Sequence[int] = (1, 3, 5),
    round_digits: int = 6,
) -> RankingMetrics:
    """Compute MRR and Top-k hit per group.

    - ``groups[i]`` is the group identifier (e.g. ``apk_id``) that sample ``i``
      belongs to. All samples sharing a group are ranked together.
    - ``truth[i]`` is 1 if the sample is a positive (relevant) item.
    - ``scores[i]`` is the predicted score; higher means more relevant.
    - A group counts toward MRR/Top-k only if it contains at least one
      positive. Groups without any positive are skipped; their count is still
      reported via ``group_count``.
    """

    if not (len(groups) == len(truth) == len(scores)):
        raise ValueError("groups, truth and scores must have the same length")
    if any(k <= 0 for k in ks):
        raise ValueError("all k values must be positive")

    grouped: dict = {}
    for group, label, score in zip(groups, truth, scores):
        grouped.setdefault(group, []).append((float(score), int(label)))

    reciprocal_ranks: list[float] = []
    hits_at_k = {k: 0 for k in ks}
    evaluated = 0
    for rows in grouped.values():
        if not any(label == 1 for _, label in rows):
            continue
        evaluated += 1
        # Sort by score desc; break ties by label desc so we don't under-count
        # hits when an irrelevant item shares the same score as a relevant one
        # (this matches typical ranking evaluation conventions).
        rows_sorted = sorted(rows, key=lambda row: (-row[0], -row[1]))
        first_hit: Optional[int] = None
        for rank, (_score, label) in enumerate(rows_sorted, start=1):
            if label == 1:
                first_hit = rank
                break
        assert first_hit is not None  # guaranteed by the earlier any() check
        reciprocal_ranks.append(1.0 / first_hit)
        for k in ks:
            if first_hit <= k:
                hits_at_k[k] += 1

    mrr = sum(reciprocal_ranks) / evaluated if evaluated else 0.0
    top_k_hit = {
        k: round(hits_at_k[k] / evaluated if evaluated else 0.0, round_digits)
        for k in ks
    }
    return RankingMetrics(
        group_count=len(grouped),
        evaluated_group_count=evaluated,
        mrr=round(mrr, round_digits),
        top_k_hit=top_k_hit,
    )


# ---------------------------------------------------------------------------
# Localization (region-level offset prediction)
# ---------------------------------------------------------------------------

Interval = Tuple[int, int]


def localization_metrics(
    samples: Iterable[Mapping],
    *,
    round_digits: int = 6,
) -> LocalizationMetrics:
    """Compute mean IoU, boundary error and offset hit rate.

    Each sample is a mapping with two keys:

    - ``truth``: sequence of ``(start, end)`` payload intervals (ground truth).
    - ``prediction``: optional sequence of ``(start, end)`` intervals produced
      by the model. A sample without ground-truth intervals is skipped
      (contributes to ``support`` but not to ``evaluated``).

    Metrics:

    - ``mean_iou``: IoU between the *union* of predicted intervals and the
      union of truth intervals (handles splits).
    - ``mean_boundary_error``: ``|pred_start - truth_start| + |pred_end -
      truth_end|`` using the bounding span of the union sets; 0 for perfect
      alignment.
    - ``offset_hit_rate``: fraction of samples whose predicted union overlaps
      any truth interval.
    """

    samples_list = list(samples)
    evaluated = 0
    iou_sum = 0.0
    boundary_sum = 0.0
    hits = 0
    for sample in samples_list:
        truth_ranges = _normalize_ranges(sample.get("truth", ()))
        if not truth_ranges:
            continue
        evaluated += 1
        pred_ranges = _normalize_ranges(sample.get("prediction", ()))

        iou = _range_union_iou(pred_ranges, truth_ranges)
        iou_sum += iou
        boundary_sum += _boundary_error(pred_ranges, truth_ranges)
        if _ranges_overlap(pred_ranges, truth_ranges):
            hits += 1

    mean_iou = iou_sum / evaluated if evaluated else 0.0
    mean_boundary_error = boundary_sum / evaluated if evaluated else 0.0
    offset_hit_rate = hits / evaluated if evaluated else 0.0
    return LocalizationMetrics(
        support=len(samples_list),
        evaluated=evaluated,
        mean_iou=round(mean_iou, round_digits),
        mean_boundary_error=round(mean_boundary_error, round_digits),
        offset_hit_rate=round(offset_hit_rate, round_digits),
    )


def _normalize_ranges(ranges: Iterable) -> list[Interval]:
    normalized: list[Interval] = []
    for item in ranges:
        start, end = item
        start_int = int(start)
        end_int = int(end)
        if end_int > start_int:
            normalized.append((start_int, end_int))
    return normalized


def _merge_ranges(ranges: Sequence[Interval]) -> list[Interval]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: list[Interval] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _range_union_iou(pred: Sequence[Interval], truth: Sequence[Interval]) -> float:
    pred_merged = _merge_ranges(pred)
    truth_merged = _merge_ranges(truth)
    if not truth_merged:
        return 0.0
    intersection = _intersection_length(pred_merged, truth_merged)
    union = _union_length(pred_merged) + _union_length(truth_merged) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _intersection_length(a: Sequence[Interval], b: Sequence[Interval]) -> int:
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _union_length(ranges: Sequence[Interval]) -> int:
    return sum(end - start for start, end in ranges)


def _boundary_error(pred: Sequence[Interval], truth: Sequence[Interval]) -> float:
    truth_merged = _merge_ranges(truth)
    if not truth_merged:
        return 0.0
    truth_start = truth_merged[0][0]
    truth_end = truth_merged[-1][1]
    if not pred:
        # No prediction: the "distance" is the full extent of the truth span.
        return float((truth_end - truth_start) * 2)
    pred_merged = _merge_ranges(pred)
    pred_start = pred_merged[0][0]
    pred_end = pred_merged[-1][1]
    return float(abs(pred_start - truth_start) + abs(pred_end - truth_end))


def _ranges_overlap(pred: Sequence[Interval], truth: Sequence[Interval]) -> bool:
    for ps, pe in pred:
        for ts, te in truth:
            if max(ps, ts) < min(pe, te):
                return True
    return False


__all__ = [
    "BinaryClassificationMetrics",
    "LocalizationMetrics",
    "RankingMetrics",
    "binary_classification_metrics",
    "localization_metrics",
    "ranking_metrics",
]
