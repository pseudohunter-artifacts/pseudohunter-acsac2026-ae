"""Sanity-check heuristic baseline for hidden executable payload localization.

**This is NOT the reported rule baseline in the paper.** The external,
community-recognised rule baseline is APKiD (see
:mod:`android_packer.baselines.apkid`); that is the one that will appear
in the main comparison tables. This module exists solely as an internal
sanity check: a transparent, self-authored heuristic that is useful for

* verifying the labeling / evaluation pipeline end-to-end (if this module
  cannot beat random on synthetic data, something is wrong upstream),
* ablating which of the four object-level signals are most informative,
* providing object/region-level decisions that APKiD cannot produce.

Because the rules and weights are chosen by us, results from this module
must never be used as a stand-in for a genuine, independently validated
rule baseline when reporting to reviewers.

Signals (each with a configurable weight):

- ``suspicious_path``: the object lives outside of a code directory
  (``classes*.dex``, ``lib/``) and inside a resource-like path
  (``assets/``, ``raw/``, ``res/raw/`` ...). Packers commonly hide
  encoded payloads under ``assets/``.
- ``unknown_extension``: the object path has no extension or an extension
  unknown to our benign allowlist (``.png``, ``.jpg``, ``.mp3`` ...).
  ``path_randomized`` transforms score strongly here.
- ``large_object``: the object exceeds ``min_large_bytes`` (default
  4 KiB). Benign assets are typically small; real DEX payloads are not.
- ``low_printable_region``: the object contains at least one region with
  printable ratio below ``low_printable_threshold``. Textual benign
  resources very rarely cross this bar.

The config carries a weight for each signal and a decision threshold;
the score is the weighted sum of triggered signals. This keeps the
output in a bounded, human-readable range and makes AUROC /
thresholding well-behaved.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)

# Extensions considered safely benign in well-behaved APK assets. Anything
# outside this set is a weak "unknown extension" signal. The list is
# intentionally conservative: it covers the majority of real APK resource
# types and errs on the side of flagging more, not less.
_BENIGN_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".mp3",
        ".mp4",
        ".m4a",
        ".ogg",
        ".wav",
        ".ttf",
        ".otf",
        ".xml",
        ".json",
        ".txt",
        ".html",
        ".css",
        ".js",
        ".properties",
        ".yml",
        ".yaml",
        ".pem",
        ".cer",
        ".crt",
        ".so",  # native libs are tracked separately, not treated as payload by this baseline
    }
)

# Directory prefixes that are "resource-like" (non-code). A payload hiding
# inside one of these plus having an unknown/large body is highly suspect.
_SUSPICIOUS_PATH_PREFIXES = (
    "assets/",
    "res/raw/",
    "raw/",
    "res/xml/",
    "lib-unpacked/",
)

# Object paths (normalised with forward slashes) that we treat as code and
# therefore *never* flag on path alone. Everything starting with these is
# assumed legitimate code storage.
_CODE_PATH_PREFIXES = (
    "classes",  # classes.dex, classes2.dex, ...
    "lib/",
)


@dataclass(frozen=True)
class SanityRulesConfig:
    """Weights and thresholds for the sanity-check heuristic baseline.

    All weights must be non-negative. ``threshold`` is applied to the sum
    of triggered weights; in the default config any single "strong" signal
    is enough to flag, while two "weak" signals together also suffice.
    """

    threshold: float = 1.0
    suspicious_path_weight: float = 0.6
    unknown_extension_weight: float = 0.6
    large_object_weight: float = 0.4
    low_printable_weight: float = 0.4
    min_large_bytes: int = 4096
    low_printable_threshold: float = 0.1

    def max_score(self) -> float:
        return (
            self.suspicious_path_weight
            + self.unknown_extension_weight
            + self.large_object_weight
            + self.low_printable_weight
        )


@dataclass(frozen=True)
class SanityRulesRegionPrediction:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    object_type: str
    offset_start: int
    offset_end: int
    score: float
    threshold: float
    predicted_label_id: int
    true_label_id: int
    true_label: str
    triggered_rules: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["triggered_rules"] = list(self.triggered_rules)
        return payload


@dataclass(frozen=True)
class SanityRulesObjectPrediction:
    apk_id: str
    object_id: str
    object_path: str
    object_type: str
    score: float
    threshold: float
    predicted_label_id: int
    true_label_id: int
    region_count: int
    predicted_region_count: int
    positive_region_count: int
    predicted_offset_start: Optional[int]
    predicted_offset_end: Optional[int]
    triggered_rules: tuple[str, ...] = ()
    object_size: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["triggered_rules"] = list(self.triggered_rules)
        return payload


@dataclass(frozen=True)
class SanityRulesApkPrediction:
    apk_id: str
    score: float
    threshold: float
    predicted_label_id: int
    true_label_id: int
    object_count: int
    predicted_object_count: int
    positive_object_count: int
    region_count: int
    predicted_region_count: int
    positive_region_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SanityRulesResult:
    region_predictions: List[SanityRulesRegionPrediction]
    object_predictions: List[SanityRulesObjectPrediction]
    apk_predictions: List[SanityRulesApkPrediction]
    report: dict


def run_sanity_rules_baseline(
    region_labels: Iterable[Mapping],
    config: Optional[SanityRulesConfig] = None,
) -> SanityRulesResult:
    """Run the sanity-check heuristic over a stream of region training labels.

    Input rows are expected to follow the ``region_training_label`` schema:
    ``apk_id``, ``object_id``, ``region_id``, ``object_path``,
    ``object_type``, ``offset_start``, ``offset_end``, ``size``,
    ``printable_ratio``, ``label``, ``label_id`` are read.
    """

    config = config or SanityRulesConfig()
    _validate_config(config)

    # Group regions by object first: the rule baseline's primitives are
    # object-level, and region predictions are broadcast from the object's
    # decision.
    object_rows: dict[tuple[str, str], list[Mapping]] = {}
    for row in region_labels:
        object_rows.setdefault((str(row["apk_id"]), str(row["object_id"])), []).append(row)

    region_predictions: list[SanityRulesRegionPrediction] = []
    object_predictions: list[SanityRulesObjectPrediction] = []

    for (apk_id, object_id), rows in sorted(object_rows.items()):
        first = rows[0]
        object_size = _object_size(rows)
        triggered, score = _score_object(first, rows, object_size, config)
        predicted_label_id = 1 if score >= config.threshold else 0

        for row in rows:
            region_predictions.append(
                SanityRulesRegionPrediction(
                    apk_id=str(row["apk_id"]),
                    object_id=str(row["object_id"]),
                    region_id=str(row["region_id"]),
                    object_path=str(row["object_path"]),
                    object_type=str(row["object_type"]),
                    offset_start=int(row["offset_start"]),
                    offset_end=int(row["offset_end"]),
                    score=score,
                    threshold=config.threshold,
                    # Broadcast the object-level decision onto every region.
                    predicted_label_id=predicted_label_id,
                    true_label_id=int(row["label_id"]),
                    true_label=str(row["label"]),
                    triggered_rules=triggered,
                )
            )

        positive_rows = [row for row in rows if int(row["label_id"])]
        offset_start: Optional[int] = None
        offset_end: Optional[int] = None
        if predicted_label_id:
            # When the object is flagged, emit the full object span as the
            # predicted localization. The rule baseline has no sub-object
            # localization signal, so we surface the trivial span rather
            # than faking finer granularity.
            offset_start = min(int(row["offset_start"]) for row in rows)
            offset_end = max(int(row["offset_end"]) for row in rows)

        object_predictions.append(
            SanityRulesObjectPrediction(
                apk_id=apk_id,
                object_id=object_id,
                object_path=str(first["object_path"]),
                object_type=str(first["object_type"]),
                score=round(score, 6),
                threshold=config.threshold,
                predicted_label_id=predicted_label_id,
                true_label_id=1 if positive_rows else 0,
                region_count=len(rows),
                predicted_region_count=len(rows) if predicted_label_id else 0,
                positive_region_count=len(positive_rows),
                predicted_offset_start=offset_start,
                predicted_offset_end=offset_end,
                triggered_rules=triggered,
                object_size=object_size,
            )
        )

    apk_predictions = _aggregate_apks(region_predictions, object_predictions, config.threshold)

    report = {
        "baseline": "sanity_rules",
        "config": asdict(config),
        "counts": {
            "regions": len(region_predictions),
            "objects": len(object_predictions),
            "apks": len(apk_predictions),
        },
        "metrics": {
            "region": _binary_metric_dict(
                truth=[row.true_label_id for row in region_predictions],
                predictions=[row.predicted_label_id for row in region_predictions],
                scores=[row.score for row in region_predictions],
            ),
            "object": _binary_metric_dict(
                truth=[row.true_label_id for row in object_predictions],
                predictions=[row.predicted_label_id for row in object_predictions],
                scores=[row.score for row in object_predictions],
            ),
            "apk": _binary_metric_dict(
                truth=[row.true_label_id for row in apk_predictions],
                predictions=[row.predicted_label_id for row in apk_predictions],
                scores=[row.score for row in apk_predictions],
            ),
        },
        "ranking": {
            "object": ranking_metrics(
                groups=[row.apk_id for row in object_predictions],
                truth=[row.true_label_id for row in object_predictions],
                scores=[row.score for row in object_predictions],
            ).to_dict(),
        },
        "localization": {
            "object": localization_metrics(
                _object_localization_samples(region_predictions, object_predictions)
            ).to_dict(),
        },
        "rule_trigger_counts": _rule_trigger_counts(object_predictions),
    }

    return SanityRulesResult(
        region_predictions=region_predictions,
        object_predictions=object_predictions,
        apk_predictions=apk_predictions,
        report=report,
    )


def _validate_config(config: SanityRulesConfig) -> None:
    for name in (
        "threshold",
        "suspicious_path_weight",
        "unknown_extension_weight",
        "large_object_weight",
        "low_printable_weight",
        "min_large_bytes",
        "low_printable_threshold",
    ):
        value = getattr(config, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if config.low_printable_threshold > 1.0:
        raise ValueError("low_printable_threshold must be <= 1.0")


def _object_size(rows: Sequence[Mapping]) -> int:
    if not rows:
        return 0
    return max(int(row["offset_end"]) for row in rows) - min(
        int(row["offset_start"]) for row in rows
    )


def _score_object(
    first_row: Mapping,
    rows: Sequence[Mapping],
    object_size: int,
    config: SanityRulesConfig,
) -> tuple[tuple[str, ...], float]:
    triggered: list[str] = []
    score = 0.0

    object_path = str(first_row["object_path"]).replace("\\", "/")
    path_lower = object_path.lower()

    if _is_suspicious_path(path_lower):
        triggered.append("suspicious_path")
        score += config.suspicious_path_weight

    if _has_unknown_extension(path_lower):
        triggered.append("unknown_extension")
        score += config.unknown_extension_weight

    if object_size >= config.min_large_bytes and not _is_code_path(path_lower):
        triggered.append("large_object")
        score += config.large_object_weight

    if any(
        float(row["printable_ratio"]) <= config.low_printable_threshold
        for row in rows
    ) and not _is_code_path(path_lower):
        triggered.append("low_printable_region")
        score += config.low_printable_weight

    return tuple(triggered), score


def _is_suspicious_path(path_lower: str) -> bool:
    if _is_code_path(path_lower):
        return False
    return any(path_lower.startswith(prefix) for prefix in _SUSPICIOUS_PATH_PREFIXES)


def _is_code_path(path_lower: str) -> bool:
    return any(path_lower.startswith(prefix) for prefix in _CODE_PATH_PREFIXES)


def _has_unknown_extension(path_lower: str) -> bool:
    if _is_code_path(path_lower):
        return False
    _, ext = os.path.splitext(path_lower)
    if not ext:
        return True
    return ext not in _BENIGN_EXTENSIONS


def _aggregate_apks(
    region_predictions: Sequence[SanityRulesRegionPrediction],
    object_predictions: Sequence[SanityRulesObjectPrediction],
    threshold: float,
) -> List[SanityRulesApkPrediction]:
    regions_by_apk: dict[str, list[SanityRulesRegionPrediction]] = {}
    objects_by_apk: dict[str, list[SanityRulesObjectPrediction]] = {}
    for row in region_predictions:
        regions_by_apk.setdefault(row.apk_id, []).append(row)
    for row in object_predictions:
        objects_by_apk.setdefault(row.apk_id, []).append(row)

    predictions: list[SanityRulesApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = round(max((row.score for row in objects), default=0.0), 6)
        predicted_objects = sum(row.predicted_label_id for row in objects)
        positive_objects = sum(row.true_label_id for row in objects)
        predictions.append(
            SanityRulesApkPrediction(
                apk_id=apk_id,
                score=score,
                threshold=threshold,
                predicted_label_id=1 if score >= threshold else 0,
                true_label_id=1 if positive_objects else 0,
                object_count=len(objects),
                predicted_object_count=predicted_objects,
                positive_object_count=positive_objects,
                region_count=len(regions),
                predicted_region_count=sum(row.predicted_label_id for row in regions),
                positive_region_count=sum(row.true_label_id for row in regions),
            )
        )
    return predictions


def _binary_metric_dict(
    *,
    truth: Sequence[int],
    predictions: Sequence[int],
    scores: Optional[Sequence[float]] = None,
) -> dict:
    return binary_classification_metrics(
        truth=truth,
        predictions=predictions,
        scores=scores,
    ).to_dict()


def _object_localization_samples(
    regions: Sequence[SanityRulesRegionPrediction],
    objects: Sequence[SanityRulesObjectPrediction],
) -> list[dict]:
    regions_by_object: dict[tuple[str, str], list[SanityRulesRegionPrediction]] = {}
    for region in regions:
        regions_by_object.setdefault((region.apk_id, region.object_id), []).append(region)

    samples: list[dict] = []
    for obj in objects:
        if not obj.true_label_id:
            continue
        rows = regions_by_object.get((obj.apk_id, obj.object_id), [])
        truth = [
            (row.offset_start, row.offset_end)
            for row in rows
            if row.true_label_id == 1
        ]
        prediction: list[tuple[int, int]] = []
        if obj.predicted_offset_start is not None and obj.predicted_offset_end is not None:
            prediction.append((obj.predicted_offset_start, obj.predicted_offset_end))
        samples.append({"truth": truth, "prediction": prediction})
    return samples


def _rule_trigger_counts(objects: Sequence[SanityRulesObjectPrediction]) -> dict[str, int]:
    """Report how often each rule fired across all objects.

    Useful for sanity-checking whether a single dominant rule is driving
    the entire score (which would be a red flag during ablation).
    """

    counts: dict[str, int] = {}
    for obj in objects:
        for rule in obj.triggered_rules:
            counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "SanityRulesApkPrediction",
    "SanityRulesConfig",
    "SanityRulesObjectPrediction",
    "SanityRulesRegionPrediction",
    "SanityRulesResult",
    "run_sanity_rules_baseline",
]
