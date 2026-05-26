"""Entropy-threshold baseline for region/object/APK localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)


@dataclass(frozen=True)
class EntropyBaselineConfig:
    entropy_threshold: float = 7.0
    entropy_weight: float = 1.0
    nonprintable_weight: float = 0.0


@dataclass(frozen=True)
class RegionPrediction:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    object_type: str
    offset_start: int
    offset_end: int
    entropy: float
    printable_ratio: float
    score: float
    threshold: float
    predicted_label_id: int
    true_label_id: int
    true_label: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPrediction:
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApkPrediction:
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
class EntropyBaselineResult:
    region_predictions: List[RegionPrediction]
    object_predictions: List[ObjectPrediction]
    apk_predictions: List[ApkPrediction]
    report: dict


def run_entropy_baseline(
    region_labels: Iterable[Mapping],
    config: Optional[EntropyBaselineConfig] = None,
) -> EntropyBaselineResult:
    config = config or EntropyBaselineConfig()
    if config.entropy_threshold < 0.0:
        raise ValueError("entropy_threshold must be non-negative")
    if config.entropy_weight < 0.0:
        raise ValueError("entropy_weight must be non-negative")
    if config.nonprintable_weight < 0.0:
        raise ValueError("nonprintable_weight must be non-negative")

    regions = [_predict_region(row, config) for row in region_labels]
    objects = _aggregate_objects(regions, config.entropy_threshold)
    apks = _aggregate_apks(regions, objects, config.entropy_threshold)
    report = {
        "baseline": "entropy_threshold",
        "config": asdict(config),
        "counts": {
            "regions": len(regions),
            "objects": len(objects),
            "apks": len(apks),
        },
        "metrics": {
            "region": _binary_metric_dict(
                truth=[row.true_label_id for row in regions],
                predictions=[row.predicted_label_id for row in regions],
                scores=[row.score for row in regions],
            ),
            "object": _binary_metric_dict(
                truth=[row.true_label_id for row in objects],
                predictions=[row.predicted_label_id for row in objects],
                scores=[row.score for row in objects],
            ),
            "apk": _binary_metric_dict(
                truth=[row.true_label_id for row in apks],
                predictions=[row.predicted_label_id for row in apks],
                scores=[row.score for row in apks],
            ),
        },
        # Object-level ranking per APK, to align with experiment_goals.md's
        # Top-k / MRR main indicators.
        "ranking": {
            "object": ranking_metrics(
                groups=[row.apk_id for row in objects],
                truth=[row.true_label_id for row in objects],
                scores=[row.score for row in objects],
            ).to_dict(),
        },
        # Object-level localization: IoU / boundary error / offset hit rate
        # between the union of positive regions (truth) and the bounding box
        # of predicted-positive regions (prediction). Only positive objects
        # contribute to the averages; negatives have no ground-truth span.
        "localization": {
            "object": localization_metrics(
                _object_localization_samples(regions, objects)
            ).to_dict(),
        },
    }
    return EntropyBaselineResult(
        region_predictions=regions,
        object_predictions=objects,
        apk_predictions=apks,
        report=report,
    )


def score_region(
    entropy: float,
    printable_ratio: float,
    config: Optional[EntropyBaselineConfig] = None,
) -> float:
    config = config or EntropyBaselineConfig()
    return round(
        (float(entropy) * config.entropy_weight)
        + ((1.0 - float(printable_ratio)) * config.nonprintable_weight),
        6,
    )


def _predict_region(row: Mapping, config: EntropyBaselineConfig) -> RegionPrediction:
    score = score_region(
        entropy=float(row["entropy"]),
        printable_ratio=float(row["printable_ratio"]),
        config=config,
    )
    return RegionPrediction(
        apk_id=str(row["apk_id"]),
        object_id=str(row["object_id"]),
        region_id=str(row["region_id"]),
        object_path=str(row["object_path"]),
        object_type=str(row["object_type"]),
        offset_start=int(row["offset_start"]),
        offset_end=int(row["offset_end"]),
        entropy=float(row["entropy"]),
        printable_ratio=float(row["printable_ratio"]),
        score=score,
        threshold=config.entropy_threshold,
        predicted_label_id=1 if score >= config.entropy_threshold else 0,
        true_label_id=int(row["label_id"]),
        true_label=str(row["label"]),
    )


def _aggregate_objects(
    region_predictions: Sequence[RegionPrediction],
    threshold: float,
) -> List[ObjectPrediction]:
    groups: dict[tuple[str, str], list[RegionPrediction]] = {}
    for row in region_predictions:
        groups.setdefault((row.apk_id, row.object_id), []).append(row)

    predictions: list[ObjectPrediction] = []
    for (_apk_id, _object_id), rows in sorted(groups.items()):
        first = rows[0]
        predicted_rows = [row for row in rows if row.predicted_label_id]
        positive_rows = [row for row in rows if row.true_label_id]
        score = round(max(row.score for row in rows), 6)
        predictions.append(
            ObjectPrediction(
                apk_id=first.apk_id,
                object_id=first.object_id,
                object_path=first.object_path,
                object_type=first.object_type,
                score=score,
                threshold=threshold,
                # Drive the hard label from the score so it stays consistent
                # with both the ranking view and AUROC: ``score >= threshold``
                # implies at least one positive region under the same rule.
                predicted_label_id=1 if score >= threshold else 0,
                true_label_id=1 if positive_rows else 0,
                region_count=len(rows),
                predicted_region_count=len(predicted_rows),
                positive_region_count=len(positive_rows),
                predicted_offset_start=(
                    min(row.offset_start for row in predicted_rows)
                    if predicted_rows
                    else None
                ),
                predicted_offset_end=(
                    max(row.offset_end for row in predicted_rows)
                    if predicted_rows
                    else None
                ),
            )
        )
    return predictions


def _aggregate_apks(
    region_predictions: Sequence[RegionPrediction],
    object_predictions: Sequence[ObjectPrediction],
    threshold: float,
) -> List[ApkPrediction]:
    regions_by_apk: dict[str, list[RegionPrediction]] = {}
    objects_by_apk: dict[str, list[ObjectPrediction]] = {}
    for row in region_predictions:
        regions_by_apk.setdefault(row.apk_id, []).append(row)
    for row in object_predictions:
        objects_by_apk.setdefault(row.apk_id, []).append(row)

    predictions: list[ApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = round(max((row.score for row in objects), default=0.0), 6)
        predicted_objects = sum(row.predicted_label_id for row in objects)
        positive_objects = sum(row.true_label_id for row in objects)
        predictions.append(
            ApkPrediction(
                apk_id=apk_id,
                score=score,
                threshold=threshold,
                # Keep the hard label in sync with the reported score so that
                # AUROC / precision / recall remain consistent with the binary
                # decision surface.
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
    regions: Sequence[RegionPrediction],
    objects: Sequence[ObjectPrediction],
) -> list[dict]:
    """Build ``localization_metrics`` samples for every positive object.

    - Truth = union of ``(offset_start, offset_end)`` intervals of the
      object's true-positive regions.
    - Prediction = the bounding ``[min_start, max_end]`` span built from the
      object's regions whose baseline score crosses the threshold (already
      summarised on :class:`ObjectPrediction`).

    Negative objects are skipped because they have no ground-truth span.
    """

    regions_by_object: dict[tuple[str, str], list[RegionPrediction]] = {}
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
