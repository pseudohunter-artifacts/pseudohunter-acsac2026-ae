"""Late-fusion baseline combining Typed-Instance MIL and byte-CNN scores.

The fusion keeps both component models untouched: callers provide an Ours/MIL
result and a byte-CNN result for the same region rows, and this module blends
their calibrated probabilities into the standard region/object/APK prediction
schema used by the multi-baseline runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)


__all__ = [
    "MilByteCnnFusionApkPrediction",
    "MilByteCnnFusionConfig",
    "MilByteCnnFusionObjectPrediction",
    "MilByteCnnFusionRegionPrediction",
    "MilByteCnnFusionResult",
    "fuse_mil_and_byte_cnn_results",
]


@dataclass(frozen=True)
class MilByteCnnFusionConfig:
    """Score-level fusion knobs.

    ``mil_weight`` and ``byte_cnn_weight`` are normalised before blending, so
    any positive scale is accepted.  ``score_transform='identity'`` preserves
    each component probability; ``'logit_average'`` averages in logit space and
    maps back through sigmoid, which can be useful when both scores are already
    calibrated probabilities.
    """

    mil_weight: float = 0.5
    byte_cnn_weight: float = 0.5
    threshold: float = 0.5
    score_transform: str = "identity"


@dataclass(frozen=True)
class MilByteCnnFusionRegionPrediction:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    offset_start: int
    offset_end: int
    score: float
    predicted_label_id: int
    true_label_id: int
    mil_score: float
    byte_cnn_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MilByteCnnFusionObjectPrediction:
    apk_id: str
    object_id: str
    object_path: str
    score: float
    predicted_label_id: int
    true_label_id: int
    region_count: int
    positive_region_count: int
    mil_score: float
    byte_cnn_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MilByteCnnFusionApkPrediction:
    apk_id: str
    score: float
    predicted_label_id: int
    true_label_id: int
    object_count: int
    positive_object_count: int
    region_count: int
    positive_region_count: int
    mil_score: float
    byte_cnn_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MilByteCnnFusionResult:
    region_predictions: List[MilByteCnnFusionRegionPrediction]
    object_predictions: List[MilByteCnnFusionObjectPrediction]
    apk_predictions: List[MilByteCnnFusionApkPrediction]
    report: dict


def fuse_mil_and_byte_cnn_results(
    *,
    mil_result: Any,
    byte_cnn_result: Any,
    config: MilByteCnnFusionConfig | None = None,
) -> MilByteCnnFusionResult:
    """Blend MIL and byte-CNN predictions over matching region keys."""

    cfg = config or MilByteCnnFusionConfig()
    _validate_config(cfg)

    mil_by_key = {_region_key(row): row for row in mil_result.region_predictions}
    byte_by_key = {_region_key(row): row for row in byte_cnn_result.region_predictions}
    if set(mil_by_key) != set(byte_by_key):
        missing_mil = sorted(set(byte_by_key) - set(mil_by_key))[:5]
        missing_byte = sorted(set(mil_by_key) - set(byte_by_key))[:5]
        raise ValueError(
            "MIL and byte_cnn predictions must cover the same region keys; "
            f"missing_from_mil={missing_mil}, missing_from_byte_cnn={missing_byte}"
        )

    region_predictions: List[MilByteCnnFusionRegionPrediction] = []
    for key in sorted(mil_by_key):
        mil_row = mil_by_key[key]
        byte_row = byte_by_key[key]
        if int(mil_row.true_label_id) != int(byte_row.true_label_id):
            raise ValueError(f"truth mismatch for region key {key!r}")
        fused_score = _blend_scores(float(mil_row.score), float(byte_row.score), cfg)
        region_predictions.append(
            MilByteCnnFusionRegionPrediction(
                apk_id=str(mil_row.apk_id),
                object_id=str(mil_row.object_id),
                region_id=str(mil_row.region_id),
                object_path=str(mil_row.object_path),
                offset_start=int(mil_row.offset_start),
                offset_end=int(mil_row.offset_end),
                score=fused_score,
                predicted_label_id=1 if fused_score >= cfg.threshold else 0,
                true_label_id=int(mil_row.true_label_id),
                mil_score=round(float(mil_row.score), 6),
                byte_cnn_score=round(float(byte_row.score), 6),
            )
        )

    object_predictions = _aggregate_objects(region_predictions, cfg.threshold)
    apk_predictions = _aggregate_apks(
        region_predictions,
        object_predictions,
        cfg.threshold,
    )
    report = _build_report(region_predictions, object_predictions, apk_predictions, cfg)
    return MilByteCnnFusionResult(
        region_predictions=region_predictions,
        object_predictions=object_predictions,
        apk_predictions=apk_predictions,
        report=report,
    )


def _validate_config(cfg: MilByteCnnFusionConfig) -> None:
    if cfg.mil_weight < 0.0 or cfg.byte_cnn_weight < 0.0:
        raise ValueError("fusion weights must be non-negative")
    if cfg.mil_weight + cfg.byte_cnn_weight <= 0.0:
        raise ValueError("at least one fusion weight must be positive")
    if not 0.0 <= cfg.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {cfg.threshold}")
    if cfg.score_transform not in ("identity", "logit_average"):
        raise ValueError(
            "score_transform must be one of 'identity', 'logit_average'; "
            f"got {cfg.score_transform!r}"
        )


def _region_key(row: Any) -> Tuple[str, str, str, int, int]:
    return (
        str(row.apk_id),
        str(row.object_id),
        str(row.region_id),
        int(row.offset_start),
        int(row.offset_end),
    )


def _blend_scores(mil_score: float, byte_cnn_score: float, cfg: MilByteCnnFusionConfig) -> float:
    total = cfg.mil_weight + cfg.byte_cnn_weight
    mil_w = cfg.mil_weight / total
    byte_w = cfg.byte_cnn_weight / total
    if cfg.score_transform == "identity":
        blended = mil_w * mil_score + byte_w * byte_cnn_score
    else:
        blended = _sigmoid(mil_w * _logit(mil_score) + byte_w * _logit(byte_cnn_score))
    return round(float(blended), 6)


def _logit(score: float) -> float:
    import math

    clipped = min(max(float(score), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-float(value)))


def _aggregate_objects(
    region_preds: Sequence[MilByteCnnFusionRegionPrediction],
    threshold: float,
) -> List[MilByteCnnFusionObjectPrediction]:
    groups: Dict[Tuple[str, str], List[MilByteCnnFusionRegionPrediction]] = {}
    for region in region_preds:
        groups.setdefault((region.apk_id, region.object_id), []).append(region)

    out: List[MilByteCnnFusionObjectPrediction] = []
    for (apk_id, object_id), regions in sorted(groups.items()):
        score = max(region.score for region in regions)
        mil_score = max(region.mil_score for region in regions)
        byte_cnn_score = max(region.byte_cnn_score for region in regions)
        true_label = 1 if any(region.true_label_id for region in regions) else 0
        out.append(
            MilByteCnnFusionObjectPrediction(
                apk_id=apk_id,
                object_id=object_id,
                object_path=regions[0].object_path,
                score=round(score, 6),
                predicted_label_id=1 if score >= threshold else 0,
                true_label_id=true_label,
                region_count=len(regions),
                positive_region_count=sum(region.true_label_id for region in regions),
                mil_score=round(mil_score, 6),
                byte_cnn_score=round(byte_cnn_score, 6),
            )
        )
    return out


def _aggregate_apks(
    region_preds: Sequence[MilByteCnnFusionRegionPrediction],
    object_preds: Sequence[MilByteCnnFusionObjectPrediction],
    threshold: float,
) -> List[MilByteCnnFusionApkPrediction]:
    regions_by_apk: Dict[str, List[MilByteCnnFusionRegionPrediction]] = {}
    objects_by_apk: Dict[str, List[MilByteCnnFusionObjectPrediction]] = {}
    for region in region_preds:
        regions_by_apk.setdefault(region.apk_id, []).append(region)
    for obj in object_preds:
        objects_by_apk.setdefault(obj.apk_id, []).append(obj)

    out: List[MilByteCnnFusionApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = max((obj.score for obj in objects), default=0.0)
        mil_score = max((obj.mil_score for obj in objects), default=0.0)
        byte_cnn_score = max((obj.byte_cnn_score for obj in objects), default=0.0)
        true_label = 1 if any(obj.true_label_id for obj in objects) else 0
        out.append(
            MilByteCnnFusionApkPrediction(
                apk_id=apk_id,
                score=round(score, 6),
                predicted_label_id=1 if score >= threshold else 0,
                true_label_id=true_label,
                object_count=len(objects),
                positive_object_count=sum(obj.true_label_id for obj in objects),
                region_count=len(regions),
                positive_region_count=sum(region.true_label_id for region in regions),
                mil_score=round(mil_score, 6),
                byte_cnn_score=round(byte_cnn_score, 6),
            )
        )
    return out


def _object_localization_samples(
    regions: Sequence[MilByteCnnFusionRegionPrediction],
    objects: Sequence[MilByteCnnFusionObjectPrediction],
) -> List[dict]:
    by_object: Dict[Tuple[str, str], List[MilByteCnnFusionRegionPrediction]] = {}
    for region in regions:
        by_object.setdefault((region.apk_id, region.object_id), []).append(region)

    samples: List[dict] = []
    for obj in objects:
        if not obj.true_label_id:
            continue
        rows = by_object.get((obj.apk_id, obj.object_id), [])
        truth = [
            (region.offset_start, region.offset_end)
            for region in rows
            if region.true_label_id == 1
        ]
        positive_regions = [region for region in rows if region.predicted_label_id == 1]
        prediction: List[Tuple[int, int]] = []
        if positive_regions:
            prediction.append(
                (
                    min(region.offset_start for region in positive_regions),
                    max(region.offset_end for region in positive_regions),
                )
            )
        samples.append({"truth": truth, "prediction": prediction})
    return samples


def _build_report(
    region_preds: Sequence[MilByteCnnFusionRegionPrediction],
    object_preds: Sequence[MilByteCnnFusionObjectPrediction],
    apk_preds: Sequence[MilByteCnnFusionApkPrediction],
    cfg: MilByteCnnFusionConfig,
) -> dict:
    region_metrics = binary_classification_metrics(
        truth=[region.true_label_id for region in region_preds],
        predictions=[region.predicted_label_id for region in region_preds],
        scores=[region.score for region in region_preds],
    )
    object_metrics = binary_classification_metrics(
        truth=[obj.true_label_id for obj in object_preds],
        predictions=[obj.predicted_label_id for obj in object_preds],
        scores=[obj.score for obj in object_preds],
    )
    apk_metrics = binary_classification_metrics(
        truth=[apk.true_label_id for apk in apk_preds],
        predictions=[apk.predicted_label_id for apk in apk_preds],
        scores=[apk.score for apk in apk_preds],
    )
    ranking = ranking_metrics(
        groups=[obj.apk_id for obj in object_preds],
        truth=[obj.true_label_id for obj in object_preds],
        scores=[obj.score for obj in object_preds],
    )
    localization = localization_metrics(_object_localization_samples(region_preds, object_preds))
    total = cfg.mil_weight + cfg.byte_cnn_weight
    return {
        "baseline": "mil_byte_cnn_fusion",
        "threshold": cfg.threshold,
        "fusion": {
            "mil_weight": cfg.mil_weight,
            "byte_cnn_weight": cfg.byte_cnn_weight,
            "normalised_mil_weight": cfg.mil_weight / total,
            "normalised_byte_cnn_weight": cfg.byte_cnn_weight / total,
            "score_transform": cfg.score_transform,
        },
        "counts": {
            "regions": len(region_preds),
            "objects": len(object_preds),
            "apks": len(apk_preds),
        },
        "metrics": {
            "region": region_metrics.to_dict(),
            "object": object_metrics.to_dict(),
            "apk": apk_metrics.to_dict(),
        },
        "ranking": {"object": ranking.to_dict()},
        "localization": {"object": localization.to_dict()},
    }
