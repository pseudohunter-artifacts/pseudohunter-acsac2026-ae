"""Byte-CNN baseline for region-level hidden payload localization.

This is the lightweight byte-only neural baseline that fills the project
matrix slot between n-gram LR and the heavier Ours-MIL path. It keeps the
same output dataclass schema as ``ngram_logreg`` and
``payload_hunter_lite`` so the multi-baseline runner can aggregate it
without special-casing.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)
from android_packer.features import ObjectByteLoader
from android_packer.models.byte_cnn import (
    ByteCnnRegionScorerConfig,
    build_byte_cnn_region_scorer,
)


__all__ = [
    "ByteCnnApkPrediction",
    "ByteCnnBaselineConfig",
    "ByteCnnModel",
    "ByteCnnObjectPrediction",
    "ByteCnnRegionPrediction",
    "ByteCnnResult",
    "run_byte_cnn_baseline",
    "train_byte_cnn",
]


_SUPPORTED_TRAIN_MODES = ("same_set", "holdout_transform", "holdout_package")
_MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ByteCnnBaselineConfig:
    """Training and inference knobs for :func:`train_byte_cnn`."""

    model_config: ByteCnnRegionScorerConfig = field(
        default_factory=ByteCnnRegionScorerConfig
    )
    train_mode: str = "same_set"
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_class_weight: float = 10.0
    random_state: int = 0
    verbose: bool = False
    threshold: float = 0.5
    loader_cache_size: int = 8192
    device: str = "auto"


@dataclass(frozen=True)
class ByteCnnRegionPrediction:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    offset_start: int
    offset_end: int
    score: float
    predicted_label_id: int
    true_label_id: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ByteCnnObjectPrediction:
    apk_id: str
    object_id: str
    object_path: str
    score: float
    predicted_label_id: int
    true_label_id: int
    region_count: int
    positive_region_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ByteCnnApkPrediction:
    apk_id: str
    score: float
    predicted_label_id: int
    true_label_id: int
    object_count: int
    positive_object_count: int
    region_count: int
    positive_region_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ByteCnnResult:
    region_predictions: List[ByteCnnRegionPrediction]
    object_predictions: List[ByteCnnObjectPrediction]
    apk_predictions: List[ByteCnnApkPrediction]
    report: dict


class ByteCnnModel:
    """Trained byte-CNN region scorer wrapper."""

    def __init__(self, scorer: Any, config: ByteCnnBaselineConfig) -> None:
        self._scorer = scorer
        self._config = config

    @property
    def config(self) -> ByteCnnBaselineConfig:
        return self._config

    @property
    def threshold(self) -> float:
        return self._config.threshold

    def save(self, path: Path) -> None:
        torch, _ = _require_torch()
        payload = {
            "version": _MODEL_FORMAT_VERSION,
            "config": asdict(self._config),
            "scorer_state_dict": {
                key: value.detach().cpu()
                for key, value in self._scorer.state_dict().items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "ByteCnnModel":
        torch, _ = _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("version") != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported byte_cnn model format version {payload.get('version')!r}; "
                f"expected {_MODEL_FORMAT_VERSION}"
            )
        cfg = _config_from_dict(payload["config"])
        device = _resolve_device(cfg.device)
        scorer = build_byte_cnn_region_scorer(cfg.model_config)
        scorer.load_state_dict(payload["scorer_state_dict"])
        scorer.to(device)
        scorer.eval()
        return cls(scorer=scorer, config=cfg)

    def predict(
        self,
        region_rows: Iterable[Mapping],
        apk_index: Mapping[str, Path],
        loader: Optional[ObjectByteLoader] = None,
        threshold: Optional[float] = None,
    ) -> ByteCnnResult:
        torch, _ = _require_torch()
        rows = [dict(row) for row in region_rows]
        if not rows:
            return _empty_result()

        device = _resolve_device(self._config.device)
        self._scorer.to(device)
        self._scorer.eval()
        resolved_loader = loader or ObjectByteLoader(
            cache_size=self._config.loader_cache_size
        )
        decision_threshold = self.threshold if threshold is None else float(threshold)

        scores: List[float] = []
        batch_size = max(1, self._config.batch_size)
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                token_batch = _rows_to_token_tensor(
                    batch_rows,
                    apk_index=apk_index,
                    loader=resolved_loader,
                    config=self._config.model_config,
                    device=device,
                )
                logits = self._scorer(token_batch)
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                scores.extend(float(round(p, 6)) for p in probs)

        region_preds = _build_region_predictions(rows, scores, decision_threshold)
        object_preds = _aggregate_objects(region_preds)
        apk_preds = _aggregate_apks(region_preds, object_preds)
        report = _build_report(region_preds, object_preds, apk_preds, decision_threshold)
        return ByteCnnResult(
            region_predictions=region_preds,
            object_predictions=object_preds,
            apk_predictions=apk_preds,
            report=report,
        )


def train_byte_cnn(
    region_rows: Iterable[Mapping],
    apk_index: Mapping[str, Path],
    config: Optional[ByteCnnBaselineConfig] = None,
) -> ByteCnnModel:
    """Fit a byte-CNN model on region labels.

    Only ``train_mode='same_set'`` is accepted here. Holdout modes are
    orchestrated by :func:`run_byte_cnn_baseline` or the multi-baseline
    runner, both of which train one same-set model per fold.
    """

    torch, nn = _require_torch()
    cfg = config or ByteCnnBaselineConfig()
    if cfg.train_mode != "same_set":
        raise ValueError(
            f"train_byte_cnn only accepts train_mode='same_set'; got {cfg.train_mode!r}. "
            "Use run_byte_cnn_baseline for holdout modes."
        )
    rows = [dict(row) for row in region_rows]
    if not rows:
        raise ValueError("train_byte_cnn received no training rows")

    labels = [int(row.get("label_id", 0)) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError(
            "Training data contains a single class; cannot fit byte_cnn."
        )

    torch.manual_seed(cfg.random_state)
    device = _resolve_device(cfg.device)
    scorer = build_byte_cnn_region_scorer(cfg.model_config)
    scorer.to(device)
    scorer.train()

    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    pos_weight = torch.tensor(
        [cfg.positive_class_weight], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = ObjectByteLoader(cache_size=cfg.loader_cache_size)

    n = len(rows)
    batch_size = max(1, min(cfg.batch_size, n))
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

    for epoch in range(cfg.epochs):
        generator = torch.Generator()
        generator.manual_seed(cfg.random_state * 1000 + epoch)
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_rows = [rows[int(i)] for i in batch_idx.tolist()]
            token_batch = _rows_to_token_tensor(
                batch_rows,
                apk_index=apk_index,
                loader=loader,
                config=cfg.model_config,
                device=device,
            )
            yb = labels_tensor[batch_idx.to(device)]
            logits = scorer(token_batch)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        if cfg.verbose and num_batches:
            import logging

            logging.getLogger(__name__).info(
                "byte_cnn epoch=%d loss=%.6f", epoch, epoch_loss / num_batches
            )

    scorer.eval()
    return ByteCnnModel(scorer=scorer, config=cfg)


def run_byte_cnn_baseline(
    region_rows: Iterable[Mapping],
    apk_index: Mapping[str, Path],
    config: Optional[ByteCnnBaselineConfig] = None,
) -> ByteCnnResult:
    """Train and predict under same-set or leave-one-group-out modes."""

    cfg = config or ByteCnnBaselineConfig()
    if cfg.train_mode not in _SUPPORTED_TRAIN_MODES:
        raise ValueError(
            f"unsupported train_mode {cfg.train_mode!r}; expected one of {_SUPPORTED_TRAIN_MODES}"
        )
    rows = [dict(row) for row in region_rows]
    if not rows:
        return _empty_result()

    if cfg.train_mode == "same_set":
        warnings.warn(
            "byte_cnn train_mode='same_set' produces in-sample numbers; "
            "use holdout_transform or holdout_package for OOD estimates.",
            stacklevel=2,
        )
        model = train_byte_cnn(rows, apk_index, _with_train_mode(cfg, "same_set"))
        return model.predict(rows, apk_index)

    group_key = (
        "transform_family" if cfg.train_mode == "holdout_transform" else "package_name"
    )
    for row in rows:
        if group_key not in row:
            if group_key == "package_name":
                row["package_name"] = _derive_package_name(str(row["apk_id"]))
            else:
                raise KeyError(
                    f"holdout_transform requires every row to carry {group_key!r}; "
                    f"missing in apk_id={row.get('apk_id')!r}"
                )

    groups: Dict[str, List[Mapping]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    if len(groups) < 2:
        raise ValueError(
            f"train_mode={cfg.train_mode!r} requires >= 2 distinct {group_key!r} "
            f"values; got {len(groups)}"
        )

    all_region_preds: List[ByteCnnRegionPrediction] = []
    all_object_preds: List[ByteCnnObjectPrediction] = []
    all_apk_preds: List[ByteCnnApkPrediction] = []
    inner_cfg = _with_train_mode(cfg, "same_set")
    for held in sorted(groups):
        train_rows = [row for group, items in groups.items() if group != held for row in items]
        test_rows = groups[held]
        model = train_byte_cnn(train_rows, apk_index, inner_cfg)
        fold_result = model.predict(test_rows, apk_index)
        all_region_preds.extend(fold_result.region_predictions)
        all_object_preds.extend(fold_result.object_predictions)
        all_apk_preds.extend(fold_result.apk_predictions)

    stitched_report = _build_report(
        all_region_preds,
        all_object_preds,
        all_apk_preds,
        cfg.threshold,
        extra={"train_mode": cfg.train_mode, "folds": sorted(groups)},
    )
    return ByteCnnResult(
        region_predictions=all_region_preds,
        object_predictions=all_object_preds,
        apk_predictions=all_apk_preds,
        report=stitched_report,
    )


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "byte_cnn baseline requires torch. Install via ``pip install -e .[dl]``."
        ) from exc
    return torch, nn


def _resolve_device(device_str: str) -> Any:
    torch, _ = _require_torch()
    s = (device_str or "auto").strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda" or s.startswith("cuda:"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"byte_cnn device={device_str!r} requested but CUDA is unavailable; falling back to CPU.",
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device(s)
    warnings.warn(
        f"byte_cnn: unknown device={device_str!r}; falling back to CPU.",
        stacklevel=2,
    )
    return torch.device("cpu")


def _rows_to_token_tensor(
    rows: Sequence[Mapping],
    *,
    apk_index: Mapping[str, Path],
    loader: ObjectByteLoader,
    config: ByteCnnRegionScorerConfig,
    device: Any,
) -> Any:
    torch, _ = _require_torch()
    pad = int(config.pad_token_id)
    max_len = int(config.max_length)
    encoded: List[List[int]] = []
    for row in rows:
        apk_id = str(row.get("apk_id", ""))
        if apk_id not in apk_index:
            raise KeyError(
                "byte_cnn cannot load region bytes because apk_id "
                f"{apk_id!r} is missing from apk_index"
            )
        data = loader.region_bytes(
            Path(apk_index[apk_id]),
            str(row["object_path"]),
            int(row["offset_start"]),
            int(row["offset_end"]),
        )
        token_ids = list(data[:max_len])
        if len(token_ids) < max_len:
            token_ids.extend([pad] * (max_len - len(token_ids)))
        encoded.append(token_ids)
    return torch.tensor(encoded, dtype=torch.long, device=device)


def _with_train_mode(
    cfg: ByteCnnBaselineConfig, mode: str
) -> ByteCnnBaselineConfig:
    return ByteCnnBaselineConfig(
        model_config=cfg.model_config,
        train_mode=mode,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        positive_class_weight=cfg.positive_class_weight,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
        threshold=cfg.threshold,
        loader_cache_size=cfg.loader_cache_size,
        device=cfg.device,
    )


def _config_from_dict(d: Mapping[str, Any]) -> ByteCnnBaselineConfig:
    model_raw = dict(d.get("model_config", {}))
    if "kernel_sizes" in model_raw:
        model_raw["kernel_sizes"] = tuple(model_raw["kernel_sizes"])
    model_config = ByteCnnRegionScorerConfig(**model_raw)
    return ByteCnnBaselineConfig(
        model_config=model_config,
        train_mode=d.get("train_mode", "same_set"),
        epochs=int(d.get("epochs", 5)),
        batch_size=int(d.get("batch_size", 64)),
        learning_rate=float(d.get("learning_rate", 1e-3)),
        weight_decay=float(d.get("weight_decay", 1e-4)),
        positive_class_weight=float(d.get("positive_class_weight", 10.0)),
        random_state=int(d.get("random_state", 0)),
        verbose=bool(d.get("verbose", False)),
        threshold=float(d.get("threshold", 0.5)),
        loader_cache_size=int(d.get("loader_cache_size", 8192)),
        device=str(d.get("device", "auto")),
    )


def _derive_package_name(apk_id: str) -> str:
    parts = apk_id.split("_")
    head: List[str] = []
    for part in parts:
        if part.isdigit() or (len(part) >= 6 and all(c in "0123456789abcdef" for c in part)):
            break
        head.append(part)
    return ".".join(head) if head else apk_id


def _build_region_predictions(
    rows: Sequence[Mapping], scores: Sequence[float], threshold: float
) -> List[ByteCnnRegionPrediction]:
    preds: List[ByteCnnRegionPrediction] = []
    for row, score in zip(rows, scores):
        preds.append(
            ByteCnnRegionPrediction(
                apk_id=str(row["apk_id"]),
                object_id=str(row["object_id"]),
                region_id=str(row["region_id"]),
                object_path=str(row["object_path"]),
                offset_start=int(row["offset_start"]),
                offset_end=int(row["offset_end"]),
                score=score,
                predicted_label_id=1 if score >= threshold else 0,
                true_label_id=int(row["label_id"]),
            )
        )
    return preds


def _aggregate_objects(
    region_preds: Sequence[ByteCnnRegionPrediction],
) -> List[ByteCnnObjectPrediction]:
    groups: Dict[Tuple[str, str], List[ByteCnnRegionPrediction]] = {}
    for region in region_preds:
        groups.setdefault((region.apk_id, region.object_id), []).append(region)

    out: List[ByteCnnObjectPrediction] = []
    for (apk_id, object_id), regions in sorted(groups.items()):
        score = max(region.score for region in regions)
        predicted = max(region.predicted_label_id for region in regions)
        true_label = 1 if any(region.true_label_id for region in regions) else 0
        out.append(
            ByteCnnObjectPrediction(
                apk_id=apk_id,
                object_id=object_id,
                object_path=regions[0].object_path,
                score=round(score, 6),
                predicted_label_id=predicted,
                true_label_id=true_label,
                region_count=len(regions),
                positive_region_count=sum(region.true_label_id for region in regions),
            )
        )
    return out


def _aggregate_apks(
    region_preds: Sequence[ByteCnnRegionPrediction],
    object_preds: Sequence[ByteCnnObjectPrediction],
) -> List[ByteCnnApkPrediction]:
    regions_by_apk: Dict[str, List[ByteCnnRegionPrediction]] = {}
    objects_by_apk: Dict[str, List[ByteCnnObjectPrediction]] = {}
    for region in region_preds:
        regions_by_apk.setdefault(region.apk_id, []).append(region)
    for obj in object_preds:
        objects_by_apk.setdefault(obj.apk_id, []).append(obj)

    out: List[ByteCnnApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = max((obj.score for obj in objects), default=0.0)
        predicted = max((obj.predicted_label_id for obj in objects), default=0)
        true_label = 1 if any(obj.true_label_id for obj in objects) else 0
        out.append(
            ByteCnnApkPrediction(
                apk_id=apk_id,
                score=round(score, 6),
                predicted_label_id=predicted,
                true_label_id=true_label,
                object_count=len(objects),
                positive_object_count=sum(obj.true_label_id for obj in objects),
                region_count=len(regions),
                positive_region_count=sum(region.true_label_id for region in regions),
            )
        )
    return out


def _object_localization_samples(
    regions: Sequence[ByteCnnRegionPrediction],
    objects: Sequence[ByteCnnObjectPrediction],
) -> List[dict]:
    by_object: Dict[Tuple[str, str], List[ByteCnnRegionPrediction]] = {}
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
        prediction: List[Tuple[int, int]] = []
        positive_regions = [region for region in rows if region.predicted_label_id == 1]
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
    region_preds: Sequence[ByteCnnRegionPrediction],
    object_preds: Sequence[ByteCnnObjectPrediction],
    apk_preds: Sequence[ByteCnnApkPrediction],
    threshold: float,
    *,
    extra: Optional[Mapping[str, Any]] = None,
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
    localization = localization_metrics(
        _object_localization_samples(region_preds, object_preds)
    )
    report = {
        "baseline": "byte_cnn",
        "threshold": threshold,
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
        "calibration": {
            "region": _threshold_operating_points(
                [region.true_label_id for region in region_preds],
                [region.score for region in region_preds],
                threshold,
            ),
            "object": _threshold_operating_points(
                [obj.true_label_id for obj in object_preds],
                [obj.score for obj in object_preds],
                threshold,
            ),
            "apk": _threshold_operating_points(
                [apk.true_label_id for apk in apk_preds],
                [apk.score for apk in apk_preds],
                threshold,
            ),
            "object_top_k": _object_top_k_operating_points(object_preds),
        },
    }
    if extra:
        report.update(dict(extra))
    return report


def _threshold_operating_points(
    truth: Sequence[int],
    scores: Sequence[float],
    default_threshold: float,
    *,
    round_digits: int = 6,
) -> dict:
    """Summarise how score threshold choice affects detector F1."""

    truth_list = [int(v) for v in truth]
    score_list = [float(v) for v in scores]
    if len(truth_list) != len(score_list):
        raise ValueError("truth and scores must have the same length")

    default_metrics = binary_classification_metrics(
        truth=truth_list,
        scores=score_list,
        threshold=float(default_threshold),
        round_digits=round_digits,
    ).to_dict()
    payload = {
        "default_threshold": round(float(default_threshold), round_digits),
        "at_default_threshold": default_metrics,
        "best_f1": None,
    }
    if not truth_list:
        return payload

    positives = sum(truth_list)
    paired = sorted(zip(score_list, truth_list), key=lambda item: item[0], reverse=True)
    best: Optional[dict] = None
    true_positives = 0
    false_positives = 0
    idx = 0
    while idx < len(paired):
        threshold = paired[idx][0]
        group_true_positives = 0
        group_false_positives = 0
        while idx < len(paired) and paired[idx][0] == threshold:
            if paired[idx][1] == 1:
                group_true_positives += 1
            else:
                group_false_positives += 1
            idx += 1
        true_positives += group_true_positives
        false_positives += group_false_positives
        false_negatives = positives - true_positives
        predicted_positives = true_positives + false_positives
        precision = true_positives / predicted_positives if predicted_positives else 0.0
        recall = true_positives / positives if positives else 0.0
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = (2 * true_positives / denominator) if denominator else 0.0
        current = {
            "threshold": round(float(threshold), round_digits),
            "precision": round(float(precision), round_digits),
            "recall": round(float(recall), round_digits),
            "f1": round(float(f1), round_digits),
            "predicted_positives": predicted_positives,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        if best is None or _is_better_threshold_candidate(current, best):
            best = current
    payload["best_f1"] = best
    return payload


def _is_better_threshold_candidate(current: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
    return (
        float(current["f1"]),
        float(current["precision"]),
        float(current["threshold"]),
    ) > (
        float(best["f1"]),
        float(best["precision"]),
        float(best["threshold"]),
    )


def _object_top_k_operating_points(
    object_preds: Sequence[ByteCnnObjectPrediction],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict:
    """Treat the top-k objects per APK as positives and report detector metrics."""

    if not object_preds:
        return {str(k): None for k in ks}

    by_apk: Dict[str, List[ByteCnnObjectPrediction]] = {}
    for obj in object_preds:
        by_apk.setdefault(obj.apk_id, []).append(obj)

    out: dict[str, dict] = {}
    truth = [obj.true_label_id for obj in object_preds]
    for k in ks:
        selected: set[Tuple[str, str]] = set()
        for apk_id, rows in by_apk.items():
            ranked = sorted(
                rows,
                key=lambda obj: (-obj.score, obj.object_id),
            )
            for obj in ranked[: min(k, len(ranked))]:
                selected.add((apk_id, obj.object_id))
        predictions = [
            1 if (obj.apk_id, obj.object_id) in selected else 0
            for obj in object_preds
        ]
        metrics = binary_classification_metrics(
            truth=truth,
            predictions=predictions,
            scores=[obj.score for obj in object_preds],
        ).to_dict()
        out[str(k)] = metrics
    return out


def _empty_result() -> ByteCnnResult:
    return ByteCnnResult(
        region_predictions=[],
        object_predictions=[],
        apk_predictions=[],
        report={
            "baseline": "byte_cnn",
            "counts": {"regions": 0, "objects": 0, "apks": 0},
            "metrics": {},
        },
    )
