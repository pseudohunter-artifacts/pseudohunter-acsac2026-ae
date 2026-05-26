"""Byte-level n-gram + logistic regression baseline.

Learning-based counterpart to the hand-designed entropy / rule
baselines:

- Inputs: region training label rows (as produced by
  ``build-training-labels``) plus an APK index that maps
  ``apk_id -> apk_path`` so that raw bytes can be pulled from disk.
- Features: byte unigram histogram, hashed bigram histogram, and a
  small set of scalar statistics (entropy, printable ratio, etc.),
  all computed by :mod:`android_packer.features.byte_features`.
- Model: sklearn LogisticRegression on top of a DictVectorizer. The
  solver is ``liblinear`` + ``class_weight="balanced"`` which handles
  the hidden-payload class imbalance with a deterministic fit.
- Outputs: three-layer predictions + report identical in shape to
  the entropy and sanity_rules baselines, so the evaluation module,
  downstream dashboards and the multi-baseline runner can consume
  this baseline without special-casing.

The sklearn dependency lives behind a lazy ``import`` inside
:func:`train_ngram_logreg` / :meth:`NgramLogRegModel.load` so that
importing this module never costs numpy startup for pipelines that
only run the stdlib-only baselines.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)
from android_packer.features import (
    ByteFeatureConfig,
    ObjectByteLoader,
    region_byte_features,
)


# Serialised model format version. Bumped whenever the pickle layout
# changes so that ``load`` can refuse old artefacts rather than load
# them into a subtly different runtime.
_MODEL_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Public dataclasses (mirror the entropy / sanity_rules baseline shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NgramLogRegConfig:
    """Hyperparameters for the n-gram LR baseline.

    The defaults are chosen for the MVP: bigram enabled with 1 024
    buckets (matches :class:`ByteFeatureConfig` defaults), L2 penalty
    with ``C=1.0``, and ``class_weight="balanced"`` so that the rare
    hidden-payload class is not steamrolled by benign regions.
    """

    feature_config: ByteFeatureConfig = field(default_factory=ByteFeatureConfig)
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str = "balanced"
    random_state: int = 0
    # Decision threshold on ``P(positive)``. Kept explicit so the CLI
    # can expose it and so sensitivity analyses can sweep it without
    # retraining the model.
    threshold: float = 0.5
    # ObjectByteLoader cache size. 64 means we keep up to 64 distinct
    # (apk, object) buffers resident, which is comfortable for the
    # 4 KiB / stride 2 KiB default windowing over ~20 objects per APK.
    loader_cache_size: int = 64
    # F0b: Vectorizer backend. ``True`` uses sklearn's ``FeatureHasher``
    # which keeps memory constant regardless of sample count (no feature
    # vocabulary), solving the DictVectorizer OOM observed at ~160k
    # regions. Set to ``False`` only in tests that compare equivalence
    # on small inputs.
    use_hashing_vectorizer: bool = True
    # Number of hashing buckets used by FeatureHasher. 2**18 = 262144 is
    # large enough to keep collisions negligible for our feature names
    # (256 unigram + up to ``bigram_hash_dim`` bigram + ~20 scalars).
    hashing_n_features: int = 262144


@dataclass(frozen=True)
class NgramLogRegRegionPrediction:
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
class NgramLogRegObjectPrediction:
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
class NgramLogRegApkPrediction:
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
class NgramLogRegResult:
    region_predictions: List[NgramLogRegRegionPrediction]
    object_predictions: List[NgramLogRegObjectPrediction]
    apk_predictions: List[NgramLogRegApkPrediction]
    report: dict


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------


class NgramLogRegModel:
    """Wrapper around a fitted sklearn pipeline.

    Holds the (DictVectorizer, LogisticRegression) pair plus the
    exact :class:`ByteFeatureConfig` used at training time, so that
    :meth:`predict` applies identical feature extraction. The
    vectoriser's learned feature order lives inside DictVectorizer
    itself; we do not need to carry a second copy.
    """

    def __init__(
        self,
        vectorizer,
        classifier,
        feature_config: ByteFeatureConfig,
        threshold: float,
    ) -> None:
        self._vectorizer = vectorizer
        self._classifier = classifier
        self._feature_config = feature_config
        self._threshold = float(threshold)

    @property
    def feature_config(self) -> ByteFeatureConfig:
        return self._feature_config

    @property
    def threshold(self) -> float:
        return self._threshold

    # ----- I/O -----

    def save(self, path: Path) -> None:
        payload = {
            "version": _MODEL_FORMAT_VERSION,
            "vectorizer": self._vectorizer,
            "classifier": self._classifier,
            "feature_config": asdict(self._feature_config),
            "threshold": self._threshold,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "NgramLogRegModel":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        version = payload.get("version")
        if version != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {version!r}; "
                f"expected {_MODEL_FORMAT_VERSION}"
            )
        return cls(
            vectorizer=payload["vectorizer"],
            classifier=payload["classifier"],
            feature_config=ByteFeatureConfig(**payload["feature_config"]),
            threshold=payload["threshold"],
        )

    # ----- Inference -----

    def predict(
        self,
        region_rows: Iterable[Mapping],
        apk_index: Mapping[str, Path],
        loader: Optional[ObjectByteLoader] = None,
    ) -> NgramLogRegResult:
        """Score a stream of region rows and produce a full report."""

        if loader is None:
            loader = ObjectByteLoader(cache_size=max(1, len(apk_index) * 2 or 1))

        rows = list(region_rows)
        scores = _score_rows(
            rows,
            apk_index=apk_index,
            vectorizer=self._vectorizer,
            classifier=self._classifier,
            feature_config=self._feature_config,
            loader=loader,
        )

        region_preds = _build_region_predictions(rows, scores, self._threshold)
        object_preds = _aggregate_objects(region_preds)
        apk_preds = _aggregate_apks(region_preds, object_preds)
        report = _build_report(region_preds, object_preds, apk_preds, self._threshold)
        return NgramLogRegResult(
            region_predictions=region_preds,
            object_predictions=object_preds,
            apk_predictions=apk_preds,
            report=report,
        )


def train_ngram_logreg(
    region_rows: Iterable[Mapping],
    apk_index: Mapping[str, Path],
    config: Optional[NgramLogRegConfig] = None,
) -> NgramLogRegModel:
    """Fit the baseline on region training labels.

    Every row must carry ``apk_id``, ``object_path``, ``offset_start``,
    ``offset_end`` and ``label_id``. Rows whose ``apk_id`` is missing
    from ``apk_index`` are skipped with no warning; callers are
    expected to validate coverage upstream (the CLI surfaces it).
    """

    # Lazy sklearn import keeps pipelines that don't need the learned
    # baseline free of the numpy startup tax.
    from sklearn.feature_extraction import DictVectorizer, FeatureHasher  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore

    cfg = config or NgramLogRegConfig()
    loader = ObjectByteLoader(cache_size=cfg.loader_cache_size)

    feature_dicts: list[dict] = []
    labels: list[int] = []
    for row in region_rows:
        apk_id = str(row["apk_id"])
        if apk_id not in apk_index:
            continue
        bytes_ = loader.region_bytes(
            Path(apk_index[apk_id]),
            str(row["object_path"]),
            int(row["offset_start"]),
            int(row["offset_end"]),
        )
        fv = region_byte_features(bytes_, config=cfg.feature_config)
        feature_dicts.append(fv.to_dict())
        labels.append(int(row["label_id"]))

    if not feature_dicts:
        raise ValueError(
            "No training rows were usable; verify apk_index covers the input apk_ids."
        )
    if len(set(labels)) < 2:
        raise ValueError(
            "Training data contains a single class; cannot fit a binary classifier."
        )

    # F0b: FeatureHasher keeps memory constant vs. DictVectorizer which
    # builds an O(n_features) vocabulary. For ~160k regions the former
    # ran OOM on 16 GB machines; FeatureHasher resolves that at the cost
    # of O(1/n_features) hash collisions, which are negligible at 2**18.
    if cfg.use_hashing_vectorizer:
        vectorizer = FeatureHasher(
            n_features=cfg.hashing_n_features,
            input_type="dict",
            # alternate_sign=False gives us non-negative counts, which
            # matches the DictVectorizer semantics our features assume
            # (every feature is a raw count or a non-negative scalar).
            alternate_sign=False,
        )
        X = vectorizer.transform(feature_dicts)
    else:
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(feature_dicts)
    y = labels

    classifier = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        solver="liblinear",
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )
    classifier.fit(X, y)

    return NgramLogRegModel(
        vectorizer=vectorizer,
        classifier=classifier,
        feature_config=cfg.feature_config,
        threshold=cfg.threshold,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_rows(
    rows: Sequence[Mapping],
    *,
    apk_index: Mapping[str, Path],
    vectorizer,
    classifier,
    feature_config: ByteFeatureConfig,
    loader: ObjectByteLoader,
) -> List[float]:
    """Return ``P(positive)`` for every row, with per-row bytes loading."""

    feature_dicts: list[dict] = []
    for row in rows:
        apk_id = str(row["apk_id"])
        if apk_id not in apk_index:
            # Unknown APK: emit zeroed feature dict so the row still
            # aligns 1:1 with the scoring matrix. Its score will be
            # whatever the model predicts for an all-zero input, which
            # is deterministic and documented.
            feature_dicts.append({})
            continue
        bytes_ = loader.region_bytes(
            Path(apk_index[apk_id]),
            str(row["object_path"]),
            int(row["offset_start"]),
            int(row["offset_end"]),
        )
        feature_dicts.append(
            region_byte_features(bytes_, config=feature_config).to_dict()
        )

    X = vectorizer.transform(feature_dicts)
    # predict_proba returns [P(neg), P(pos)]; always take column 1.
    proba = classifier.predict_proba(X)[:, 1]
    return [float(round(p, 6)) for p in proba]


def _build_region_predictions(
    rows: Sequence[Mapping],
    scores: Sequence[float],
    threshold: float,
) -> List[NgramLogRegRegionPrediction]:
    preds: list[NgramLogRegRegionPrediction] = []
    for row, score in zip(rows, scores):
        preds.append(
            NgramLogRegRegionPrediction(
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
    region_preds: Sequence[NgramLogRegRegionPrediction],
) -> List[NgramLogRegObjectPrediction]:
    # Group by (apk_id, object_id); object score = max region score,
    # which matches the entropy baseline's aggregation rule and keeps
    # AUROC / MRR comparisons on equal footing.
    groups: dict[tuple[str, str], list[NgramLogRegRegionPrediction]] = {}
    for r in region_preds:
        groups.setdefault((r.apk_id, r.object_id), []).append(r)

    out: list[NgramLogRegObjectPrediction] = []
    for (apk_id, object_id), regions in sorted(groups.items()):
        score = max(r.score for r in regions)
        predicted = max(r.predicted_label_id for r in regions)
        true_label = 1 if any(r.true_label_id for r in regions) else 0
        positive_regions = sum(r.true_label_id for r in regions)
        out.append(
            NgramLogRegObjectPrediction(
                apk_id=apk_id,
                object_id=object_id,
                object_path=regions[0].object_path,
                score=round(score, 6),
                predicted_label_id=predicted,
                true_label_id=true_label,
                region_count=len(regions),
                positive_region_count=positive_regions,
            )
        )
    return out


def _aggregate_apks(
    region_preds: Sequence[NgramLogRegRegionPrediction],
    object_preds: Sequence[NgramLogRegObjectPrediction],
) -> List[NgramLogRegApkPrediction]:
    regions_by_apk: dict[str, list[NgramLogRegRegionPrediction]] = {}
    objects_by_apk: dict[str, list[NgramLogRegObjectPrediction]] = {}
    for r in region_preds:
        regions_by_apk.setdefault(r.apk_id, []).append(r)
    for o in object_preds:
        objects_by_apk.setdefault(o.apk_id, []).append(o)

    out: list[NgramLogRegApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = max((o.score for o in objects), default=0.0)
        predicted = max((o.predicted_label_id for o in objects), default=0)
        true_label = 1 if any(o.true_label_id for o in objects) else 0
        out.append(
            NgramLogRegApkPrediction(
                apk_id=apk_id,
                score=round(score, 6),
                predicted_label_id=predicted,
                true_label_id=true_label,
                object_count=len(objects),
                positive_object_count=sum(o.true_label_id for o in objects),
                region_count=len(regions),
                positive_region_count=sum(r.true_label_id for r in regions),
            )
        )
    return out


def _object_localization_samples(
    regions: Sequence[NgramLogRegRegionPrediction],
    objects: Sequence[NgramLogRegObjectPrediction],
) -> list[dict]:
    """Build ``localization_metrics`` samples for every positive object.

    Matches the :mod:`baselines.entropy` contract:

    - ``truth``: union of ``(offset_start, offset_end)`` intervals from
      the object's true-positive regions.
    - ``prediction``: list containing a single span ``[min_start,
      max_end]`` over the regions the model flagged positive. An empty
      list when the model fires on none of the object's regions, which
      mean_iou correctly handles as a miss.

    Negative-true objects are skipped (no ground truth to score against).
    """

    by_object: dict[tuple[str, str], list[NgramLogRegRegionPrediction]] = {}
    for r in regions:
        by_object.setdefault((r.apk_id, r.object_id), []).append(r)

    samples: list[dict] = []
    for obj in objects:
        if not obj.true_label_id:
            continue
        rows = by_object.get((obj.apk_id, obj.object_id), [])
        truth = [
            (r.offset_start, r.offset_end)
            for r in rows
            if r.true_label_id == 1
        ]
        prediction: list[tuple[int, int]] = []
        positive_regions = [r for r in rows if r.predicted_label_id == 1]
        if positive_regions:
            prediction.append(
                (
                    min(r.offset_start for r in positive_regions),
                    max(r.offset_end for r in positive_regions),
                )
            )
        samples.append({"truth": truth, "prediction": prediction})
    return samples


def _build_report(
    region_preds: Sequence[NgramLogRegRegionPrediction],
    object_preds: Sequence[NgramLogRegObjectPrediction],
    apk_preds: Sequence[NgramLogRegApkPrediction],
    threshold: float,
) -> dict:
    region_metrics = binary_classification_metrics(
        truth=[r.true_label_id for r in region_preds],
        predictions=[r.predicted_label_id for r in region_preds],
        scores=[r.score for r in region_preds],
    )
    object_metrics = binary_classification_metrics(
        truth=[o.true_label_id for o in object_preds],
        predictions=[o.predicted_label_id for o in object_preds],
        scores=[o.score for o in object_preds],
    )
    apk_metrics = binary_classification_metrics(
        truth=[a.true_label_id for a in apk_preds],
        predictions=[a.predicted_label_id for a in apk_preds],
        scores=[a.score for a in apk_preds],
    )

    # Ranking: per-APK, object-level ranking by score.
    ranking = ranking_metrics(
        groups=[o.apk_id for o in object_preds],
        truth=[o.true_label_id for o in object_preds],
        scores=[o.score for o in object_preds],
    )

    # Localization: object-level IoU / boundary error using the top
    # positive region per object.
    loc_samples = _object_localization_samples(region_preds, object_preds)
    localization = localization_metrics(loc_samples)

    counts = {
        "regions": len(region_preds),
        "objects": len(object_preds),
        "apks": len(apk_preds),
    }
    return {
        "baseline": "ngram_logreg",
        "threshold": threshold,
        "counts": counts,
        "metrics": {
            "region": region_metrics.to_dict(),
            "object": object_metrics.to_dict(),
            "apk": apk_metrics.to_dict(),
        },
        "ranking": {"object": ranking.to_dict()},
        "localization": {"object": localization.to_dict()},
    }


__all__ = [
    "NgramLogRegApkPrediction",
    "NgramLogRegConfig",
    "NgramLogRegModel",
    "NgramLogRegObjectPrediction",
    "NgramLogRegRegionPrediction",
    "NgramLogRegResult",
    "train_ngram_logreg",
]
