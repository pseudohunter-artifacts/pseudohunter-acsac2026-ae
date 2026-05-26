"""PayloadHunter-Lite baseline (F-Lite-c).

Public entry points:

* :class:`PayloadHunterLiteConfig` — hyperparameters + ``train_mode``.
* :func:`train_payload_hunter_lite` — fit a :class:`PayloadHunterLiteModel`
  on region rows + label_id.
* :class:`PayloadHunterLiteModel` — trained scorer+aggregator bundle;
  ``predict`` returns :class:`PayloadHunterLiteResult` with shape
  mirroring :mod:`baselines.ngram_logreg` so downstream reporting /
  :func:`run_synthetic_multi_baseline` stay symmetric.

Design anchors:

* Method spec: ``docs/method/ours_method_spec.md`` §11 (PayloadHunter-Lite).
* Feature assembly: :mod:`android_packer.features.handcrafted`
  (Pass-2a, 15 dims; Pass-2b will extend to 34 dims without changing
  this module's API).
* Torch lazy-import contract: keep the core pipeline installable
  without the ``[dl]`` optional extra; this module raises a clean
  :class:`ImportError` only when :func:`train_payload_hunter_lite` or
  :meth:`PayloadHunterLiteModel.predict` is actually called without
  torch present.
* Three ``train_mode`` values match the spec §4 data contract and
  ``docs/method/dataset_plan.md`` §2.4:

  - ``same_set`` — train = test (in-sample upper bound; emits a
    ``same_set`` warning in the report matching ``ngram_logreg``'s
    semantics).
  - ``holdout_transform`` — leave-one-transform-family-out; requires
    ``transform_family`` to be present on every row.
  - ``holdout_package`` — leave-one-package-out; requires
    ``package_name`` (derived from ``apk_id`` if absent).

  Holdout modes emit **N sub-models** (one per held-out group); test
  predictions are concatenated so the returned 
  :class:`PayloadHunterLiteResult` is a single stitched report.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from android_packer.evaluation import (
    binary_classification_metrics,
    localization_metrics,
    ranking_metrics,
)
from android_packer.features.byte_features import ObjectByteLoader
from android_packer.features.handcrafted import (
    HandcraftedFeatureConfig,
    extract_handcrafted_features,
    handcrafted_feature_names,
)
from android_packer.models.payload_hunter_lite import (
    LiteObjectAggregatorConfig,
    LiteRegionScorerConfig,
    build_lite_object_aggregator,
    build_lite_region_scorer,
)


__all__ = [
    "PayloadHunterLiteApkPrediction",
    "PayloadHunterLiteConfig",
    "PayloadHunterLiteModel",
    "PayloadHunterLiteObjectPrediction",
    "PayloadHunterLiteRegionPrediction",
    "PayloadHunterLiteResult",
    "train_payload_hunter_lite",
    "run_payload_hunter_lite_baseline",
]


_SUPPORTED_TRAIN_MODES = ("same_set", "holdout_transform", "holdout_package")
_MODEL_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Public dataclasses (mirror ngram_logreg shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadHunterLiteConfig:
    """Hyperparameters for PayloadHunter-Lite training + inference.

    Defaults are chosen to match :mod:`docs/method/ours_method_spec.md`
    §11 (feature_dim=15 for Pass-2a handcrafted assembly, hidden_dim
    128, 2 hidden layers, AdamW lr=1e-3 batch=256 for 10 epochs). CPU
    training on the 99-task Track A v2 dataset should finish in under
    30 minutes.
    """

    # --- feature assembly ---
    handcrafted_config: HandcraftedFeatureConfig = field(
        default_factory=HandcraftedFeatureConfig
    )

    # --- model sub-configs ---
    scorer_config: LiteRegionScorerConfig = field(
        default_factory=lambda: LiteRegionScorerConfig(feature_dim=15)
    )
    aggregator_config: LiteObjectAggregatorConfig = field(
        default_factory=lambda: LiteObjectAggregatorConfig(input_dim=15)
    )

    # --- training ---
    # ``same_set`` is the default to keep parity with ngram_logreg's
    # default; the CLI surfaces the flag so investigators get honest
    # holdout numbers on demand.
    train_mode: str = "same_set"
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Weight on the object-level BCE loss. The region-level BCE loss
    # is always at weight 1.0. 0.5 keeps object supervision informative
    # but lets region signal dominate (spec §11.3.4).
    object_loss_weight: float = 0.5
    # Class weight on the positive class in region-level BCE, to
    # counter the ~60:1 imbalance observed in synthetic region labels.
    # ``None`` = uniform; a float > 1 up-weights the rare positive
    # class. The default 1.0 matches ``class_weight="balanced"`` after
    # normalisation by batch size.
    positive_class_weight: float = 10.0
    # Set True to log per-epoch train loss to the root logger.
    verbose: bool = False
    # RNG seed for weight initialisation + batch shuffling.
    random_state: int = 0

    # --- inference ---
    threshold: float = 0.5
    # ObjectByteLoader cache size for Group-B byte features.
    # IMPORTANT: must be >= total unique (apk, object) pairs in the
    # training set or feature extraction thrashes (profile 2026-05-01
    # showed 87% of fold time was cache-miss ZIP reopens at
    # cache_size=64). 8192 comfortably holds the 84-task v2 training
    # union (~5000-7000 unique objects); each cached buffer averages
    # ~50 KB so the whole cache fits in ~400 MB RAM.
    loader_cache_size: int = 8192

    # --- device selection (GPU opt-in, 2026-05-01) ---
    # Torch device for training and inference. Accepts:
    #   - "auto" (default): CUDA if available, else CPU.
    #   - "cuda" / "cuda:0" / "cpu": explicit override.
    # Model.save() always serialises to CPU tensors so cpu-only
    # reloaders keep working; Model.load() then rehydrates onto the
    # configured device. Training/inference moves the modules and
    # tensors onto the resolved device. Keeping the feature-assembly
    # path pure-CPU since it is stdlib-only.
    device: str = "auto"


@dataclass(frozen=True)
class PayloadHunterLiteRegionPrediction:
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
class PayloadHunterLiteObjectPrediction:
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
class PayloadHunterLiteApkPrediction:
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
class PayloadHunterLiteResult:
    region_predictions: List[PayloadHunterLiteRegionPrediction]
    object_predictions: List[PayloadHunterLiteObjectPrediction]
    apk_predictions: List[PayloadHunterLiteApkPrediction]
    report: dict


# ---------------------------------------------------------------------------
# Trained model wrapper
# ---------------------------------------------------------------------------


class PayloadHunterLiteModel:
    """Bundle of (region scorer, object aggregator, feature schema).

    Both torch modules are built eagerly (the ``build_*`` factories
    require torch to be installed). Feature extraction itself remains
    pure stdlib; the torch dependency is only entered when
    :meth:`predict` or the training loop reaches the forward pass.
    """

    def __init__(
        self,
        region_scorer: Any,
        object_aggregator: Any,
        config: PayloadHunterLiteConfig,
        feature_names: Sequence[str],
    ) -> None:
        self._region_scorer = region_scorer
        self._object_aggregator = object_aggregator
        self._config = config
        self._feature_names = tuple(feature_names)

    @property
    def config(self) -> PayloadHunterLiteConfig:
        return self._config

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self._feature_names

    @property
    def threshold(self) -> float:
        return self._config.threshold

    # ----- I/O -----

    def save(self, path: Path) -> None:
        """Persist scorer + aggregator state_dicts + config.

        Always writes CPU-side state_dicts so a GPU-trained checkpoint
        reloads cleanly on a CPU-only machine (and vice versa).
        """

        torch, _ = _require_torch()
        payload = {
            "version": _MODEL_FORMAT_VERSION,
            "config": asdict(self._config),
            "feature_names": list(self._feature_names),
            "region_scorer_state_dict": {
                k: v.detach().cpu() for k, v in self._region_scorer.state_dict().items()
            },
            "object_aggregator_state_dict": {
                k: v.detach().cpu() for k, v in self._object_aggregator.state_dict().items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "PayloadHunterLiteModel":
        """Load a model saved via :meth:`save`.

        Rehydrates onto the configured device (``config.device``).
        Checkpoints are always saved as CPU tensors, so cpu-only
        reloaders keep working regardless of the machine that
        produced the checkpoint.
        """

        torch, _ = _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("version") != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {payload.get('version')!r}; "
                f"expected {_MODEL_FORMAT_VERSION}"
            )
        cfg = _config_from_dict(payload["config"])
        device = _resolve_device(cfg.device)
        scorer = build_lite_region_scorer(cfg.scorer_config)
        scorer.load_state_dict(payload["region_scorer_state_dict"])
        scorer.to(device)
        scorer.eval()
        aggregator = build_lite_object_aggregator(cfg.aggregator_config)
        aggregator.load_state_dict(payload["object_aggregator_state_dict"])
        aggregator.to(device)
        aggregator.eval()
        return cls(
            region_scorer=scorer,
            object_aggregator=aggregator,
            config=cfg,
            feature_names=payload["feature_names"],
        )

    # ----- Inference -----

    def predict(
        self,
        region_rows: Iterable[Mapping],
        apk_index: Optional[Mapping[str, Path]] = None,
        loader: Optional[ObjectByteLoader] = None,
    ) -> PayloadHunterLiteResult:
        """Score a stream of region rows and produce a full report.

        Parameters
        ----------
        region_rows:
            Iterable of region_training_label.jsonl-shaped rows. Every
            row must carry ``apk_id``, ``object_id``, ``region_id``,
            ``object_path``, ``offset_start``, ``offset_end``,
            ``entropy``, and ``label_id``.
        apk_index:
            ``apk_id -> apk path on disk``. Required for Group-B byte
            features (otherwise those dims are zero-filled, which is a
            no-op when ``handcrafted_config.include_byte_distribution``
            is False).
        loader:
            Optional pre-built :class:`ObjectByteLoader`.
        """

        torch, _ = _require_torch()
        rows = [dict(r) for r in region_rows]
        if not rows:
            return _empty_result()

        # Build feature vectors by mutating rows in-place.
        byte_loader_fn = _make_byte_loader(self._config, apk_index, loader)
        extract_handcrafted_features(
            rows,
            byte_loader=byte_loader_fn,
            config=self._config.handcrafted_config,
        )
        feature_matrix = _vectorise(rows, self._feature_names)

        # Forward pass on resolved device (cuda if auto + available).
        device = _resolve_device(self._config.device)
        self._region_scorer.to(device)
        self._object_aggregator.to(device)
        self._region_scorer.eval()
        self._object_aggregator.eval()
        with torch.no_grad():
            X = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
            region_logits = self._region_scorer(X).squeeze(-1)  # [N]
            region_scores_tensor = torch.sigmoid(region_logits)

            # Object-level scores via attention aggregator. Groups of
            # regions sharing (apk_id, object_id) are pooled
            # separately; aggregator input = handcrafted features +
            # scorer logit.
            object_scores: Dict[Tuple[str, str], float] = {}
            groups: Dict[Tuple[str, str], List[int]] = {}
            for idx, row in enumerate(rows):
                key = (str(row["apk_id"]), str(row["object_id"]))
                groups.setdefault(key, []).append(idx)
            for key, indices in groups.items():
                sub_feats = X[indices]
                sub_logits = region_logits[indices]
                obj_logit, _attn = self._object_aggregator(sub_feats, sub_logits)
                object_scores[key] = float(torch.sigmoid(obj_logit).item())

        scores_list = [float(round(s.item(), 6)) for s in region_scores_tensor]

        region_preds = _build_region_predictions(rows, scores_list, self.threshold)
        object_preds = _aggregate_objects(
            region_preds, object_scores, self.threshold
        )
        apk_preds = _aggregate_apks(region_preds, object_preds)
        report = _build_report(
            region_preds, object_preds, apk_preds, self.threshold
        )
        return PayloadHunterLiteResult(
            region_predictions=region_preds,
            object_predictions=object_preds,
            apk_predictions=apk_preds,
            report=report,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_payload_hunter_lite(
    region_rows: Iterable[Mapping],
    apk_index: Optional[Mapping[str, Path]] = None,
    config: Optional[PayloadHunterLiteConfig] = None,
) -> PayloadHunterLiteModel:
    """Fit a PayloadHunter-Lite model on region training labels.

    Only ``train_mode="same_set"`` is supported here. Holdout modes
    are handled by :func:`run_payload_hunter_lite_baseline`, which
    invokes this function once per held-out group and stitches the
    test predictions.
    """

    torch, nn = _require_torch()
    cfg = config or PayloadHunterLiteConfig()
    if cfg.train_mode != "same_set":
        raise ValueError(
            f"train_payload_hunter_lite only accepts train_mode='same_set'; "
            f"got {cfg.train_mode!r}. Use run_payload_hunter_lite_baseline "
            f"for holdout modes."
        )

    rows = [dict(r) for r in region_rows]
    if not rows:
        raise ValueError("train_payload_hunter_lite received no training rows")

    # Feature assembly.
    byte_loader_fn = _make_byte_loader(cfg, apk_index, None)
    extract_handcrafted_features(
        rows, byte_loader=byte_loader_fn, config=cfg.handcrafted_config
    )
    feat_names = handcrafted_feature_names(cfg.handcrafted_config)
    # Re-align configs so model dims match feature dim emitted by the
    # handcrafted assembly. This keeps users from having to keep the
    # numbers manually in-sync.
    dim = len(feat_names)
    cfg = _rebind_model_dim(cfg, dim)

    feature_matrix = _vectorise(rows, feat_names)
    labels = [int(r["label_id"]) for r in rows]
    # Object groups: label = OR of region labels.
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in enumerate(rows):
        key = (str(row["apk_id"]), str(row["object_id"]))
        groups.setdefault(key, []).append(idx)
    obj_labels = {k: int(any(labels[i] for i in v)) for k, v in groups.items()}

    torch.manual_seed(cfg.random_state)
    device = _resolve_device(cfg.device)
    scorer = build_lite_region_scorer(cfg.scorer_config)
    aggregator = build_lite_object_aggregator(cfg.aggregator_config)
    scorer.to(device)
    aggregator.to(device)
    scorer.train()
    aggregator.train()
    optimizer = torch.optim.AdamW(
        list(scorer.parameters()) + list(aggregator.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Class weight for region-level BCE. Positive class is rare; up-
    # weighting helps recall on the hidden-payload class.
    pos_weight = torch.tensor(
        [cfg.positive_class_weight], dtype=torch.float32, device=device
    )
    region_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    object_bce = nn.BCEWithLogitsLoss()  # object groups are balanced-ish.

    X = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    n = X.shape[0]
    batch_size = max(1, min(cfg.batch_size, n))
    indices = torch.arange(n)
    for epoch in range(cfg.epochs):
        # Shuffle.
        perm = torch.randperm(n, generator=_mk_generator(cfg, epoch))
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            region_logits = scorer(Xb).squeeze(-1)
            loss_region = region_bce(region_logits, yb)

            # Object-level supervision: sample one group per training
            # step (constrained by batch_idx to keep gradient flow
            # meaningful). We pool only groups fully contained in the
            # batch to avoid leaking gradient across batch boundaries.
            obj_losses: List[Any] = []
            batch_set = set(int(i) for i in batch_idx.tolist())
            for key, member_indices in groups.items():
                if not all(i in batch_set for i in member_indices):
                    continue
                sub = torch.tensor(member_indices, dtype=torch.long, device=device)
                sub_batch_positions = [
                    (batch_idx == i).nonzero(as_tuple=True)[0].item()
                    for i in member_indices
                ]
                sub_batch_positions_t = torch.tensor(
                    sub_batch_positions, dtype=torch.long, device=device
                )
                sub_feats = Xb[sub_batch_positions_t]
                sub_logits = region_logits[sub_batch_positions_t]
                obj_logit, _attn = aggregator(sub_feats, sub_logits)
                obj_label = torch.tensor(
                    float(obj_labels[key]), dtype=torch.float32, device=device
                )
                obj_losses.append(
                    object_bce(obj_logit.unsqueeze(0), obj_label.unsqueeze(0))
                )

            if obj_losses:
                loss_obj = torch.stack(obj_losses).mean()
                loss = loss_region + cfg.object_loss_weight * loss_obj
            else:
                loss = loss_region

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        if cfg.verbose and num_batches:
            import logging

            logging.getLogger(__name__).info(
                "payload_hunter_lite epoch=%d loss=%.6f",
                epoch,
                epoch_loss / num_batches,
            )

    scorer.eval()
    aggregator.eval()
    return PayloadHunterLiteModel(
        region_scorer=scorer,
        object_aggregator=aggregator,
        config=cfg,
        feature_names=feat_names,
    )


# ---------------------------------------------------------------------------
# Orchestrator: same_set + holdout_transform + holdout_package
# ---------------------------------------------------------------------------


def run_payload_hunter_lite_baseline(
    region_rows: Iterable[Mapping],
    apk_index: Optional[Mapping[str, Path]] = None,
    config: Optional[PayloadHunterLiteConfig] = None,
) -> PayloadHunterLiteResult:
    """End-to-end entry: train + predict under the chosen ``train_mode``.

    Mirrors :mod:`baselines.ngram_logreg`'s orchestrator signature so
    the multi-baseline runner treats PayloadHunter-Lite as a drop-in
    baseline.
    """

    cfg = config or PayloadHunterLiteConfig()
    if cfg.train_mode not in _SUPPORTED_TRAIN_MODES:
        raise ValueError(
            f"unsupported train_mode {cfg.train_mode!r}; "
            f"expected one of {_SUPPORTED_TRAIN_MODES}"
        )
    rows = [dict(r) for r in region_rows]
    if not rows:
        return _empty_result()

    if cfg.train_mode == "same_set":
        warnings.warn(
            "PayloadHunterLite train_mode='same_set' produces in-sample "
            "numbers; pass 'holdout_transform' or 'holdout_package' for "
            "honest out-of-distribution scores.",
            stacklevel=2,
        )
        inner_cfg = _with_train_mode(cfg, "same_set")
        model = train_payload_hunter_lite(rows, apk_index, inner_cfg)
        return model.predict(rows, apk_index)

    # Holdout modes: leave one group out, train on the rest, predict
    # the held-out group; then concatenate.
    group_key = (
        "transform_family" if cfg.train_mode == "holdout_transform" else "package_name"
    )
    for row in rows:
        if group_key not in row:
            if group_key == "package_name":
                # Derive from apk_id prefix (e.g. "org_fdroid_..." ->
                # "org.fdroid...") when not pre-populated.
                row["package_name"] = _derive_package_name(str(row["apk_id"]))
            else:
                raise KeyError(
                    f"holdout_transform requires every row to carry "
                    f"{group_key!r}; missing in apk_id={row.get('apk_id')!r}"
                )

    groups: Dict[str, List[Mapping]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    if len(groups) < 2:
        raise ValueError(
            f"train_mode={cfg.train_mode!r} requires >= 2 distinct "
            f"{group_key!r} values; got {len(groups)}"
        )

    all_region_preds: List[PayloadHunterLiteRegionPrediction] = []
    all_object_preds: List[PayloadHunterLiteObjectPrediction] = []
    all_apk_preds: List[PayloadHunterLiteApkPrediction] = []

    inner_cfg = _with_train_mode(cfg, "same_set")
    for held in sorted(groups):
        train_rows = [r for g, items in groups.items() if g != held for r in items]
        test_rows = groups[held]
        model = train_payload_hunter_lite(train_rows, apk_index, inner_cfg)
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
    return PayloadHunterLiteResult(
        region_predictions=all_region_preds,
        object_predictions=all_object_preds,
        apk_predictions=all_apk_preds,
        report=stitched_report,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PayloadHunter-Lite baseline requires torch. "
            "Install via ``pip install -e \".[dl]\"``."
        ) from exc
    return torch, nn


def _resolve_device(device_str: str) -> Any:
    """Map the config ``device`` string to a ``torch.device``.

    Accepts ``"auto"`` (CUDA if available else CPU), ``"cpu"``,
    ``"cuda"``, or an explicit ``"cuda:N"``. Unknown strings fall
    back to CPU with a single warning so orchestrators keep running
    on clusters that silently reshuffle device IDs.
    """

    torch, _ = _require_torch()
    s = (device_str or "auto").strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda" or s.startswith("cuda:"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"PayloadHunter-Lite device={device_str!r} requested but CUDA "
                f"is unavailable; falling back to CPU.",
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device(s)
    warnings.warn(
        f"PayloadHunter-Lite: unknown device={device_str!r}; falling back to CPU.",
        stacklevel=2,
    )
    return torch.device("cpu")


def _mk_generator(cfg: PayloadHunterLiteConfig, epoch: int) -> Any:
    torch, _ = _require_torch()
    gen = torch.Generator()
    gen.manual_seed(cfg.random_state * 1000 + epoch)
    return gen


def _make_byte_loader(
    cfg: PayloadHunterLiteConfig,
    apk_index: Optional[Mapping[str, Path]],
    loader: Optional[ObjectByteLoader],
) -> Optional[Callable[[Mapping], bytes]]:
    """Return a callable ``row -> bytes`` or ``None`` to zero-fill.

    If Group-B byte features are disabled in the config we return
    ``None`` and the handcrafted assembly fills zeros — fine, keeps
    the feature-vector shape.
    """

    if not cfg.handcrafted_config.include_byte_distribution:
        return None
    if apk_index is None:
        return None
    resolved_loader = loader or ObjectByteLoader(cache_size=cfg.loader_cache_size)

    def _load(row: Mapping) -> bytes:
        apk_id = str(row["apk_id"])
        if apk_id not in apk_index:
            return b""
        return resolved_loader.region_bytes(
            Path(apk_index[apk_id]),
            str(row["object_path"]),
            int(row["offset_start"]),
            int(row["offset_end"]),
        )

    return _load


def _vectorise(
    rows: Sequence[MutableMapping[str, Any]],
    feat_names: Sequence[str],
) -> List[List[float]]:
    """Turn in-place-populated feature keys into a [N, D] matrix."""

    matrix: List[List[float]] = []
    for row in rows:
        matrix.append([float(row.get(name, 0.0)) for name in feat_names])
    return matrix


def _rebind_model_dim(
    cfg: PayloadHunterLiteConfig, dim: int
) -> PayloadHunterLiteConfig:
    """Re-emit the config with scorer/aggregator dims aligned to ``dim``."""

    if cfg.scorer_config.feature_dim == dim and cfg.aggregator_config.input_dim == dim:
        return cfg
    new_scorer = LiteRegionScorerConfig(
        feature_dim=dim,
        hidden_dim=cfg.scorer_config.hidden_dim,
        num_hidden_layers=cfg.scorer_config.num_hidden_layers,
        dropout=cfg.scorer_config.dropout,
        activation=cfg.scorer_config.activation,
    )
    new_agg = LiteObjectAggregatorConfig(
        input_dim=dim,
        attn_hidden_dim=cfg.aggregator_config.attn_hidden_dim,
        dropout=cfg.aggregator_config.dropout,
        return_attention=cfg.aggregator_config.return_attention,
    )
    return PayloadHunterLiteConfig(
        handcrafted_config=cfg.handcrafted_config,
        scorer_config=new_scorer,
        aggregator_config=new_agg,
        train_mode=cfg.train_mode,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        object_loss_weight=cfg.object_loss_weight,
        positive_class_weight=cfg.positive_class_weight,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
        threshold=cfg.threshold,
        loader_cache_size=cfg.loader_cache_size,
    )


def _with_train_mode(
    cfg: PayloadHunterLiteConfig, mode: str
) -> PayloadHunterLiteConfig:
    return PayloadHunterLiteConfig(
        handcrafted_config=cfg.handcrafted_config,
        scorer_config=cfg.scorer_config,
        aggregator_config=cfg.aggregator_config,
        train_mode=mode,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        object_loss_weight=cfg.object_loss_weight,
        positive_class_weight=cfg.positive_class_weight,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
        threshold=cfg.threshold,
        loader_cache_size=cfg.loader_cache_size,
    )


def _config_from_dict(d: Mapping[str, Any]) -> PayloadHunterLiteConfig:
    # The top-level HandcraftedFeatureConfig contains a nested
    # EntropyDeltaConfig dataclass; ``asdict`` on the outer object
    # recurses it into a plain dict, so we have to rebuild both on
    # the load path to restore exact-type equivalence.
    from android_packer.features.entropy_delta import EntropyDeltaConfig

    handcrafted_raw = dict(d["handcrafted_config"])
    entropy_delta_raw = handcrafted_raw.get("entropy_delta_config")
    if isinstance(entropy_delta_raw, Mapping):
        handcrafted_raw["entropy_delta_config"] = EntropyDeltaConfig(
            **entropy_delta_raw
        )
    handcrafted = HandcraftedFeatureConfig(**handcrafted_raw)

    return PayloadHunterLiteConfig(
        handcrafted_config=handcrafted,
        scorer_config=LiteRegionScorerConfig(**d["scorer_config"]),
        aggregator_config=LiteObjectAggregatorConfig(**d["aggregator_config"]),
        train_mode=d.get("train_mode", "same_set"),
        epochs=d.get("epochs", 10),
        batch_size=d.get("batch_size", 256),
        learning_rate=d.get("learning_rate", 1e-3),
        weight_decay=d.get("weight_decay", 1e-4),
        object_loss_weight=d.get("object_loss_weight", 0.5),
        positive_class_weight=d.get("positive_class_weight", 10.0),
        verbose=d.get("verbose", False),
        random_state=d.get("random_state", 0),
        threshold=d.get("threshold", 0.5),
        loader_cache_size=d.get("loader_cache_size", 64),
    )


def _derive_package_name(apk_id: str) -> str:
    """Best-effort derive ``package_name`` from an apk_id.

    Synthetic task names look like
    ``org_fdroid_fdroid_1023052_985f5181_xor``; the package part
    ``org_fdroid_fdroid`` is the first ``_``-delimited prefix up to
    the first all-digit segment. Returns the apk_id as-is when no
    such split succeeds (holdout will then degrade gracefully to
    leave-one-apk-out for that row).
    """

    parts = apk_id.split("_")
    # Strip trailing transform family / version / hash segments, keep
    # the leading alpha-only reversed-domain-style tokens.
    head: List[str] = []
    for p in parts:
        if p.isdigit() or all(c in "0123456789abcdef" for c in p) and len(p) >= 6:
            break
        head.append(p)
    return ".".join(head) if head else apk_id


# ----- Prediction assembly (mirror ngram_logreg helpers) ---------------------


def _build_region_predictions(
    rows: Sequence[Mapping],
    scores: Sequence[float],
    threshold: float,
) -> List[PayloadHunterLiteRegionPrediction]:
    preds: List[PayloadHunterLiteRegionPrediction] = []
    for row, score in zip(rows, scores):
        preds.append(
            PayloadHunterLiteRegionPrediction(
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
    region_preds: Sequence[PayloadHunterLiteRegionPrediction],
    object_scores: Mapping[Tuple[str, str], float],
    threshold: float,
) -> List[PayloadHunterLiteObjectPrediction]:
    groups: Dict[Tuple[str, str], List[PayloadHunterLiteRegionPrediction]] = {}
    for r in region_preds:
        groups.setdefault((r.apk_id, r.object_id), []).append(r)

    out: List[PayloadHunterLiteObjectPrediction] = []
    for (apk_id, object_id), regions in sorted(groups.items()):
        # Use aggregator-pooled score if available, else fall back to
        # max over region scores (parity with ngram_logreg when the
        # aggregator is disabled in an ablation).
        score = object_scores.get(
            (apk_id, object_id), max(r.score for r in regions)
        )
        predicted = 1 if score >= threshold else 0
        true_label = 1 if any(r.true_label_id for r in regions) else 0
        out.append(
            PayloadHunterLiteObjectPrediction(
                apk_id=apk_id,
                object_id=object_id,
                object_path=regions[0].object_path,
                score=round(score, 6),
                predicted_label_id=predicted,
                true_label_id=true_label,
                region_count=len(regions),
                positive_region_count=sum(r.true_label_id for r in regions),
            )
        )
    return out


def _aggregate_apks(
    region_preds: Sequence[PayloadHunterLiteRegionPrediction],
    object_preds: Sequence[PayloadHunterLiteObjectPrediction],
) -> List[PayloadHunterLiteApkPrediction]:
    regions_by_apk: Dict[str, List[PayloadHunterLiteRegionPrediction]] = {}
    objects_by_apk: Dict[str, List[PayloadHunterLiteObjectPrediction]] = {}
    for r in region_preds:
        regions_by_apk.setdefault(r.apk_id, []).append(r)
    for o in object_preds:
        objects_by_apk.setdefault(o.apk_id, []).append(o)

    out: List[PayloadHunterLiteApkPrediction] = []
    for apk_id in sorted(set(regions_by_apk) | set(objects_by_apk)):
        regions = regions_by_apk.get(apk_id, [])
        objects = objects_by_apk.get(apk_id, [])
        score = max((o.score for o in objects), default=0.0)
        predicted = max((o.predicted_label_id for o in objects), default=0)
        true_label = 1 if any(o.true_label_id for o in objects) else 0
        out.append(
            PayloadHunterLiteApkPrediction(
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
    regions: Sequence[PayloadHunterLiteRegionPrediction],
    objects: Sequence[PayloadHunterLiteObjectPrediction],
) -> List[dict]:
    by_object: Dict[Tuple[str, str], List[PayloadHunterLiteRegionPrediction]] = {}
    for r in regions:
        by_object.setdefault((r.apk_id, r.object_id), []).append(r)
    samples: List[dict] = []
    for obj in objects:
        if not obj.true_label_id:
            continue
        rows = by_object.get((obj.apk_id, obj.object_id), [])
        truth = [
            (r.offset_start, r.offset_end) for r in rows if r.true_label_id == 1
        ]
        prediction: List[Tuple[int, int]] = []
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
    region_preds: Sequence[PayloadHunterLiteRegionPrediction],
    object_preds: Sequence[PayloadHunterLiteObjectPrediction],
    apk_preds: Sequence[PayloadHunterLiteApkPrediction],
    threshold: float,
    *,
    extra: Optional[Mapping[str, Any]] = None,
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
    ranking = ranking_metrics(
        groups=[o.apk_id for o in object_preds],
        truth=[o.true_label_id for o in object_preds],
        scores=[o.score for o in object_preds],
    )
    loc_samples = _object_localization_samples(region_preds, object_preds)
    localization = localization_metrics(loc_samples)
    counts = {
        "regions": len(region_preds),
        "objects": len(object_preds),
        "apks": len(apk_preds),
    }
    report = {
        "baseline": "payload_hunter_lite",
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
    if extra:
        report.update(dict(extra))
    return report


def _empty_result() -> PayloadHunterLiteResult:
    return PayloadHunterLiteResult(
        region_predictions=[],
        object_predictions=[],
        apk_predictions=[],
        report={
            "baseline": "payload_hunter_lite",
            "counts": {"regions": 0, "objects": 0, "apks": 0},
            "metrics": {},
        },
    )
