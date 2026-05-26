"""Ours (Typed-Instance MIL) baseline wrapper — F-MIL-e.

Public entry points mirror :mod:`android_packer.baselines.payload_hunter_lite`
and :mod:`android_packer.baselines.ngram_logreg` so that the evaluation
pipeline and :mod:`android_packer.cli.run_synthetic_multi_baseline` can
consume the Ours method without special-casing:

* :class:`OursApkPrediction`, :class:`OursObjectPrediction`,
  :class:`OursRegionPrediction`, :class:`OursResult` — dataclasses
  **field-compatible** with their NgramLogReg / PayloadHunterLite
  counterparts (§8 hard contract).
* :class:`OursBaselineConfig` — wraps :class:`MILTrainerConfig` plus
  I/O knobs.
* :func:`train_ours_baseline` — fit an :class:`OursBaselineModel` on
  region rows + per-apk typed-instance features.
* :func:`run_ours_baseline` — load a trained model and predict on a
  region-row iterable (the multi-baseline runner entry).

Design notes
------------
The Ours method's fundamental scoring unit is the **typed instance**
(an APK object: one DEX, one asset, one ELF, …).  To keep parity with
the region-first baselines we project the instance score back onto
every region of the underlying object — this is by construction
(``object_score == any instance_score_for_that_object``) because we
emit exactly one instance per object for the MVP.  The projection is
documented in ``docs/method/ours_method_spec.md`` §12.2.

The feature extraction for typed instances currently reuses the
15-dim handcrafted assembly from PayloadHunter-Lite; a richer
byte-encoder feature path is planned for F-MIL-e stage-2 and will plug
in here behind the same baseline contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from android_packer.features.byte_features import ObjectByteLoader
from android_packer.features.handcrafted import (
    HandcraftedFeatureConfig,
    extract_handcrafted_features,
    handcrafted_feature_names,
)
from android_packer.models.ours import OursConfig
from android_packer.models.typed_encoder import (
    N_TYPED_INSTANCE_TYPES,
    TYPED_INSTANCE_TYPES,
    TypedEncoderConfig,
    instance_type_id,
)


__all__ = [
    "FAMILY_TO_PAYLOAD_KIND",
    "OursApkPrediction",
    "OursBaselineConfig",
    "OursBaselineModel",
    "OursObjectPrediction",
    "OursRegionPrediction",
    "OursResult",
    "run_ours_baseline",
    "train_ours_baseline",
    "train_ours_baseline_from_objects",
]


_MODEL_FORMAT_VERSION = 1


#: Ground-truth mapping from synthetic ``transform_family`` to the
#: typed-instance kind label.  This replaces the path-based heuristic
#: that ``_object_instance_type`` used before the L42 fix: the injection
#: path is drawn from the seed's native directory pool (post-L4 fix), so
#: the path carries **no** signal about what the transform actually did
#: to the bytes.  The transform family *is* the signal; record it.
#:
#: ``path_randomized`` / ``signature_strip`` / ``embedded_archive`` are
#: metadata-only transforms (they don't introduce executable payload
#: bytes per se), but the per-instance typed head still benefits from
#: routing them through a distinct channel so the model can learn
#: "metadata was touched here" independently from "crypt payload lives
#: here".  We route them through ``shim`` because that is the closest
#: existing type id.  If future work adds a ``metadata_only`` kind in
#: :mod:`labeling.injected_packer_adapter`, update this table in lockstep.
FAMILY_TO_PAYLOAD_KIND: Dict[str, str] = {
    "xor": "encrypted_dex",
    "base64": "encrypted_dex",
    "split_xor": "encrypted_dex",
    "dex_string_encrypted": "encrypted_dex",
    "dex_method_inlined": "extracted_method_body",
    "embedded_asset": "compressed_payload",
    "embedded_archive": "compressed_payload",
    "so_embedded": "native_stub",
    "multi_dex_shim": "shim",
    "path_randomized": "shim",
    "signature_strip": "shim",
}


# ---------------------------------------------------------------------------
# Dataclasses (field-compatible with NgramLogReg / PayloadHunterLite)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OursRegionPrediction:
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
class OursObjectPrediction:
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
class OursApkPrediction:
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
class OursResult:
    region_predictions: List[OursRegionPrediction]
    object_predictions: List[OursObjectPrediction]
    apk_predictions: List[OursApkPrediction]
    report: dict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OursBaselineConfig:
    """I/O + training knobs for :func:`train_ours_baseline`.

    ``ours_config`` controls the typed-encoder + MIL architecture; the
    rest mirrors :class:`PayloadHunterLiteConfig` (threshold, cache
    size, etc.) so configs can be copy-pasted with minimal surgery.
    """

    ours_config: OursConfig = field(default_factory=OursConfig)
    handcrafted_config: HandcraftedFeatureConfig = field(
        default_factory=HandcraftedFeatureConfig
    )
    # Training hyper-params (forwarded into MILTrainerConfig).
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_diff_pseudo: float = 0.3
    lambda_sparsity: float = 0.01
    bag_pos_weight: float = 1.0
    random_state: int = 0
    verbose: bool = False
    # L41 (2026-05-07): bag subsampling knobs, forwarded into MILTrainerConfig.
    train_max_bag_size: Optional[int] = 256
    train_min_positive_fraction: float = 0.01
    # L43 (2026-05-07): supervision mode -- see MILTrainerConfig.
    supervision_mode: str = "instance_aided"
    # Inference.
    threshold: float = 0.5
    loader_cache_size: int = 4096
    # L44 (2026-05-10): scoring mode for instance/region score propagation.
    # "instance_logit"       — legacy: sigmoid(instance_logit) per object (default)
    # "attention"            — attention weight (normalized to [0,1] by max) as score
    # "attention_x_bag"      — attention[i]/max(attention) * sigmoid(bag_logit)
    # "attention_anomaly"    — (1 - attention[i]/max(attention)) * sigmoid(bag_logit)
    #                          Low-attention instances score HIGH (anomaly = payload)
    # "attention_auto"       — auto-detect direction: on first APK in each predict
    #                          call, check whether positive instances correlate with
    #                          high or low attention, then pick x_bag or anomaly.
    scoring_mode: str = "instance_logit"
    # L48 (2026-05-11): device for MIL neural-network training + inference.
    # "auto"   — use CUDA if available, else CPU (recommended default)
    # "cpu"    — force CPU
    # "cuda"   — force CUDA (raises if unavailable)
    # "cuda:N" — specific GPU index
    device: str = "auto"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class OursBaselineModel:
    """Trained Ours model + the feature config used at training time.

    Keeps both pieces bundled so :meth:`predict` can re-materialise
    instance features deterministically on new region rows.
    """

    def __init__(
        self,
        module: Any,
        baseline_config: OursBaselineConfig,
    ) -> None:
        self._module = module
        self._config = baseline_config

    @property
    def module(self) -> Any:
        return self._module

    @property
    def config(self) -> OursBaselineConfig:
        return self._config

    # ----- I/O ----- (torch pickle is fine for the MVP; revisit for portability)

    def save(self, path: Path) -> None:
        import torch

        payload = {
            "version": _MODEL_FORMAT_VERSION,
            "state_dict": self._module.state_dict(),
            "baseline_config": asdict(self._config),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "OursBaselineModel":
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("version") != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Ours baseline version {payload.get('version')!r}; "
                f"expected {_MODEL_FORMAT_VERSION}"
            )
        cfg_dict = payload["baseline_config"]
        # Rehydrate sub-configs.  Keep the reconstruction minimal: the
        # multi-baseline runner always produces the right JSON shape.
        baseline_config = OursBaselineConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(**cfg_dict["ours_config"]["typed"]),
                mil_pooling=cfg_dict["ours_config"].get("mil_pooling", "attention"),
            ),
            handcrafted_config=HandcraftedFeatureConfig(
                **cfg_dict["handcrafted_config"]
            ),
            epochs=cfg_dict["epochs"],
            batch_size=cfg_dict["batch_size"],
            learning_rate=cfg_dict["learning_rate"],
            weight_decay=cfg_dict["weight_decay"],
            lambda_diff_pseudo=cfg_dict["lambda_diff_pseudo"],
            lambda_sparsity=cfg_dict["lambda_sparsity"],
            bag_pos_weight=cfg_dict["bag_pos_weight"],
            random_state=cfg_dict["random_state"],
            train_max_bag_size=cfg_dict.get("train_max_bag_size", 256),
            train_min_positive_fraction=cfg_dict.get(
                "train_min_positive_fraction", 0.01
            ),
            supervision_mode=cfg_dict.get("supervision_mode", "instance_aided"),
            threshold=cfg_dict["threshold"],
            loader_cache_size=cfg_dict["loader_cache_size"],
        )
        from android_packer.models.ours import build_ours

        module = build_ours(baseline_config.ours_config)
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return cls(module, baseline_config)

    # ----- Inference -----

    def predict(
        self,
        region_rows: Iterable[Mapping],
        apk_index: Mapping[str, Path],
        loader: Optional[ObjectByteLoader] = None,
    ) -> OursResult:
        return _predict_impl(
            module=self._module,
            baseline_config=self._config,
            region_rows=region_rows,
            apk_index=apk_index,
            loader=loader,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


_LABEL_NOT_PROVIDED = object()  # sentinel: inference caller passed no label_id


def _object_instance_type(
    object_id: str,
    object_path: str,
    *,
    transform_families: Optional[Sequence[str]] = None,
    label_id: object = _LABEL_NOT_PROVIDED,
) -> int:
    """Resolve a typed-instance id for an APK object.

    L42 fix (2026-05-07): the pre-fix heuristic inferred the type from
    the object path (``*.dex`` -> encrypted_dex, ``*.so`` -> native_stub,
    etc.).  That was unsound on the synthetic corpus, because post-L4
    the injection path is drawn from the seed's native directory pool
    and carries **no** signal about what the transform actually did to
    the bytes.  An ``xor`` payload landing at
    ``res/drawable/foo.png`` got typed as ``shim`` (the default
    fallback), and a benign ``assets/data/config.json`` got typed as
    ``compressed_payload`` regardless of its contents.

    Resolution order (first match wins):
    1. ``transform_families`` is non-empty AND ``label_id > 0`` →
       :data:`FAMILY_TO_PAYLOAD_KIND` lookup.  This is the ground-truth
       path and the one used at training time on the synthetic corpus.
    2. ``label_id == 0`` (benign object, no injection) →
       ``benign_other``.  Native ``res/``, ``kotlin/``, ``META-INF/``
       instances land here regardless of path.
    3. Legacy path-based heuristic (unchanged), used only when neither
       label_id nor transform_families is available (e.g. real Track B
       packers before the labeller has run).  This keeps the method
       applicable to inference-time cases where ground-truth is not
       present.
    """

    # L47 (2026-05-11): Type routing design:
    #   TRAINING: uses label_id + transform_families for ground-truth type
    #     assignment. This is LEGITIMATE supervised signal.
    #   INFERENCE: uses simplified path-based heuristic WITHOUT benign_other.
    #     All objects get assigned to one of 6 functional types based on path.
    #     The model must rely on features, not type identity, to discriminate.
    #
    # Sentinel: inference calls pass no kwargs → label_id defaults to the
    # _LABEL_NOT_PROVIDED sentinel object.  Training calls always pass an
    # explicit label_id integer (even 0 for benign rows).

    _is_inference = (label_id is _LABEL_NOT_PROVIDED)

    if not _is_inference:
        # --- TRAINING MODE: GT-based type assignment ---
        if label_id > 0 and transform_families:
            for family in transform_families:
                kind = FAMILY_TO_PAYLOAD_KIND.get(family)
                if kind is not None:
                    return instance_type_id(kind)
        if label_id == 0:
            return instance_type_id("benign_other")

    # --- INFERENCE MODE: simplified path heuristic (no benign_other) ---
    # All objects get routed to functional type bins. The model cannot use
    # type identity alone to discriminate; it must use learned features.
    lower_path = object_path.lower() if object_path else ""
    lower_id = object_id.lower()
    check = lower_path or lower_id

    # DEX files
    if check.endswith(".dex") or "classes" in check:
        return instance_type_id("encrypted_dex")
    # Native shared libraries
    if check.endswith(".so") or "/lib/" in check:
        return instance_type_id("native_stub")
    # Assets
    if "assets/" in check or check.startswith("assets/"):
        return instance_type_id("compressed_payload")
    # Metadata files (signatures, manifests, certificates, resources.arsc)
    if "meta-inf/" in check or "manifest" in check or check.endswith(".arsc"):
        return instance_type_id("metadata_table")
    # Default fallback: shim (includes res/, kotlin/, org/, etc.)
    # NOTE: we do NOT use benign_other at inference time. The model should
    # distinguish benign vs packed by features, not by type identity.
    return instance_type_id("shim")


def _aggregate_object_features(
    region_rows: Sequence[Mapping],
    apk_index: Mapping[str, Path],
    *,
    handcrafted_config: HandcraftedFeatureConfig,
    loader: ObjectByteLoader,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[str]]:
    """Group region rows into (apk_id, object_id) buckets with a single
    pooled feature vector per object.

    Uses the PayloadHunter-Lite feature assembly verbatim (same Pass-2a
    15-dim set) so downstream ablations can swap Ours ↔ Lite on the
    same feature substrate.  Pool strategy: **mean** across regions.

    Returns ``(objects_dict, feature_names)``.  The feature names list is
    the one we fed into the handcrafted assembly; the trainer uses its
    length as ``input_dim``.
    """

    import numpy as np

    # Materialise rows to MutableMapping so in-place feature assembly
    # works, and so we can index later.  We shallow-copy each row dict.
    rows: List[Dict[str, Any]] = [dict(row) for row in region_rows]
    if not rows:
        return {}, handcrafted_feature_names(handcrafted_config)

    byte_loader_fn = _make_byte_loader(handcrafted_config, apk_index, loader)
    extract_handcrafted_features(
        rows, byte_loader=byte_loader_fn, config=handcrafted_config
    )
    feat_names = handcrafted_feature_names(handcrafted_config)

    # --- Group H: DEX section structure features (Tier 1A improvement) -------
    # When enabled, load full object bytes for each region row and compute
    # the 12-dim DEX structural feature vector, then fill it into the row
    # dict so that the generic ``feat_names`` extraction below picks it up.
    # We do a lazy import here because dex_structure_features is optional
    # (zero-dependency, but behind the include_dex_structure flag) and to
    # keep the core pipeline free of unconditional imports.
    #
    # Performance: parse_dex_item_spans() is O(file size) and must NOT be
    # called once per region row (a single DEX can produce 500+ windows).
    # We use extract_dex_structure_features_with_cache() which caches the
    # parsed span list keyed by (apk_id, object_path), reducing DEX parsing
    # from O(rows) → O(unique objects).
    if handcrafted_config.include_dex_structure and apk_index:
        from android_packer.features.dex_structure_features import (  # noqa: PLC0415
            DEX_STRUCTURE_FEATURE_NAMES as _DEX_NAMES,
            extract_dex_structure_features_with_cache as _extract_dex_cached,
        )

        _spans_cache: Dict[Tuple[str, str], Any] = {}
        for row in rows:
            apk_id = str(row["apk_id"])
            if apk_id not in apk_index:
                for _n in _DEX_NAMES:
                    row[_n] = 0.0
                continue
            _obj_path = str(row["object_path"])
            _cache_key = (apk_id, _obj_path)
            try:
                _obj_bytes = loader._read_object(
                    str(apk_index[apk_id]),
                    _obj_path,
                )
            except Exception:  # noqa: BLE001 — graceful fallback for bad ZIP entries
                _obj_bytes = b""
            _off_start = int(row.get("offset_start", 0))
            _off_end = int(row.get("offset_end", _off_start))
            _region_size = max(0, _off_end - _off_start)
            _dex_feats = _extract_dex_cached(
                _obj_bytes, _off_start, _region_size, _spans_cache, _cache_key
            )
            for _n, _v in zip(_DEX_NAMES, _dex_feats):
                row[_n] = _v

    objects: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        apk_id = str(row["apk_id"])
        object_id = str(row["object_id"])
        key = (apk_id, object_id)
        feat_vec = np.asarray(
            [float(row.get(name, 0.0)) for name in feat_names],
            dtype=np.float32,
        )
        region_label = int(row.get("label_id", 0))
        # Prefer explicit true_label_id when present; else fall back to
        # label_id so synthetic rows without the former still work.
        row_true = int(row.get("true_label_id", region_label))

        if key not in objects:
            objects[key] = {
                "apk_id": apk_id,
                "object_id": object_id,
                "object_path": str(row.get("object_path", object_id)),
                "true_label_id": row_true,
                "feature_sum": feat_vec.copy(),
                "region_count": 1,
                "positive_region_count": region_label,
                "region_rows": [row],
                # L42: carry transform_families for ground-truth typed
                # routing; union across regions of the same object since
                # split_xor emits one row per part but all share family.
                "transform_families": list(row.get("transform_families", []) or []),
            }
        else:
            obj = objects[key]
            obj["feature_sum"] = obj["feature_sum"] + feat_vec
            obj["region_count"] += 1
            obj["positive_region_count"] += region_label
            obj["region_rows"].append(row)
            obj["true_label_id"] = max(obj["true_label_id"], row_true)
            for fam in (row.get("transform_families", []) or []):
                if fam not in obj["transform_families"]:
                    obj["transform_families"].append(fam)

    for obj in objects.values():
        obj["feature_vec"] = obj["feature_sum"] / obj["region_count"]

    return objects, feat_names


def _make_byte_loader(
    handcrafted_config: HandcraftedFeatureConfig,
    apk_index: Mapping[str, Path],
    loader: ObjectByteLoader,
):
    """Return ``row -> bytes`` or ``None`` (mirrors PayloadHunter-Lite)."""

    if not handcrafted_config.include_byte_distribution:
        return None
    if not apk_index:
        return None

    def _load(row: Mapping) -> bytes:
        apk_id = str(row["apk_id"])
        if apk_id not in apk_index:
            return b""
        return loader.region_bytes(
            Path(apk_index[apk_id]),
            str(row["object_path"]),
            int(row["offset_start"]),
            int(row["offset_end"]),
        )

    return _load


def _resolve_torch_device(device_str: str) -> "torch.device":
    """Resolve "auto" / "cpu" / "cuda" / "cuda:N" to a torch.device.

    L48 (2026-05-11): centralised device resolver so both training and
    inference use the same logic.
    """
    import torch

    s = (device_str or "auto").strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda" or s.startswith("cuda:"):
        if not torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"ours: device={device_str!r} requested but CUDA is unavailable;"
                " falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return torch.device(s)
    import warnings
    warnings.warn(
        f"ours: unknown device={device_str!r}; falling back to CPU.",
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.device("cpu")


def train_ours_baseline_from_objects(
    objects: Mapping[Tuple[str, str], Mapping[str, Any]],
    config: Optional[OursBaselineConfig] = None,
) -> OursBaselineModel:
    """Train an Ours model from pre-aggregated object features."""

    import numpy as np

    from android_packer.training.mil_trainer import (
        MILBag,
        MILTrainerConfig,
        train_ours,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "train_ours_baseline requires torch ([dl] extra)."
        ) from exc

    cfg = config or OursBaselineConfig()

    # L48 (2026-05-11): resolve training device once; bags will be moved there.
    train_device = _resolve_torch_device(cfg.device)

    # Group objects by APK → build one MILBag per APK.
    by_apk: Dict[str, List[Mapping[str, Any]]] = {}
    for (apk_id, _oid), obj in objects.items():
        by_apk.setdefault(apk_id, []).append(obj)

    bags: List[MILBag] = []
    for apk_id, obj_list in by_apk.items():
        feats = np.stack([o["feature_vec"] for o in obj_list], axis=0)
        types = np.asarray(
            [
                _object_instance_type(
                    o["object_id"],
                    o["object_path"],
                    transform_families=o.get("transform_families") or None,
                    label_id=int(o.get("true_label_id", 0)),
                )
                for o in obj_list
            ],
            dtype=np.int64,
        )
        bag_label = max(o["true_label_id"] for o in obj_list)
        inst_labels = np.asarray(
            [o["true_label_id"] for o in obj_list], dtype=np.float32
        )
        # L46 (2026-05-11): ablation — collapse all type IDs to 0 when
        # n_types == 1, disabling the per-type head specialisation.
        if cfg.ours_config.typed.n_types == 1:
            types[:] = 0

        bags.append(
            MILBag(
                bag_id=apk_id,
                # L48 (2026-05-11): move feature tensors to training device
                # so mil_trainer picks up GPU placement automatically via
                # ref_device = bags[0].features.device.
                features=torch.from_numpy(feats).float().to(train_device),
                types=torch.from_numpy(types).to(train_device),
                bag_label=int(bool(bag_label)),
                instance_labels=torch.from_numpy(inst_labels).to(train_device),
            )
        )

    if not bags:
        raise ValueError(
            "train_ours_baseline: no bags were constructed; check "
            "region_rows / apk_index wiring"
        )

    # Resolve ``input_dim`` from actual handcrafted feature length.
    input_dim = bags[0].features.shape[1]
    effective_n_types = cfg.ours_config.typed.n_types
    if cfg.ours_config.typed.input_dim != input_dim:
        # Build a corrected config in place (dataclass is frozen).
        corrected_typed = TypedEncoderConfig(
            input_dim=input_dim,
            hidden_dim=cfg.ours_config.typed.hidden_dim,
            num_trunk_layers=cfg.ours_config.typed.num_trunk_layers,
            head_hidden_dim=cfg.ours_config.typed.head_hidden_dim,
            dropout=cfg.ours_config.typed.dropout,
            n_types=effective_n_types,
        )
        corrected_ours = OursConfig(
            typed=corrected_typed,
            mil_pooling=cfg.ours_config.mil_pooling,
            topk=cfg.ours_config.topk,
            noisy_or=cfg.ours_config.noisy_or,
            attention=cfg.ours_config.attention,
            use_feature_attention=cfg.ours_config.use_feature_attention,
        )
    else:
        corrected_ours = cfg.ours_config

    trainer_cfg = MILTrainerConfig(
        ours_config=corrected_ours,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        lambda_diff_pseudo=cfg.lambda_diff_pseudo,
        lambda_sparsity=cfg.lambda_sparsity,
        bag_pos_weight=cfg.bag_pos_weight,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
        train_max_bag_size=cfg.train_max_bag_size,
        train_min_positive_fraction=cfg.train_min_positive_fraction,
        supervision_mode=cfg.supervision_mode,
    )
    module = train_ours(bags, trainer_cfg)

    # L47 (2026-05-11): resolve attention_auto direction on TRAINING data
    # so that inference never needs ground-truth labels.
    if cfg.scoring_mode == "attention_auto":
        import torch as _torch
        module.eval()
        pos_attn_vals = []
        neg_attn_vals = []
        with _torch.no_grad():
            for bag in bags:
                if bag.bag_label == 1 and bag.instance_labels is not None:
                    _bag_logit, _attn, _inst = module(bag.features, bag.types)
                    attn_np = _attn.cpu().numpy()
                    il = bag.instance_labels.cpu().numpy()
                    if il.sum() > 0 and il.sum() < len(il):
                        pos_attn_vals.append(float(attn_np[il == 1].mean()))
                        neg_attn_vals.append(float(attn_np[il == 0].mean()))
        if pos_attn_vals and neg_attn_vals:
            overall_pos = float(np.mean(pos_attn_vals))
            overall_neg = float(np.mean(neg_attn_vals))
            if overall_pos >= overall_neg:
                module._resolved_scoring_direction = "attention_x_bag"
            else:
                module._resolved_scoring_direction = "attention_anomaly"
        else:
            # Cannot resolve (e.g. all bags are all-positive) — default
            module._resolved_scoring_direction = "attention_x_bag"

    # Rebuild config so that on-disk save reflects the corrected shape.
    resolved_cfg = OursBaselineConfig(
        ours_config=corrected_ours,
        handcrafted_config=cfg.handcrafted_config,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lambda_diff_pseudo=cfg.lambda_diff_pseudo,
        lambda_sparsity=cfg.lambda_sparsity,
        bag_pos_weight=cfg.bag_pos_weight,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
        train_max_bag_size=cfg.train_max_bag_size,
        train_min_positive_fraction=cfg.train_min_positive_fraction,
        supervision_mode=cfg.supervision_mode,
        threshold=cfg.threshold,
        loader_cache_size=cfg.loader_cache_size,
        scoring_mode=cfg.scoring_mode,
        device=cfg.device,
    )
    return OursBaselineModel(module, resolved_cfg)


def train_ours_baseline(
    region_rows: Sequence[Mapping],
    apk_index: Mapping[str, Path],
    config: Optional[OursBaselineConfig] = None,
) -> OursBaselineModel:
    """Train an Ours (Typed-Instance MIL) model on region rows."""

    cfg = config or OursBaselineConfig()
    loader = ObjectByteLoader(cache_size=cfg.loader_cache_size)
    objects, _feat_names = _aggregate_object_features(
        region_rows,
        apk_index,
        handcrafted_config=cfg.handcrafted_config,
        loader=loader,
    )
    return train_ours_baseline_from_objects(objects, cfg)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _object_localization_samples(
    regions: Sequence[OursRegionPrediction],
    objects: Sequence[OursObjectPrediction],
) -> list[dict]:
    """Build ``localization_metrics`` samples for every positive object.

    Mirrors :func:`android_packer.baselines.ngram_logreg._object_localization_samples`
    so the Ours report localisation block is comparable to other baselines'
    region-union IoU / boundary-error numbers.

    - ``truth``: union of ``(offset_start, offset_end)`` intervals from
      the object's true-positive regions.
    - ``prediction``: list containing a single span ``[min_start,
      max_end]`` over the regions the model flagged positive; empty
      when the model fires on none (mean_iou handles as a miss).
    """

    by_object: dict[tuple[str, str], list[OursRegionPrediction]] = {}
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


def _predict_impl(
    *,
    module: Any,
    baseline_config: OursBaselineConfig,
    region_rows: Iterable[Mapping],
    apk_index: Mapping[str, Path],
    loader: Optional[ObjectByteLoader],
) -> OursResult:
    import numpy as np
    import torch

    from android_packer.evaluation import (
        binary_classification_metrics,
        localization_metrics,
        ranking_metrics,
    )

    _VALID_SCORING_MODES = {
        "instance_logit", "attention", "attention_x_bag",
        "attention_anomaly", "attention_auto",
    }
    if baseline_config.scoring_mode not in _VALID_SCORING_MODES:
        raise ValueError(
            f"Invalid scoring_mode={baseline_config.scoring_mode!r}; "
            f"must be one of {sorted(_VALID_SCORING_MODES)}"
        )

    if loader is None:
        loader = ObjectByteLoader(cache_size=baseline_config.loader_cache_size)

    rows = list(region_rows)
    objects, _feat_names = _aggregate_object_features(
        rows,
        apk_index,
        handcrafted_config=baseline_config.handcrafted_config,
        loader=loader,
    )

    # Score every APK as one bag.
    by_apk: Dict[str, List[Dict[str, Any]]] = {}
    for (apk_id, _oid), obj in objects.items():
        by_apk.setdefault(apk_id, []).append(obj)

    apk_predictions: List[OursApkPrediction] = []
    object_predictions: List[OursObjectPrediction] = []
    region_predictions: List[OursRegionPrediction] = []

    module.eval()
    threshold = baseline_config.threshold
    # L44: attention_auto resolves the actual direction on the first
    # positive-bag APK by checking whether positive instances correlate
    # with high or low attention.  Once resolved it stays fixed.
    _resolved_scoring_mode: Optional[str] = None
    # L48 (2026-05-11): resolve inference device from model parameters so
    # feature tensors are moved to GPU when the model was trained on GPU.
    try:
        infer_device = next(module.parameters()).device
    except StopIteration:
        infer_device = torch.device("cpu")

    for apk_id, obj_list in by_apk.items():
        feats = torch.from_numpy(
            np.stack([o["feature_vec"] for o in obj_list], axis=0)
        ).float().to(infer_device)  # L48: move to model device (GPU or CPU)
        # L47 (2026-05-11): CRITICAL FIX — at inference time, do NOT pass
        # ground-truth label_id to _object_instance_type.  Passing label_id
        # caused a data leak: packed objects were routed to packed-specific
        # type heads while benign objects went to benign_other, giving the
        # model the answer in its input.  The path-based heuristic (path 3)
        # is the only legitimate inference-time routing.
        types = torch.from_numpy(
            np.asarray(
                [
                    _object_instance_type(
                        o["object_id"],
                        o["object_path"],
                    )
                    for o in obj_list
                ],
                dtype=np.int64,
            )
        ).to(infer_device)  # L48: move to model device (GPU or CPU)
        # L46 (2026-05-11): collapse types for n_types=1 ablation.
        if baseline_config.ours_config.typed.n_types == 1:
            types[:] = 0
        with torch.no_grad():
            bag_logit, attention, instance_logits = module(feats, types)
            bag_score = float(torch.sigmoid(bag_logit).item())
            # L44 (2026-05-10): compute per-instance scores based on
            # scoring_mode.  Under bag-only supervision, instance_logits
            # are uncalibrated; attention weights are the calibrated
            # instance-level signal.
            scoring_mode = _resolved_scoring_mode or baseline_config.scoring_mode

            # --- attention_auto: resolve direction ---
            # L47 (2026-05-11): CRITICAL FIX — previously this block read
            # true_label_id from test objects to determine scoring direction,
            # which is a ground-truth leak.  The direction should be resolved
            # during TRAINING (on the training validation set) and stored as
            # a model attribute.  At inference time, we use the stored
            # direction.  If none was stored (legacy models), default to
            # attention_anomaly which is the empirically dominant direction
            # (packed = low attention = anomalous).
            if scoring_mode == "attention_auto":
                if _resolved_scoring_mode is None:
                    # Check if the model stored a resolved direction during training
                    stored_direction = getattr(module, "_resolved_scoring_direction", None)
                    if stored_direction is not None:
                        _resolved_scoring_mode = stored_direction
                    else:
                        # Default: attention_x_bag — packed instances receive
                        # HIGH attention because they're structurally distinct
                        # from the benign majority in feature space.
                        _resolved_scoring_mode = "attention_x_bag"
                scoring_mode = _resolved_scoring_mode

            if scoring_mode == "attention":
                attn_np = attention.cpu().numpy()
                attn_max = float(attn_np.max())
                if attn_max > 0:
                    inst_scores = (attn_np / attn_max).tolist()
                else:
                    inst_scores = attn_np.tolist()
            elif scoring_mode == "attention_x_bag":
                attn_np = attention.cpu().numpy()
                attn_max = float(attn_np.max())
                if attn_max > 0:
                    inst_scores = ((attn_np / attn_max) * bag_score).tolist()
                else:
                    inst_scores = (attn_np * bag_score).tolist()
            elif scoring_mode == "attention_anomaly":
                # MIL attention concentrates on "typical" instances that
                # explain the bag pattern.  Payload instances are anomalous
                # (low attention).  Score = (1 - normalized_attn) * bag_score
                # so low-attention + packed-bag → high suspicion.
                attn_np = attention.cpu().numpy()
                attn_max = float(attn_np.max())
                if attn_max > 0:
                    inst_scores = (
                        (1.0 - attn_np / attn_max) * bag_score
                    ).tolist()
                else:
                    inst_scores = [bag_score] * len(attn_np)
            else:
                # "instance_logit" — legacy behavior.
                inst_scores = torch.sigmoid(instance_logits).cpu().numpy().tolist()

        apk_true = max(o["true_label_id"] for o in obj_list)
        apk_pred = int(bag_score >= threshold)

        # L44: for attention_anomaly (or auto-resolved to anomaly), scores
        # live in [0, bag_score] so scale the instance threshold accordingly.
        if scoring_mode == "attention_anomaly":
            inst_threshold = threshold * bag_score
        else:
            inst_threshold = threshold

        positive_object_count = 0
        region_count_apk = 0
        positive_region_count_apk = 0
        for obj, inst_score in zip(obj_list, inst_scores):
            obj_pred = int(inst_score >= inst_threshold)
            if obj_pred == 1:
                positive_object_count += 1
            for row in obj["region_rows"]:
                region_count_apk += 1
                region_true = int(row.get("label_id", 0))
                positive_region_count_apk += region_true
                region_predictions.append(
                    OursRegionPrediction(
                        apk_id=apk_id,
                        object_id=obj["object_id"],
                        region_id=row.get("region_id", ""),
                        object_path=obj["object_path"],
                        offset_start=int(row.get("offset_start", 0)),
                        offset_end=int(row.get("offset_end", 0)),
                        score=float(inst_score),  # project instance -> regions
                        predicted_label_id=obj_pred,
                        true_label_id=region_true,
                    )
                )
            object_predictions.append(
                OursObjectPrediction(
                    apk_id=apk_id,
                    object_id=obj["object_id"],
                    object_path=obj["object_path"],
                    score=float(inst_score),
                    predicted_label_id=obj_pred,
                    true_label_id=obj["true_label_id"],
                    region_count=obj["region_count"],
                    positive_region_count=obj["positive_region_count"],
                )
            )

        apk_predictions.append(
            OursApkPrediction(
                apk_id=apk_id,
                score=bag_score,
                predicted_label_id=apk_pred,
                true_label_id=apk_true,
                object_count=len(obj_list),
                positive_object_count=positive_object_count,
                region_count=region_count_apk,
                positive_region_count=positive_region_count_apk,
            )
        )

    # Build report with the same shape as :mod:`ngram_logreg` /
    # :mod:`payload_hunter_lite` so the multi-baseline aggregator can
    # consume it without special-casing.  The three evaluation helpers
    # have positional-kwarg APIs; match them exactly:
    #
    # * ``binary_classification_metrics(truth=, predictions=, scores=)``
    # * ``ranking_metrics(groups=, truth=, scores=)``
    # * ``localization_metrics([{"truth": […], "prediction": […]}])``
    region_metrics = binary_classification_metrics(
        truth=[r.true_label_id for r in region_predictions],
        predictions=[r.predicted_label_id for r in region_predictions],
        scores=[r.score for r in region_predictions],
    )
    object_metrics = binary_classification_metrics(
        truth=[o.true_label_id for o in object_predictions],
        predictions=[o.predicted_label_id for o in object_predictions],
        scores=[o.score for o in object_predictions],
    )
    apk_metrics = binary_classification_metrics(
        truth=[a.true_label_id for a in apk_predictions],
        predictions=[a.predicted_label_id for a in apk_predictions],
        scores=[a.score for a in apk_predictions],
    )

    ranking = ranking_metrics(
        groups=[o.apk_id for o in object_predictions],
        truth=[o.true_label_id for o in object_predictions],
        scores=[o.score for o in object_predictions],
    )

    loc_samples = _object_localization_samples(
        region_predictions, object_predictions
    )
    localization = localization_metrics(loc_samples)

    report = {
        "baseline": "ours",
        "threshold": threshold,
        "mil_pooling": baseline_config.ours_config.mil_pooling,
        "typed_instance_types": list(TYPED_INSTANCE_TYPES),
        "counts": {
            "regions": len(region_predictions),
            "objects": len(object_predictions),
            "apks": len(apk_predictions),
        },
        "metrics": {
            "region": region_metrics.to_dict(),
            "object": object_metrics.to_dict(),
            "apk": apk_metrics.to_dict(),
        },
        "ranking": {"object": ranking.to_dict()},
        "localization": {"object": localization.to_dict()},
    }

    return OursResult(
        region_predictions=region_predictions,
        object_predictions=object_predictions,
        apk_predictions=apk_predictions,
        report=report,
    )


def run_ours_baseline(
    region_rows: Iterable[Mapping],
    apk_index: Mapping[str, Path],
    model: OursBaselineModel,
    *,
    loader: Optional[ObjectByteLoader] = None,
) -> OursResult:
    """Thin wrapper used by :mod:`run_synthetic_multi_baseline` stage-2.

    The caller is responsible for training the model via
    :func:`train_ours_baseline`; this entry-point only performs
    inference.
    """

    return model.predict(region_rows, apk_index, loader=loader)
