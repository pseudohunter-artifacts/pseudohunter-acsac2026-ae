"""Byte-level feature extraction for baselines and ablations.

This package intentionally has **zero third-party dependencies**. Every
feature can be computed from ``bytes`` using only the Python standard
library. Consumers who want to turn the resulting dicts into dense
numpy arrays do so at the call site (e.g. the n-gram baseline pulls in
sklearn), which keeps the core pipeline installable without numpy.
"""

from android_packer.features.byte_features import (
    ByteFeatureConfig,
    ObjectByteLoader,
    RegionFeatureVector,
    extract_region_bytes,
    region_byte_features,
)
from android_packer.features.dex_features import (
    DexStructuralFeatureConfig,
    extract_region_structural_features,
)
from android_packer.features.dex_item_parser import (
    DEX_ITEM_TYPES,
    DexItemSpan,
    DexParseError,
    parse_dex_item_spans,
    region_item_type_labels,
)
from android_packer.features.entropy_delta import (
    DEFAULT_NEIGHBOR_HALF_WINDOW,
    EntropyDeltaConfig,
    EntropyDeltaKeys,
    add_entropy_deltas_inplace,
    compute_entropy_deltas_for_group,
    entropy_delta_feature_names,
)
from android_packer.features.handcrafted import (
    HandcraftedFeatureConfig,
    RegionByteLoader,
    extract_handcrafted_features,
    handcrafted_feature_names,
)

__all__ = [
    "ByteFeatureConfig",
    "ObjectByteLoader",
    "RegionFeatureVector",
    "extract_region_bytes",
    "region_byte_features",
    # DEX-aware scalar features (F2a). **Disabled by default** for the Ours
    # baseline — kept as an optional ablation input. See
    # ``docs/method/ours_method_spec.md`` §3.2.1 and §6.3
    # (``ours_with_scalar_struct``). The authoritative research rationale
    # for the demotion is in ``docs/research_framing.md`` §4.3 / §5.3.
    "DexStructuralFeatureConfig",
    "extract_region_structural_features",
    # DEX item-type span supervision (F2b). Feeds F5 auxiliary loss; see
    # ``docs/method/ours_method_spec.md`` §3.2.1b and §5.1.
    "DEX_ITEM_TYPES",
    "DexItemSpan",
    "DexParseError",
    "parse_dex_item_spans",
    "region_item_type_labels",
    # Entropy-delta contrast features (F-Lite-b). Three locality scopes
    # (neighbor / entry / apk) computed from a shared input row schema;
    # see ``docs/method/ours_method_spec.md`` §11.3.1 and
    # ``src/android_packer/features/entropy_delta.py``.
    "DEFAULT_NEIGHBOR_HALF_WINDOW",
    "EntropyDeltaConfig",
    "EntropyDeltaKeys",
    "add_entropy_deltas_inplace",
    "compute_entropy_deltas_for_group",
    "entropy_delta_feature_names",
    # Handcrafted 15-dim (Pass-2a) feature assembly (F-Lite-b). Groups A
    # + B + G from ``docs/method/ours_method_spec.md`` §11.3.1; Groups
    # C/D/E/F (the remaining 19 dims to reach the spec's 34 dims) are
    # deferred to Pass-2b and stubbed behind ``include_*=False``.
    "HandcraftedFeatureConfig",
    "RegionByteLoader",
    "extract_handcrafted_features",
    "handcrafted_feature_names",
]
