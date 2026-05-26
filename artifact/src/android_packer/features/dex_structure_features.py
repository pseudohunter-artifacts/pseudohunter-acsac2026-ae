"""DEX structural features for Typed-Instance MIL (Tier 1A improvement).

Extends the 15-dim handcrafted feature vector with DEX section-aware
features derived from ``dex_item_parser.parse_dex_item_spans()``.

Background
----------
The current 15-dim Pass-2a features (entropy, byte distribution, region
position) are domain-agnostic statistics.  A packed ``code_item`` region
and a benign ``string_data`` region can have indistinguishable entropy
profiles, but they sit in structurally different DEX sections.  Adding
section-distribution features lets the model learn "high-entropy bytes
in code_item are suspicious; high-entropy bytes in string_data are
normal compressed strings".

Feature vector (12 dimensions, ``DEX_STRUCTURE_FEATURE_NAMES``)
---------------------------------------------------------------
- ``dex_sec_header``, ``dex_sec_string_ids``, ``dex_sec_type_ids``,
  ``dex_sec_proto_ids``, ``dex_sec_field_ids``, ``dex_sec_method_ids``,
  ``dex_sec_class_defs``, ``dex_sec_code_item``, ``dex_sec_string_data``,
  ``dex_sec_other``:  fraction of the region's bytes that belong to each
  DEX section (sums to 1.0 across covered bytes; 0.0 for an uncovered
  byte whose section is ``other``).
- ``dex_dominant_section``: argmax of the 10-dim distribution, normalised
  to [0, 1] by dividing by 9 (number of sections minus 1). Provides a
  single compact signal about "what kind of region is this".
- ``dex_cross_section_count_log2``: number of distinct section boundaries
  within the region (how fragmented is this region across sections?),
  log-compressed to keep the range small.

For non-DEX objects (assets, native stubs, etc.) this module emits a
zero vector of the same length so that the upstream feature assembly
remains a fixed-width tensor.

Anti-bypass design
------------------
These features enter the model as **feature-space inputs**, NOT as hard
decision thresholds (see ``docs/method/improvement_plan_L47.md`` §4).
The model must still rely on the learned weights to detect packed payload;
section type merely provides structural context that makes the learning
easier.  Even if a packer author randomises their section layout, the
model degrades gracefully toward the section-agnostic entropy baseline.

Usage
-----
Typical call site is ``_aggregate_object_features()`` in
``src/android_packer/baselines/ours.py``::

    if cfg.handcrafted_config.include_dex_structure:
        dex_feats = extract_dex_structure_features(obj_bytes, region_offset, region_size)
        feat_vec = np.concatenate([feat_vec, np.array(dex_feats, dtype=np.float32)])

The returned list always has exactly ``len(DEX_STRUCTURE_FEATURE_NAMES)``
elements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from android_packer.features.dex_item_parser import (
    DEX_ITEM_TYPES,
    DexParseError,
    parse_dex_item_spans,
)


__all__ = [
    "DEX_STRUCTURE_FEATURE_NAMES",
    "N_DEX_STRUCTURE_FEATURES",
    "extract_dex_structure_features",
    "extract_dex_structure_features_with_cache",
]


# ---------------------------------------------------------------------------
# Public feature vocabulary (order is pinned — do not reorder once a
# checkpoint is trained on this module's output).
# ---------------------------------------------------------------------------

#: One fractional coverage dim per DEX item type, then two derived scalars.
DEX_STRUCTURE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"dex_sec_{name}" for name in DEX_ITEM_TYPES
) + (
    "dex_dominant_section",
    "dex_cross_section_count_log2",
)

N_DEX_STRUCTURE_FEATURES: int = len(DEX_STRUCTURE_FEATURE_NAMES)  # 12


# ---------------------------------------------------------------------------
# Zero vector (returned for non-DEX objects)
# ---------------------------------------------------------------------------

_ZERO_VECTOR: list[float] = [0.0] * N_DEX_STRUCTURE_FEATURES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_features_from_spans(
    spans: Any,
    region_offset: int,
    region_size: int,
) -> List[float]:
    """Inner computation: given pre-parsed spans, compute the feature vector.

    This is split out so both the public API and the cache-aware variant
    can share the same logic without re-parsing the DEX.
    """
    n_types = len(DEX_ITEM_TYPES)

    region_end = region_offset + region_size
    type_counts = [0] * n_types

    # Build a list of (start, end, type_idx) for spans that overlap the window
    overlapping = []
    for span in spans:
        span_end = span.offset + span.size
        if span_end <= region_offset or span.offset >= region_end:
            continue
        overlap_start = max(span.offset, region_offset)
        overlap_end = min(span_end, region_end)
        overlapping.append((overlap_start, overlap_end, span.item_type))

    # Sort by start offset so we can track section transitions
    overlapping.sort(key=lambda t: t[0])

    # Sweep over the region, crediting covered bytes to their section type
    covered_bytes = 0
    prev_boundary = region_offset
    last_type_seen: Optional[int] = None
    distinct_boundaries: set[int] = set()
    other_idx = n_types - 1  # "other" is the last entry in DEX_ITEM_TYPES

    for ov_start, ov_end, type_idx in overlapping:
        # Gap before this span: credit to "other"
        if ov_start > prev_boundary:
            gap = ov_start - prev_boundary
            type_counts[other_idx] += gap
            covered_bytes += gap
            if last_type_seen is not None and last_type_seen != other_idx:
                distinct_boundaries.add(ov_start)
            last_type_seen = other_idx

        span_len = ov_end - ov_start
        type_counts[type_idx] += span_len
        covered_bytes += span_len

        # Record section boundaries (transitions between different sections)
        if last_type_seen is not None and last_type_seen != type_idx:
            distinct_boundaries.add(ov_start)
        last_type_seen = type_idx
        prev_boundary = ov_end

    # Trailing gap after last span: credit to "other"
    if prev_boundary < region_end:
        gap = region_end - prev_boundary
        type_counts[other_idx] += gap
        covered_bytes += gap
        if last_type_seen is not None and last_type_seen != other_idx:
            distinct_boundaries.add(prev_boundary)

    # --- section distribution (10 dims) --------------------------------------
    total = covered_bytes if covered_bytes > 0 else 1  # avoid div-by-zero
    section_fracs = [c / total for c in type_counts]

    # --- dominant section (1 dim, normalised to [0, 1]) ----------------------
    dominant_idx = max(range(n_types), key=lambda i: type_counts[i])
    dominant_norm = dominant_idx / max(n_types - 1, 1)  # in [0, 1]

    # --- cross-section count (1 dim, log-compressed) -------------------------
    n_boundaries = len(distinct_boundaries)
    cross_section_log2 = math.log2(n_boundaries + 1)  # log2(1)=0 for pure regions

    return section_fracs + [dominant_norm, cross_section_log2]


# Sentinel: parse failed (non-DEX or malformed) — stored in cache to avoid re-trying
_PARSE_FAILED = object()


def extract_dex_structure_features(
    object_bytes: bytes,
    region_offset: int,
    region_size: int,
) -> List[float]:
    """Compute DEX structural features for one 4 KB window of an object.

    Parameters
    ----------
    object_bytes:
        Full decompressed bytes of the APK member (a single DEX file or
        any other ZIP entry). If the bytes do not start with a valid DEX
        magic, a zero vector is returned.
    region_offset:
        Byte offset of the region window inside ``object_bytes``.
    region_size:
        Length of the region window in bytes.

    Returns
    -------
    List[float]
        Feature vector of length ``N_DEX_STRUCTURE_FEATURES`` (12).
        All values are in [0, 1] or small non-negative log-compressed
        values.  Always returns the zero vector for non-DEX objects.

    .. note::
        For batch processing of many windows from the same DEX file, prefer
        :func:`extract_dex_structure_features_with_cache` to avoid parsing
        the same DEX bytes repeatedly (once per call vs. once per object).
    """

    if region_size <= 0:
        return list(_ZERO_VECTOR)

    # --- attempt DEX parse ---------------------------------------------------
    try:
        spans = parse_dex_item_spans(object_bytes)
    except (DexParseError, TypeError, ValueError):
        # Non-DEX or malformed: structural features are undefined.
        return list(_ZERO_VECTOR)

    return _compute_features_from_spans(spans, region_offset, region_size)


def extract_dex_structure_features_with_cache(
    object_bytes: bytes,
    region_offset: int,
    region_size: int,
    cache: Dict[Tuple[str, str], Any],
    cache_key: Tuple[str, str],
) -> List[float]:
    """Cache-aware variant of :func:`extract_dex_structure_features`.

    Parses ``object_bytes`` only on the **first call** for a given
    ``cache_key``; subsequent calls for the same key reuse the stored
    span list.  This avoids the O(rows) DEX parsing cost when many region
    windows are extracted from the same DEX file.

    Parameters
    ----------
    object_bytes:
        Full decompressed bytes of the APK member.
    region_offset, region_size:
        Window position inside ``object_bytes``.
    cache:
        A ``dict`` shared across all calls for the same APK/fold.
        Maps ``cache_key`` → parsed spans list (or ``_PARSE_FAILED``
        sentinel if the object is non-DEX / malformed).  Pass an empty
        ``{}`` at the start of each fold and let it grow.
    cache_key:
        Hashable key identifying the specific APK member, e.g.
        ``(apk_id, object_path)``.

    Returns
    -------
    List[float]
        Feature vector of length ``N_DEX_STRUCTURE_FEATURES`` (12).
    """

    if region_size <= 0:
        return list(_ZERO_VECTOR)

    if cache_key not in cache:
        try:
            cache[cache_key] = parse_dex_item_spans(object_bytes)
        except (DexParseError, TypeError, ValueError):
            cache[cache_key] = _PARSE_FAILED

    spans = cache[cache_key]
    if spans is _PARSE_FAILED:
        return list(_ZERO_VECTOR)

    return _compute_features_from_spans(spans, region_offset, region_size)
