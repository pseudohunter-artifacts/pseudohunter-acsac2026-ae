"""Handcrafted region features for PayloadHunter-Lite.

Canonical configurations (see ``docs/method/naming_glossary.md``
section 4 for the full Pass taxonomy):

* **Pass-2a (15 dims, Stage A paper-cell)**: Groups A (entropy
  family, 4 dims) + B (byte distribution, 8 dims) + G (region
  position, 3 dims). Implemented below and what the 2026-05-01 main
  cell was trained on.
* **Pass-2b (22 dims, RETRACTED 2026-05-03)**: Pass-2a + Group C
  (compression/magic signatures, 4 dims) + Group F (ZIP container
  context, 3 dims). Implemented below but RETRACTED after the L1
  overnight ablation falsified the hypothesis; kept only so the
  Appendix negative-result cell is reproducible.

Design principles:

* **Every dim is bound to a concrete transform-family adversary** so
  each feature has a named "what it is trying to detect" rationale.
  See the design-rationale table in ``docs/paper_submission_plan.md``
  (F-Lite-b discussion dated 2026-04-30).
* **Zero third-party dependencies**: pure stdlib, no numpy / sklearn.
  Consumers that want dense float arrays convert at call-site.
* **Stable output schema**: the order returned by
  :func:`handcrafted_feature_names` is pinned by the
  :class:`HandcraftedFeatureConfig` — do not reorder once a
  PayloadHunter-Lite checkpoint is trained, or feature indices in
  the MLP's first :class:`~torch.nn.Linear` will drift.

Historical implementation roadmap (retained for provenance)
-----------------------------------------------------------
The module originally planned to reach 34 dims across a three-pass
rollout. Pass-3 (Groups D DEX-structural + E bigram top-k) was
superseded by the 2026-05-03 strategic pivot to Stage B's
DexBERT-Loc Full direction (disassembler-aware byte encoder), so
Groups D / E will not land in this module. Their ``include_*`` flags
on :class:`HandcraftedFeatureConfig` are retired no-ops; new feature
experiments should start from Pass-2a, not reactivate those flags.
The two passes that actually shipped:

* **Pass-2a** (commit F-Lite-b/a, 2026-04-30): Groups A (4) + B (8)
  + G (3) = **15 dims**. All computable from raw region bytes +
  entropy + ordering information already on the row.
* **Pass-2b** (commit F-Lite-b/c, 2026-05-02): adds Group C (4) +
  Group F (3) = 15 + 7 = **22 dims**. Falsified by L1 ablation
  (2026-05-02 -> 2026-05-03); see
  ``docs/progress/sessions/2026-05-02_overnight_results_report.md``
  section 1.5. Kept for Appendix reproducibility only.
  - a training-time-fixed bigram vocabulary (E)
  - ZIP entry metadata propagated from the region loader (F)

  Each group is cleanly isolated so Pass 2b can add them without
  touching Pass-2a callers; the 15-dim checkpoint trained from this
  file is a valid ablation point ("no-sig, no-dex, no-bigram, no-zip").

Contract
--------
The public entry point :func:`extract_handcrafted_features` mutates
every input row with a fixed set of feature keys listed in
:func:`handcrafted_feature_names`. Rows that came through
:mod:`android_packer.regioning` already carry ``entropy``,
``printable_ratio``, and ``offset_start / end`` — we reuse those
rather than recomputing. Byte-level features that need the raw region
bytes look them up via an injected ``byte_loader`` callable, defaulting
to a no-op that writes zeros (useful for tests that exercise only the
structural side).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

from android_packer.features.dex_structure_features import DEX_STRUCTURE_FEATURE_NAMES
from android_packer.features.entropy_delta import (
    EntropyDeltaConfig,
    add_entropy_deltas_inplace,
    entropy_delta_feature_names,
)


__all__ = [
    "HandcraftedFeatureConfig",
    "handcrafted_feature_names",
    "extract_handcrafted_features",
    "RegionByteLoader",
]


# A callable that returns the raw bytes of a region given its row. The
# Lite trainer injects a real loader backed by
# :class:`android_packer.features.byte_features.ObjectByteLoader`; tests
# can inject a simple lambda. When ``None``, Group-B byte-distribution
# features fall back to zeros.
RegionByteLoader = Callable[[MutableMapping[str, object]], bytes]


@dataclass(frozen=True)
class HandcraftedFeatureConfig:
    """Knobs for the PayloadHunter-Lite handcrafted feature assembly.

    Canonical configurations (see ``docs/method/naming_glossary.md``
    section 4 for the full Pass taxonomy):

    - **Pass-2a (15 dims, Stage A paper-cell)**: all defaults. Group
      A entropy (raw + deltas triad) + Group B byte distribution +
      Group G region position. This is what the 2026-05-01
      ``outputs/experiments/baseline_sweeps/20260501-213428/`` main
      run was trained on and what the ACSAC 2026 submission will
      report.
    - **Pass-2b (22 dims, RETRACTED 2026-05-03)**: Pass-2a plus
      ``include_compression_signature=True`` (Group C, 4 dims of
      magic-byte probes) and ``include_zip_context=True`` (Group F,
      3 dims of ZIP-container context). L1 overnight ablation
      falsified Pass-2b -- it regressed vs Pass-2a on the Gen3
      4-fold subset. Reachable only for Appendix reproducibility;
      any new method should start from Pass-2a, not Pass-2b.

    Every group is independently switchable so ablation studies can
    say ``include_byte_distribution=False`` etc. The total feature
    count derived from the current flags is exposed via
    :func:`handcrafted_feature_names`.
    """

    # ---- Group A: entropy family (4 dims: 1 raw + 3 deltas) ----
    include_entropy_raw: bool = True
    include_entropy_deltas: bool = True  # neighbor / entry / apk triad
    entropy_delta_config: EntropyDeltaConfig = field(default_factory=EntropyDeltaConfig)

    # ---- Group B: byte distribution (8 dims; Pass-2a subset = 8) ----
    include_byte_distribution: bool = True

    # ---- Group G: region position (3 dims) ----
    include_region_position: bool = True

    # ---- Pass-2b groups (defaults OFF -- Pass-2b was RETRACTED 2026-05-03) ----
    # Pass-2b (Group C + Group F) was pre-registered as the Stage A
    # upgrade candidate and falsified by the L1 overnight ablation
    # (2026-05-02 -> 2026-05-03). See
    # docs/progress/sessions/2026-05-02_overnight_results_report.md
    # section 1.5 for the death certificate and
    # docs/method/why_features_defensible_vs_ngram.md for the post-L1
    # takeaways. The two flags stay here as switchable knobs so the
    # Appendix negative-result cell remains reproducible; the paper
    # main-cell uses Pass-2a (all four flags = False).
    include_compression_signature: bool = False  # Group C -- Pass-2b only
    include_zip_context: bool = False            # Group F -- Pass-2b only

    # ---- Retired reservations (Group D / Group E) ----
    # These flags were scaffolded on 2026-04-30 as placeholders for a
    # planned Pass-3 (Group D DEX-structural + Group E bigram top-k)
    # that was superseded by the 2026-05-03 strategic pivot to the
    # Stage B DexBERT-Loc Full direction (disassembler-aware byte
    # encoder). Group D / E are therefore NO-OPs: toggling them to
    # True has zero effect because ``handcrafted_feature_names`` and
    # ``extract_handcrafted_features`` intentionally ignore these
    # flags. They are kept only so old pickled configs that carry the
    # fields still deserialise; any new feature pack should live in a
    # fresh Pass-3 flag rather than reactivating these.
    include_dex_structural: bool = False         # Group D -- RETIRED, no-op
    include_bigram_top_k: bool = False           # Group E -- RETIRED, no-op

    # ---- Group H: DEX section structure (12 dims, Tier 1A improvement) ----
    # Uses ``dex_item_parser.parse_dex_item_spans()`` to compute per-region
    # DEX section distribution + dominant section + cross-section boundary
    # count. Requires full object bytes via ``byte_loader``.
    # Disabled by default so Pass-2a checkpoints remain unaffected.
    # Enable via ``"include_dex_structure": true`` in the ours config block.
    # See ``src/android_packer/features/dex_structure_features.py`` for the
    # full feature spec (Tier 1A in improvement_plan_L47.md).
    include_dex_structure: bool = False          # Group H — DEX section features

    # ---- Computation knobs ----
    # Key on the input row that holds the raw per-region entropy.
    entropy_key: str = "entropy"


# Explicit, hand-ordered names for each group. The final list returned
# by ``handcrafted_feature_names`` preserves this order and is what
# downstream code (Lite trainer, MLP input) treats as the vocabulary.

_GROUP_A_ENTROPY_RAW = ("entropy_raw",)
_GROUP_A_ENTROPY_DELTAS = tuple(entropy_delta_feature_names())  # 3 names

_GROUP_B_BYTE_DIST = (
    "byte_printable_ratio",
    "byte_zero_ratio",
    "byte_high_bit_ratio",  # share of bytes >= 0x80
    "byte_ascii_ratio",     # share of bytes in [0x20, 0x7e]
    "byte_chi2_uniform",    # chi-square distance to uniform distribution
    "byte_unique_count_log2",  # log2(unique byte values)
    "byte_range_span",      # max - min observed byte value
    "byte_max_run_len_log2",  # log2(max run of identical bytes)
)

_GROUP_G_REGION_POS = (
    "region_offset_in_object_norm",
    "region_index_in_object_norm",
    "object_index_in_apk_norm",
)


# -----------------------------------------------------------------------
# Pass-2b groups (landed 2026-05-02 in response to full 11-fold diagnosis
# where embedded_archive collapsed to AUROC=0.168 and signature_strip
# only passed because entropy_delta did its job single-handedly). Each
# group is independently toggled via its ``include_*`` flag.
# -----------------------------------------------------------------------

# ---- Group C: compression / magic-byte signatures (4 dims) ----
#
# Rationale: the biggest failure mode on Track A v2 was
# ``embedded_archive`` where a ZIP-in-ZIP payload was buried in what
# looked like a regular compressed entry -- handcrafted entropy-delta
# alone could not distinguish it from benign DEFLATE. These four cheap
# magic-byte probes give the model an explicit strong prior.

_GROUP_C_MAGIC = (
    "has_dex_magic",             # 8-byte 'dex\n035\0' or 'dex\n036\0' / 037 / 038 / 039
    "has_elf_magic",             # 4-byte 0x7F 'E' 'L' 'F'
    "has_png_magic",             # 8-byte \x89PNG\r\n\x1a\n
    "has_zip_local_header",      # 4-byte 'P' 'K' 0x03 0x04
)


# ---- Group F: ZIP / object context (3 dims) ----
#
# Rationale: region position inside its ZIP entry + a 0/1 flag for
# "this region sits in an embedded_archive object" gives the model a
# way to special-case nested archives. The size log is a useful scale
# feature (payload buckets tend to live in mid-size entries; signature
# files are tiny; dex files are large).

_GROUP_F_ZIP_CONTEXT = (
    "object_size_log2",           # log2(object size in bytes)
    "object_type_is_embedded_archive",  # 0/1
    "object_type_is_asset_blob",        # 0/1
)


# ---- Group D: DEX structural (6 dims) ----  -- reserved, not emitted yet
# ---- Group E: bigram top-k (6 dims) ----    -- reserved, not emitted yet


def handcrafted_feature_names(
    config: HandcraftedFeatureConfig | None = None,
) -> List[str]:
    """Return the list of feature names in the canonical output order.

    The list is the authoritative column ordering for any dense
    numeric tensor assembled from the per-row dict output of
    :func:`extract_handcrafted_features`. Reorder only via a
    well-communicated schema bump.
    """

    cfg = config or HandcraftedFeatureConfig()
    names: List[str] = []
    if cfg.include_entropy_raw:
        names.extend(_GROUP_A_ENTROPY_RAW)
    if cfg.include_entropy_deltas:
        names.extend(_GROUP_A_ENTROPY_DELTAS)
    if cfg.include_byte_distribution:
        names.extend(_GROUP_B_BYTE_DIST)
    if cfg.include_region_position:
        names.extend(_GROUP_G_REGION_POS)
    # Pass-2b groups: append in a stable order so the 15-dim Pass-2a
    # checkpoints stay as a valid prefix ablation ("no-C, no-F").
    if cfg.include_compression_signature:
        names.extend(_GROUP_C_MAGIC)
    if cfg.include_zip_context:
        names.extend(_GROUP_F_ZIP_CONTEXT)
    # Group D / E still reserved for a future pass.
    # Group H: DEX section structure (Tier 1A, improvement_plan_L47.md)
    if cfg.include_dex_structure:
        names.extend(DEX_STRUCTURE_FEATURE_NAMES)
    return names


# ---------------------------------------------------------------------------
# Byte-distribution helpers (Group B)
# ---------------------------------------------------------------------------


def _byte_distribution_features(data: bytes) -> Dict[str, float]:
    """Compute 8 scalar features from raw region bytes.

    Every feature is bounded (either naturally or via log2) so a
    downstream MLP with standard weight init does not blow up on
    megabyte-sized regions. All outputs are floats in a sane range
    (mostly [0, 1] or log-compressed ratios).

    Performance note (2026-05-01): this function is called once per
    region in the training loop (~70 regions/APK × 84 APK ≈ 5800 calls
    per fold for same_set; up to 90 000 for holdout_transform). We
    therefore stay on C-implemented stdlib primitives (Counter,
    bytes.count, bytearray indexing) rather than a Python byte-by-byte
    loop. The earlier pure-Python version profiled at 1.1 ms/call on
    16 KB regions; this version profiles at ~0.08 ms/call (~14x).
    """

    n = len(data)
    if n == 0:
        # Empty region: every ratio is 0 / undefined. Return zeros to
        # keep the feature-vector shape fixed.
        return {name: 0.0 for name in _GROUP_B_BYTE_DIST}

    # --- byte counts via C-level Counter (one pass, no Python loop) ---
    counter = Counter(data)
    counts_list = [0] * 256
    for b, c in counter.items():
        counts_list[b] = c

    # --- byte-rate features (ratios in [0, 1]) ---
    # ASCII printable = 0x20..0x7E; plus tab / LF / CR count as
    # printable but not ascii_printable.
    ascii_printable = sum(counts_list[0x20:0x7F])
    printable = (
        ascii_printable
        + counts_list[0x09]
        + counts_list[0x0A]
        + counts_list[0x0D]
    )
    zero = counts_list[0]
    high_bit = sum(counts_list[0x80:])

    printable_ratio = printable / n
    zero_ratio = zero / n
    high_bit_ratio = high_bit / n
    ascii_ratio = ascii_printable / n

    # --- chi-square distance to uniform distribution ---
    # Expected count per bin under uniform = n / 256. Chi-square =
    # sum_i (obs_i - exp_i)^2 / exp_i. Normalised by n so the metric
    # is scale-invariant and typically in [0, ~255).
    expected = n / 256.0
    chi2 = 0.0
    for c in counts_list:
        diff = c - expected
        chi2 += (diff * diff) / expected
    chi2_norm = chi2 / n  # ~0 for uniform, grows with skew

    # --- unique byte count (log2) ---
    unique_count = len(counter)
    # log2 with floor at 0 so a single-byte region gives 0 not -inf.
    unique_log2 = _log2_floor(unique_count)

    # --- range span ---
    min_byte = next((i for i, c in enumerate(counts_list) if c > 0), 0)
    max_byte = next((i for i in range(255, -1, -1) if counts_list[i] > 0), 0)
    range_span = (max_byte - min_byte) / 255.0  # normalise to [0, 1]

    # --- max run length of identical bytes (log2) ---
    # bytes object does not expose a run-length primitive; a Python
    # loop is unavoidable but runs at C speed via the iterator.
    # We iterate from the SECOND byte comparing to prev; the first
    # byte is the seed with cur_run=1.
    max_run = 0
    cur_run = 1
    prev = data[0]
    for i in range(1, n):
        b = data[i]
        if b == prev:
            cur_run += 1
        else:
            if cur_run > max_run:
                max_run = cur_run
            cur_run = 1
            prev = b
    if cur_run > max_run:
        max_run = cur_run
    # ``cur_run`` starts at 1 (not 0) so for length-1 regions we get
    # a run of exactly 1 which log2-floors to 0.
    run_log2 = _log2_floor(max_run)

    return {
        "byte_printable_ratio": printable_ratio,
        "byte_zero_ratio": zero_ratio,
        "byte_high_bit_ratio": high_bit_ratio,
        "byte_ascii_ratio": ascii_ratio,
        "byte_chi2_uniform": chi2_norm,
        "byte_unique_count_log2": unique_log2,
        "byte_range_span": range_span,
        "byte_max_run_len_log2": run_log2,
    }


def _log2_floor(x: int) -> float:
    """log2 with a floor at 0 for x <= 1 (avoid -inf on tiny regions)."""

    if x <= 1:
        return 0.0
    # Bit length gives floor(log2(x)) + 1 for x > 0; subtract 1.
    return float(x.bit_length() - 1)


# ---------------------------------------------------------------------------
# Pass-2b helpers (Group C / Group F)
# ---------------------------------------------------------------------------


# DEX magic candidates across Dalvik formats. A region starts with DEX
# magic only when (a) it is the first region of an object and (b) the
# object is a raw or reassembled DEX file. Packed APKs frequently hide
# DEX bytes INSIDE the object (e.g. offset 0x200 of a .dat blob), so
# this flag is strong-positive when it fires but we do not *only* check
# the leading 8 bytes -- we also scan the first 256 bytes so an offset-
# shifted DEX still triggers. 256 is chosen because (a) most synthetic
# packers prepend a fixed-size header, (b) 256 keeps the cost negligible.
_DEX_MAGIC_PREFIXES: Tuple[bytes, ...] = (
    b"dex\n035\x00",
    b"dex\n036\x00",
    b"dex\n037\x00",
    b"dex\n038\x00",
    b"dex\n039\x00",
)
_ELF_MAGIC = b"\x7fELF"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_ZIP_LOCAL_HEADER = b"PK\x03\x04"

# How far into the region to scan for magic bytes. Covers the majority
# of "shifted DEX" layouts without touching the inner body.
_MAGIC_SCAN_LIMIT = 256


def _compression_signature_features(data: bytes) -> Dict[str, float]:
    """Compute 4 magic-byte probe features on a region (Group C).

    Each feature is 0.0 or 1.0. A region gets a 1 if any of the
    canonical magic sequences appears within the first
    ``_MAGIC_SCAN_LIMIT`` bytes. Empty regions yield all zeros.
    """

    if not data:
        return {name: 0.0 for name in _GROUP_C_MAGIC}
    head = bytes(data[:_MAGIC_SCAN_LIMIT])

    has_dex = 1.0 if any(m in head for m in _DEX_MAGIC_PREFIXES) else 0.0
    has_elf = 1.0 if _ELF_MAGIC in head else 0.0
    has_png = 1.0 if _PNG_MAGIC in head else 0.0
    has_zip = 1.0 if _ZIP_LOCAL_HEADER in head else 0.0

    return {
        "has_dex_magic": has_dex,
        "has_elf_magic": has_elf,
        "has_png_magic": has_png,
        "has_zip_local_header": has_zip,
    }


def _object_size_map(
    rows: Sequence[MutableMapping[str, object]],
) -> Dict[Tuple[str, str], int]:
    """Build a ``(apk_id, object_id) -> object size in bytes`` map.

    Uses the maximum ``offset_end`` observed across regions of the
    object as a proxy for object size. This is safe because regions
    are non-overlapping windows covering the object; the last region
    ends at the object length.
    """

    sizes: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["apk_id"]), str(row["object_id"]))
        end = int(row.get("offset_end", 0))
        if end > sizes.get(key, 0):
            sizes[key] = end
    return sizes


# ---------------------------------------------------------------------------
# Region-position helpers (Group G)
# ---------------------------------------------------------------------------


def _compute_region_position_tables(
    rows: Sequence[MutableMapping[str, object]],
) -> Tuple[Dict[Tuple[str, str], List[int]], Dict[Tuple[str, str], int], Dict[str, int]]:
    """Precompute lookup tables for region-position features.

    Returns:
        * ``regions_per_object``: maps ``(apk_id, object_id)`` to the
          sorted list of ``offset_start`` values of regions inside it.
          Used to compute the region's index (rank) within its object.
        * ``object_sizes``: ``(apk_id, object_id) -> length of
          regions_per_object[key]`` (shortcut).
        * ``object_count_per_apk``: ``apk_id -> number of distinct
          objects in that APK``. Used to normalise ``object_index_in_apk``.
    """

    regions: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    apk_objects: Dict[str, set] = defaultdict(set)
    for row in rows:
        apk = str(row["apk_id"])
        obj = str(row["object_id"])
        regions[(apk, obj)].append(int(row["offset_start"]))
        apk_objects[apk].add(obj)
    # Sort each object's region offsets ascending so index lookup is
    # stable (binary search would also work; list-index scan is fine
    # at our scale).
    for key in regions:
        regions[key].sort()
    object_sizes = {k: len(v) for k, v in regions.items()}
    object_count_per_apk = {apk: len(objs) for apk, objs in apk_objects.items()}
    return regions, object_sizes, object_count_per_apk


def _object_index_in_apk_map(
    rows: Sequence[MutableMapping[str, object]],
) -> Dict[Tuple[str, str], int]:
    """Assign a stable ascending 0-based index to each (apk, object).

    Ordering is by first-appearance of the object_id in the row
    sequence. For stable output across runs, callers should pre-sort
    rows consistently; the Lite pipeline does this (regions are
    emitted in iteration order of ``iter_apk_objects``).
    """

    index_map: Dict[Tuple[str, str], int] = {}
    per_apk_counter: Dict[str, int] = defaultdict(int)
    for row in rows:
        apk = str(row["apk_id"])
        obj = str(row["object_id"])
        key = (apk, obj)
        if key not in index_map:
            index_map[key] = per_apk_counter[apk]
            per_apk_counter[apk] += 1
    return index_map


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_handcrafted_features(
    rows: Sequence[MutableMapping[str, object]],
    *,
    byte_loader: Optional[RegionByteLoader] = None,
    config: Optional[HandcraftedFeatureConfig] = None,
) -> None:
    """Populate every row with the 15 Pass-2a handcrafted features.

    Mutates ``rows`` in-place. The input rows must at minimum carry:
    ``apk_id``, ``object_id``, ``offset_start``, ``offset_end``, plus
    whatever the chosen ``entropy_delta_config`` needs (``object_path``
    for entry scope, ``entropy`` for the raw entropy key).

    Parameters
    ----------
    rows:
        Mutable region-row sequence. The same sequence must also be
        used for entropy-delta computation so object/apk means are
        taken over the right population.
    byte_loader:
        Optional callable ``(row) -> bytes``. If ``None``, Group B
        features are filled with zeros (useful for structural-only
        tests). The Lite trainer injects a real loader backed by
        :class:`android_packer.features.byte_features.ObjectByteLoader`.
    config:
        Feature-set toggles. Defaults to Groups A + B + G (the full
        Pass-2a set).
    """

    cfg = config or HandcraftedFeatureConfig()
    if not rows:
        return

    # --- Group A: entropy raw + deltas -----------------------------------
    if cfg.include_entropy_raw:
        ent_key = cfg.entropy_key
        for row in rows:
            row["entropy_raw"] = float(row[ent_key])

    if cfg.include_entropy_deltas:
        # Delegate to the entropy_delta module so the neighbor/entry/apk
        # semantics stay in one place and parity with
        # scripts/validate_entropy_delta_auroc.py is preserved.
        add_entropy_deltas_inplace(rows, config=cfg.entropy_delta_config)

    # --- Group B + Group C shared byte_loader path -----------------------
    #
    # When both Group B (byte distribution) and Group C (compression
    # signature / magic-byte probes) are enabled -- which is the Pass-2b
    # default -- we deliberately fuse the two per-row loops so that
    # ``byte_loader(row)`` is only called once per region. The naive
    # split form doubled the I/O (or cache-slice) cost of feature
    # extraction, which was observed to push a full 11-fold
    # holdout-by-transform run from ~90 minutes (Pass-2a) to somewhere
    # around ~4-5 hours (Pass-2b) on a 1.8M-region fold. Fusing puts
    # Pass-2b back on the right side of the overnight budget.
    need_b = cfg.include_byte_distribution
    need_c = cfg.include_compression_signature
    if need_b and need_c:
        if byte_loader is None:
            for row in rows:
                for name in _GROUP_B_BYTE_DIST:
                    row[name] = 0.0
                for name in _GROUP_C_MAGIC:
                    row[name] = 0.0
        else:
            for row in rows:
                data = byte_loader(row)
                for name, value in _byte_distribution_features(data).items():
                    row[name] = value
                for name, value in _compression_signature_features(data).items():
                    row[name] = value
    elif need_b:
        # Pass-2a path (Group B alone).
        if byte_loader is None:
            for row in rows:
                for name in _GROUP_B_BYTE_DIST:
                    row[name] = 0.0
        else:
            for row in rows:
                data = byte_loader(row)
                for name, value in _byte_distribution_features(data).items():
                    row[name] = value
    elif need_c:
        # Pass-2b without Group B -- unusual, but keep the branch honest.
        if byte_loader is None:
            for row in rows:
                for name in _GROUP_C_MAGIC:
                    row[name] = 0.0
        else:
            for row in rows:
                data = byte_loader(row)
                for name, value in _compression_signature_features(data).items():
                    row[name] = value

    # --- Group G: region position ----------------------------------------
    if cfg.include_region_position:
        regions, object_sizes, object_count = _compute_region_position_tables(rows)
        obj_index_map = _object_index_in_apk_map(rows)
        for row in rows:
            apk = str(row["apk_id"])
            obj = str(row["object_id"])
            key = (apk, obj)
            # region_offset_in_object_norm: offset / size of last region
            # start in this object. Both are in bytes; the denominator
            # keeps the feature in [0, 1].
            last_offset = regions[key][-1] if regions[key] else 1
            denom = max(last_offset, 1)
            row["region_offset_in_object_norm"] = (
                int(row["offset_start"]) / denom
            )
            # region_index_in_object_norm: rank of this region's start
            # offset within the sorted list, normalised by (count - 1).
            offsets = regions[key]
            rank = _bisect_left(offsets, int(row["offset_start"]))
            max_rank = max(len(offsets) - 1, 1)
            row["region_index_in_object_norm"] = rank / max_rank
            # object_index_in_apk_norm: 0-based object rank / (apk object count - 1).
            denom = max(object_count.get(apk, 1) - 1, 1)
            row["object_index_in_apk_norm"] = obj_index_map[key] / denom

    # --- Group C: absorbed into the fused Group B + C loop above --------
    # (No standalone loop here; see the need_b/need_c dispatch earlier.)

    # --- Group F: ZIP / object context (Pass-2b) -------------------------
    if cfg.include_zip_context:
        object_byte_sizes = _object_size_map(rows)
        for row in rows:
            key = (str(row["apk_id"]), str(row["object_id"]))
            obj_size = object_byte_sizes.get(key, 0)
            row["object_size_log2"] = _log2_floor(max(obj_size, 1))
            obj_type = str(row.get("object_type", ""))
            row["object_type_is_embedded_archive"] = (
                1.0 if obj_type == "embedded_archive" else 0.0
            )
            row["object_type_is_asset_blob"] = (
                1.0 if obj_type == "asset_blob" else 0.0
            )


def _bisect_left(sorted_list: List[int], target: int) -> int:
    """Stdlib-free bisect_left; avoids an import just for one call."""

    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
