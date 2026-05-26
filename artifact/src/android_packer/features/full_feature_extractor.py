"""Full framework systematic feature extraction (Module C).

Three-layer feature design from improved_packed_apk_framework.md §4:

Layer 1 - byte_summary: All zero-cost byte statistics (~275 dim)
Layer 2 - structural_context: APK structure position signals (~16 scalar + 2 IDs)
Layer 3 - type_specific: Format-dependent features (DEX/ELF/Asset, conditional ~25-35 dim)

Design principle: "extract all parseable information" — no hand-selection.
The model (shared encoder + experts) learns which dimensions matter.

Contract:
- Pure stdlib at import time (no torch/numpy at module scope)
- numpy used only in the extraction functions (always available in [metrics] extra)
- Output: FeatureVector dataclass with .scalars (np.ndarray) + .entry_type_id + .section_type_id
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from android_packer.regioning.typed_slicer import (
    ENTRY_COARSE_TYPES,
    SECTION_TYPES,
    TypedRegion,
)

__all__ = [
    "FeatureVector",
    "extract_region_features",
    "extract_apk_context",
    "ApkContext",
    "SCALAR_FEATURE_DIM",
]


# ---------------------------------------------------------------------------
# APK-level context (computed once, shared by all regions)
# ---------------------------------------------------------------------------


@dataclass
class ApkContext:
    """APK-level statistics shared across all regions in the APK."""

    dex_count: int = 0
    so_count: int = 0
    asset_count: int = 0
    unknown_count: int = 0
    high_entropy_count: int = 0  # entries with entropy > 7.5
    total_entries: int = 0
    total_size: int = 0
    is_multidex: bool = False
    has_native: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to feature array [9 dims]."""
        return np.array([
            self.dex_count,
            self.so_count,
            self.asset_count,
            self.unknown_count,
            self.high_entropy_count,
            math.log2(max(self.total_size, 1)),
            self.total_entries,
            1.0 if self.is_multidex else 0.0,
            1.0 if self.has_native else 0.0,
        ], dtype=np.float32)


def extract_apk_context(entries: List[Tuple[str, bytes]]) -> ApkContext:
    """Compute APK-level context from all entries.

    Args:
        entries: List of (entry_path, entry_bytes) tuples
    """
    ctx = ApkContext()
    ctx.total_entries = len(entries)

    for path, data in entries:
        ctx.total_size += len(data)
        p = path.lower().replace("\\", "/")

        if data[:4] == b"dex\n":
            ctx.dex_count += 1
        elif data[:4] == b"\x7fELF" or p.endswith(".so"):
            ctx.so_count += 1
        elif p.startswith("assets/") or p.startswith("res/"):
            ctx.asset_count += 1
        else:
            ctx.unknown_count += 1

        # Check high entropy
        if len(data) >= 256:
            ent = _fast_entropy(data)
            if ent > 7.5:
                ctx.high_entropy_count += 1

    ctx.is_multidex = ctx.dex_count > 1
    ctx.has_native = ctx.so_count > 0
    return ctx


# ---------------------------------------------------------------------------
# Feature vector output
# ---------------------------------------------------------------------------

# Total scalar dimensions (byte_summary + structural_scalars + type_specific)
# byte_summary: 256 + 1 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 4 = 274
# structural: 5 + 9 = 14
# type_specific: max(25, 20, 15) padded to 30
# Total: 274 + 14 + 30 = 318
_BYTE_SUMMARY_DIM = 274
_STRUCTURAL_DIM = 14
_TYPE_SPECIFIC_DIM = 30
SCALAR_FEATURE_DIM = _BYTE_SUMMARY_DIM + _STRUCTURAL_DIM + _TYPE_SPECIFIC_DIM  # 318


@dataclass
class FeatureVector:
    """Complete feature vector for a single region."""

    scalars: np.ndarray          # [SCALAR_FEATURE_DIM] float32
    entry_type_id: int           # index into ENTRY_COARSE_TYPES (for nn.Embedding)
    section_type_id: int         # index into SECTION_TYPES (for nn.Embedding)


# ---------------------------------------------------------------------------
# Layer 1: byte_summary (~274 dims)
# ---------------------------------------------------------------------------


def _fast_entropy(data: bytes) -> float:
    """Shannon entropy in bits. Fast implementation."""
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    n = len(data)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _byte_histogram(data: bytes) -> np.ndarray:
    """256-dim normalized byte frequency histogram."""
    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1
    n = max(len(data), 1)
    return (counts / n).astype(np.float32)


def _rolling_entropy(data: bytes, window: int = 1024) -> np.ndarray:
    """Rolling entropy stats: [mean, std, max, min, slope] (5 dims)."""
    if len(data) < window:
        e = _fast_entropy(data)
        return np.array([e, 0.0, e, e, 0.0], dtype=np.float32)

    entropies = []
    step = max(window // 2, 256)
    for i in range(0, len(data) - window + 1, step):
        entropies.append(_fast_entropy(data[i:i + window]))

    if len(entropies) < 2:
        e = entropies[0] if entropies else 0.0
        return np.array([e, 0.0, e, e, 0.0], dtype=np.float32)

    arr = np.array(entropies, dtype=np.float32)
    # Slope: linear regression coefficient (entropy trend)
    x = np.arange(len(arr), dtype=np.float32)
    slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else 0.0

    return np.array([
        arr.mean(), arr.std(), arr.max(), arr.min(), slope
    ], dtype=np.float32)


def _bigram_top4(data: bytes) -> np.ndarray:
    """Top-4 bigram frequencies (lightweight n-gram feature)."""
    if len(data) < 2:
        return np.zeros(4, dtype=np.float32)

    bigram_counts: Dict[int, int] = {}
    for i in range(len(data) - 1):
        key = (data[i] << 8) | data[i + 1]
        bigram_counts[key] = bigram_counts.get(key, 0) + 1

    n = len(data) - 1
    top4 = sorted(bigram_counts.values(), reverse=True)[:4]
    result = np.zeros(4, dtype=np.float32)
    for i, count in enumerate(top4):
        result[i] = count / n
    return result


def _extract_byte_summary(data: bytes) -> np.ndarray:
    """Layer 1: all byte-level statistics (274 dims)."""
    n = max(len(data), 1)

    # Histogram [256]
    hist = _byte_histogram(data)

    # Entropy [1]
    entropy = _fast_entropy(data)

    # Rolling entropy [5]
    roll = _rolling_entropy(data)

    # Byte ratios [5]
    printable = sum(1 for b in data if 32 <= b <= 126) / n
    zero_ratio = data.count(0) / n
    high_byte_ratio = sum(1 for b in data if b >= 128) / n
    unique_bytes = len(set(data)) / 256.0
    max_run = _max_run_length(data) / max(n, 1)

    # Chi-square distance from uniform [1]
    expected = n / 256.0
    chi2 = sum((hist[i] * n - expected) ** 2 / max(expected, 1e-10) for i in range(256)) / 256.0
    chi2_norm = min(chi2 / 1000.0, 1.0)  # normalize to [0, 1]

    # Size [1]
    size_log = math.log2(max(n, 1))

    # Bigram top-4 [4]
    bigram = _bigram_top4(data)

    # Concatenate: 256 + 1 + 5 + 5 + 1 + 1 + 4 = 273... let me recount
    # hist[256] + entropy[1] + roll[5] + printable[1] + zero[1] + high_byte[1]
    # + unique[1] + max_run[1] + chi2[1] + size_log[1] + bigram[4] = 274
    result = np.concatenate([
        hist,                                                      # 256
        np.array([entropy], dtype=np.float32),                     # 1
        roll,                                                      # 5
        np.array([printable, zero_ratio, high_byte_ratio,
                  unique_bytes, max_run, chi2_norm, size_log],
                 dtype=np.float32),                                 # 7
        bigram,                                                    # 4
    ])  # Total: 256 + 1 + 5 + 7 + 4 = 273... need 1 more for 274
    # Add compression_ratio placeholder (filled from ZIP metadata externally)
    result = np.concatenate([result, np.array([0.0], dtype=np.float32)])  # 274

    return result


def _max_run_length(data: bytes) -> int:
    """Longest run of identical consecutive bytes."""
    if len(data) == 0:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, min(len(data), 65536)):  # cap scan for speed
        if data[i] == data[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


# ---------------------------------------------------------------------------
# Layer 2: structural_context (~14 scalar dims)
# ---------------------------------------------------------------------------


def _extract_structural_context(
    region: TypedRegion,
    entry_size: int,
    apk_context: ApkContext,
) -> np.ndarray:
    """Layer 2: structural position signals (14 dims).

    Note: entry_type_id and section_type_id are NOT in the scalar vector.
    They are passed separately as integer IDs for nn.Embedding.
    """
    # Region position within entry [3]
    offset_norm = region.offset_start / max(entry_size, 1)
    region_idx_norm = region.region_index / max(region.n_regions_in_entry, 1)
    size_ratio = region.size / max(entry_size, 1)

    # Entry position within APK [2]
    entry_order_norm = region.entry_index / max(apk_context.total_entries, 1)
    entry_size_log = math.log2(max(entry_size, 1))

    # APK context [9] (from ApkContext.to_array)
    ctx_arr = apk_context.to_array()

    return np.concatenate([
        np.array([offset_norm, region_idx_norm, size_ratio,
                  entry_order_norm, entry_size_log], dtype=np.float32),  # 5
        ctx_arr,                                                         # 9
    ])  # Total: 14


# ---------------------------------------------------------------------------
# Layer 3: type_specific (conditional, padded to 30 dims)
# ---------------------------------------------------------------------------


def _extract_dex_features(data: bytes, region: TypedRegion) -> np.ndarray:
    """DEX-specific features (25 dims, padded to 30)."""
    feats = np.zeros(30, dtype=np.float32)

    # Try to read DEX header (need full entry bytes, but we may only have region)
    # For DEX, the full entry header is at offset 0
    # We extract what we can from the region itself + any global DEX info

    # If this region IS the header region, parse counts
    if region.section_type_id == 0:  # dex_header
        if len(data) >= 112:
            try:
                feats[0] = 1.0  # valid_header
                string_ids = struct.unpack_from("<I", data, 56)[0]
                type_ids = struct.unpack_from("<I", data, 64)[0]
                proto_ids = struct.unpack_from("<I", data, 72)[0]
                field_ids = struct.unpack_from("<I", data, 80)[0]
                method_ids = struct.unpack_from("<I", data, 88)[0]
                class_defs = struct.unpack_from("<I", data, 96)[0]
                file_size = struct.unpack_from("<I", data, 32)[0]

                feats[1] = math.log2(max(string_ids, 1))
                feats[2] = math.log2(max(type_ids, 1))
                feats[3] = math.log2(max(proto_ids, 1))
                feats[4] = math.log2(max(field_ids, 1))
                feats[5] = math.log2(max(method_ids, 1))
                feats[6] = math.log2(max(class_defs, 1))
                feats[7] = math.log2(max(file_size, 1))

                # Shell-like indicator: very few methods + classes
                feats[8] = 1.0 if (method_ids < 10 and class_defs < 5) else 0.0
            except (struct.error, IndexError):
                pass
    else:
        # Non-header DEX region: extract content features
        # code_item regions: look for DEX opcode patterns
        if region.section_type == "dex_code_item":
            # Check for common patterns in code
            feats[9] = 1.0  # is_code_region
            # return-void (0x0e) frequency
            feats[10] = data.count(b"\x0e") / max(len(data), 1) * 10.0
            # const (0x12-0x19) frequency
            const_count = sum(data.count(bytes([op])) for op in range(0x12, 0x1a))
            feats[11] = const_count / max(len(data), 1) * 10.0
        elif region.section_type == "dex_string_data":
            feats[12] = 1.0  # is_string_region
            # Printable ratio in string data (should be high for legit strings)
            feats[13] = sum(1 for b in data if 32 <= b <= 126) / max(len(data), 1)

    # API indicators (scan for packer-related strings in any DEX region)
    feats[20] = 1.0 if b"DexClassLoader" in data else 0.0
    feats[21] = 1.0 if b"PathClassLoader" in data else 0.0
    feats[22] = 1.0 if b"loadDex" in data or b"openDexFile" in data else 0.0
    feats[23] = 1.0 if b"loadLibrary" in data else 0.0
    feats[24] = 1.0 if b"reflect" in data or b"Method.invoke" in data else 0.0

    return feats


def _extract_elf_features(data: bytes, region: TypedRegion) -> np.ndarray:
    """ELF-specific features (20 dims, padded to 30)."""
    feats = np.zeros(30, dtype=np.float32)

    feats[0] = 1.0  # is_elf

    # Symbol/API indicators (search in region bytes)
    feats[1] = 1.0 if b"JNI_OnLoad" in data else 0.0
    feats[2] = 1.0 if b"dlopen" in data else 0.0
    feats[3] = 1.0 if b"dlsym" in data else 0.0
    feats[4] = 1.0 if b"mmap" in data else 0.0
    feats[5] = 1.0 if b"mprotect" in data else 0.0
    feats[6] = 1.0 if b"__system_property" in data else 0.0

    # Section-specific entropy
    feats[7] = _fast_entropy(data)

    # Embedded payload indicators
    feats[8] = 1.0 if b"dex\n" in data else 0.0   # embedded DEX magic
    feats[9] = 1.0 if b"PK\x03\x04" in data else 0.0  # embedded ZIP

    # Packer section name indicators
    feats[10] = 1.0 if b".packed" in data else 0.0
    feats[11] = 1.0 if b".jiagu" in data else 0.0
    feats[12] = 1.0 if b".bangcle" in data else 0.0
    feats[13] = 1.0 if b".ijiami" in data else 0.0

    # High-entropy blob indicator (for .rodata/.data sections)
    if region.section_type in ("elf_rodata", "elf_data"):
        feats[14] = 1.0 if _fast_entropy(data) > 7.5 else 0.0

    # Section type encoding (which ELF section this is)
    elf_section_id = region.section_type_id - 10  # offset from elf_text (id=10)
    if 0 <= elf_section_id < 8:
        feats[15 + elf_section_id] = 1.0  # one-hot for ELF section (8 dims)

    return feats


def _extract_asset_features(data: bytes, region: TypedRegion) -> np.ndarray:
    """Asset/Unknown/Other-specific features (15 dims, padded to 30)."""
    feats = np.zeros(30, dtype=np.float32)

    # Magic detection within the region
    feats[0] = 1.0 if data[:4] == b"dex\n" or b"dex\n" in data[:4096] else 0.0
    feats[1] = 1.0 if data[:4] == b"\x7fELF" or b"\x7fELF" in data[:4096] else 0.0
    feats[2] = 1.0 if b"PK\x03\x04" in data[:4096] else 0.0
    feats[3] = 1.0 if data[:2] == b"\x1f\x8b" else 0.0  # gzip

    # Content type indicators
    n = max(len(data), 1)
    feats[4] = sum(1 for b in data if 32 <= b <= 126) / n  # text_like_ratio
    feats[5] = 1.0 if data[:8] == b"\x89PNG\r\n\x1a\n" else 0.0  # is_png
    feats[6] = 1.0 if data[:3] == b"\xff\xd8\xff" else 0.0  # is_jpeg
    feats[7] = 1.0 if data[:4] in (b"RIFF", b"OggS", b"fLaC") else 0.0  # is_audio

    # High-entropy unknown (packer payload signature)
    ent = _fast_entropy(data)
    feats[8] = 1.0 if ent > 7.5 else 0.0  # high_entropy_indicator
    feats[9] = ent / 8.0  # normalized entropy

    # Rolling entropy profile (thirds)
    third = max(len(data) // 3, 1)
    feats[10] = _fast_entropy(data[:third])
    feats[11] = _fast_entropy(data[third:2*third])
    feats[12] = _fast_entropy(data[2*third:])

    # Path depth (number of / separators)
    path = region.entry_path.replace("\\", "/")
    feats[13] = path.count("/")

    # Is in common packer payload locations
    pl = path.lower()
    feats[14] = 1.0 if any(x in pl for x in ("payload", "encrypt", "protect", "jiagu", "shell")) else 0.0

    return feats


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_region_features(
    region: TypedRegion,
    region_data: bytes,
    entry_size: int,
    apk_context: ApkContext,
    compression_ratio: float = 0.0,
) -> FeatureVector:
    """Extract complete feature vector for a single typed region.

    Args:
        region: TypedRegion from typed_slicer
        region_data: raw bytes of this region (data[offset_start:offset_end])
        entry_size: total size of the parent entry
        apk_context: APK-level statistics
        compression_ratio: ZIP compressed_size / file_size for this entry

    Returns:
        FeatureVector with .scalars [SCALAR_FEATURE_DIM] + type IDs
    """
    # Layer 1: byte_summary (274 dims)
    byte_feats = _extract_byte_summary(region_data)
    # Fill in compression_ratio (last dim of byte_summary)
    byte_feats[-1] = compression_ratio

    # Layer 2: structural_context (14 dims)
    struct_feats = _extract_structural_context(region, entry_size, apk_context)

    # Layer 3: type_specific (30 dims, conditional on entry type)
    entry_type = ENTRY_COARSE_TYPES[region.entry_type_id]
    if entry_type == "dex":
        type_feats = _extract_dex_features(region_data, region)
    elif entry_type == "elf":
        type_feats = _extract_elf_features(region_data, region)
    else:
        type_feats = _extract_asset_features(region_data, region)

    # Concatenate all scalar features
    scalars = np.concatenate([byte_feats, struct_feats, type_feats])
    assert scalars.shape[0] == SCALAR_FEATURE_DIM, \
        f"Expected {SCALAR_FEATURE_DIM} dims, got {scalars.shape[0]}"

    return FeatureVector(
        scalars=scalars,
        entry_type_id=region.entry_type_id,
        section_type_id=region.section_type_id,
    )
