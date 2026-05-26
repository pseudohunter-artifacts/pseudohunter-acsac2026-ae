"""Happer paired APK differential label processor (Module F).

Generates entry-level and region-level diff labels by comparing
a packed APK against its unpacked origin. These labels serve as
attention alignment targets during Stage 3 training.

From improved_packed_apk_framework.md §8 Stage 3:
    Diff dimensions: added_entry, size_delta, entropy_delta,
    compression_ratio_delta, byte_histogram_distance, etc.

Output:
    DiffResult with per-entry and per-region diff scores [0.0-1.0]
    where 1.0 = definitely packer-injected, 0.0 = unchanged
"""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DiffResult",
    "compute_paired_diff",
    "align_happer_entries",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EntryDiff:
    """Diff information for a single entry in the packed APK."""

    name: str
    diff_score: float           # 0.0 = unchanged, 1.0 = new/packer-injected
    status: str                 # "new" | "modified" | "unchanged"
    size_delta: float           # normalized size change
    entropy_delta: float        # entropy difference
    histogram_distance: float   # L1 distance between byte histograms


@dataclass
class DiffResult:
    """Complete diff result for a paired APK."""

    origin_path: str
    packed_path: str
    entry_diffs: Dict[str, EntryDiff]      # packed_entry_name → diff info
    alignment: Dict[str, Optional[str]]    # packed_entry → matched origin_entry (or None)
    n_new: int = 0
    n_modified: int = 0
    n_unchanged: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_entropy(data: bytes) -> float:
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
    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1
    n = max(len(data), 1)
    return counts / n


def _histogram_l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """L1 distance between two normalized histograms. Range [0, 2]."""
    return float(np.abs(h1 - h2).sum())


# ---------------------------------------------------------------------------
# Entry alignment
# ---------------------------------------------------------------------------


def _parse_zip_entries(apk_path: Path) -> Dict[str, Dict]:
    """Parse ZIP entries, return {name: {size, crc, data, entropy, histogram}}."""
    entries = {}
    try:
        with zipfile.ZipFile(apk_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.file_size == 0:
                    continue
                try:
                    data = zf.read(info.filename)
                    entries[info.filename] = {
                        "size": len(data),
                        "compressed_size": info.compress_size,
                        "crc": info.CRC,
                        "data": data,
                        "entropy": _fast_entropy(data),
                        "histogram": _byte_histogram(data),
                    }
                except Exception:
                    continue
    except zipfile.BadZipFile:
        pass
    return entries


def align_happer_entries(
    origin_entries: Dict[str, Dict],
    packed_entries: Dict[str, Dict],
) -> Dict[str, Optional[str]]:
    """Align packed entries to their origin counterparts.

    Strategy:
    1. Exact name match (most entries keep their path)
    2. For unmatched packed entries: try matching by similar size + low histogram distance
    3. Remaining unmatched → None (new entry, packer-injected)

    Returns: {packed_entry_name: origin_entry_name or None}
    """
    alignment: Dict[str, Optional[str]] = {}
    used_origins = set()

    # Pass 1: exact name match
    for packed_name in packed_entries:
        if packed_name in origin_entries:
            alignment[packed_name] = packed_name
            used_origins.add(packed_name)

    # Pass 2: for unmatched, try fuzzy matching
    unmatched_packed = [n for n in packed_entries if n not in alignment]
    available_origins = {n: e for n, e in origin_entries.items() if n not in used_origins}

    for packed_name in unmatched_packed:
        packed_info = packed_entries[packed_name]
        best_match = None
        best_score = float("inf")

        for origin_name, origin_info in available_origins.items():
            # Size similarity (log ratio)
            size_ratio = abs(math.log2(max(packed_info["size"], 1)) -
                            math.log2(max(origin_info["size"], 1)))
            if size_ratio > 3.0:  # more than 8x size difference → skip
                continue

            # Histogram distance
            hist_dist = _histogram_l1_distance(
                packed_info["histogram"], origin_info["histogram"]
            )

            # Combined score (lower = better match)
            score = size_ratio + hist_dist * 2.0

            if score < best_score and score < 2.0:  # threshold for "plausible match"
                best_score = score
                best_match = origin_name

        if best_match is not None:
            alignment[packed_name] = best_match
            used_origins.add(best_match)
            del available_origins[best_match]
        else:
            alignment[packed_name] = None  # new entry (packer-injected)

    return alignment


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def compute_paired_diff(
    origin_path: Path,
    packed_path: Path,
) -> DiffResult:
    """Compute differential labels between paired origin and packed APK.

    For each entry in the packed APK:
    - "new" (score=1.0): not present in origin → packer-injected
    - "modified" (score=0.3-0.9): present but changed significantly
    - "unchanged" (score=0.0): same content (by CRC or histogram)

    Modified score is computed from:
        score = clamp(0.3 * size_factor + 0.3 * entropy_factor + 0.4 * hist_factor, 0.3, 0.9)
    """
    origin_entries = _parse_zip_entries(origin_path)
    packed_entries = _parse_zip_entries(packed_path)

    if not origin_entries or not packed_entries:
        # Fallback: all entries get 0.5 (uncertain)
        diffs = {}
        for name in packed_entries:
            diffs[name] = EntryDiff(
                name=name, diff_score=0.5, status="unknown",
                size_delta=0.0, entropy_delta=0.0, histogram_distance=0.0,
            )
        return DiffResult(
            origin_path=str(origin_path),
            packed_path=str(packed_path),
            entry_diffs=diffs,
            alignment={n: None for n in packed_entries},
        )

    # Align entries
    alignment = align_happer_entries(origin_entries, packed_entries)

    # Compute diffs
    entry_diffs: Dict[str, EntryDiff] = {}
    n_new, n_mod, n_unch = 0, 0, 0

    for packed_name, origin_name in alignment.items():
        packed_info = packed_entries[packed_name]

        if origin_name is None:
            # New entry — packer-injected
            entry_diffs[packed_name] = EntryDiff(
                name=packed_name,
                diff_score=1.0,
                status="new",
                size_delta=1.0,
                entropy_delta=packed_info["entropy"] / 8.0,
                histogram_distance=2.0,
            )
            n_new += 1
            continue

        origin_info = origin_entries[origin_name]

        # Check if unchanged (same CRC)
        if packed_info["crc"] == origin_info["crc"]:
            entry_diffs[packed_name] = EntryDiff(
                name=packed_name,
                diff_score=0.0,
                status="unchanged",
                size_delta=0.0,
                entropy_delta=0.0,
                histogram_distance=0.0,
            )
            n_unch += 1
            continue

        # Modified — compute diff magnitude
        # Size factor: how much did size change?
        size_ratio = packed_info["size"] / max(origin_info["size"], 1)
        size_factor = min(abs(math.log2(max(size_ratio, 0.01))) / 3.0, 1.0)

        # Entropy factor: how much did entropy change?
        entropy_delta = abs(packed_info["entropy"] - origin_info["entropy"])
        entropy_factor = min(entropy_delta / 4.0, 1.0)

        # Histogram factor: L1 distance (range 0-2, normalize to 0-1)
        hist_dist = _histogram_l1_distance(
            packed_info["histogram"], origin_info["histogram"]
        )
        hist_factor = min(hist_dist / 1.5, 1.0)

        # Combined score
        score = 0.3 * size_factor + 0.3 * entropy_factor + 0.4 * hist_factor
        score = max(0.1, min(0.9, score))  # clamp to (0.1, 0.9)

        entry_diffs[packed_name] = EntryDiff(
            name=packed_name,
            diff_score=score,
            status="modified",
            size_delta=size_factor,
            entropy_delta=entropy_delta,
            histogram_distance=hist_dist,
        )
        n_mod += 1

    return DiffResult(
        origin_path=str(origin_path),
        packed_path=str(packed_path),
        entry_diffs=entry_diffs,
        alignment=alignment,
        n_new=n_new,
        n_modified=n_mod,
        n_unchanged=n_unch,
    )


# ---------------------------------------------------------------------------
# Inject labels → DiffResult conversion
# ---------------------------------------------------------------------------

# Label → diff_score mapping for inject_labels.jsonl
_INJECT_LABEL_SCORES = {
    "hidden_executable_payload": 1.0,
    "packer_native_library": 1.0,
    "new_payload": 1.0,
    "benign_loader": 0.85,
    "loader_dex": 0.85,
    "resource_modification": 0.70,
    "manifest_modification": 0.60,
    "signature_modification": 0.50,
    "original_code": 0.0,
    "benign_library": 0.0,
    "unchanged": 0.0,
}


def parse_inject_labels(jsonl_path: "Path") -> Optional[DiffResult]:
    """Parse inject_labels.jsonl into DiffResult format.

    inject_labels.jsonl is produced by our packer instrumentation (s5/s6).
    It provides ground-truth per-entry labels about what the packer modified.

    Format:
        {"packer_name": "...", "entries": [{"object_path": "...", "label": "...", ...}]}

    Returns:
        DiffResult with diff_scores derived from label semantics, or None on error.
    """
    import json

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # May be multiple lines (JSONL) or single JSON
        lines = [l for l in content.split("\n") if l.strip()]
        if not lines:
            return None

        # Parse first (usually only) line
        record = json.loads(lines[0])
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None

    entries_data = record.get("entries", [])
    if not entries_data:
        return None

    entry_diffs: Dict[str, EntryDiff] = {}
    alignment: Dict[str, Optional[str]] = {}
    n_new, n_mod, n_unch = 0, 0, 0

    for entry in entries_data:
        obj_path = entry.get("object_path", "")
        label = entry.get("label", "unchanged")
        source_path = entry.get("source_object_path", None)

        # Map label → diff_score
        score = _INJECT_LABEL_SCORES.get(label, 0.5)  # default 0.5 for unknown labels

        # Determine status
        if score >= 0.9:
            status = "new"
            n_new += 1
        elif score > 0.0:
            status = "modified"
            n_mod += 1
        else:
            status = "unchanged"
            n_unch += 1

        entry_diffs[obj_path] = EntryDiff(
            name=obj_path,
            diff_score=score,
            status=status,
            size_delta=0.0,  # not available from inject labels
            entropy_delta=0.0,
            histogram_distance=score,  # proxy
        )
        alignment[obj_path] = source_path

    return DiffResult(
        origin_path="",  # not needed when using inject labels
        packed_path=str(jsonl_path).replace(".inject_labels.jsonl", ".apk"),
        entry_diffs=entry_diffs,
        alignment=alignment,
        n_new=n_new,
        n_modified=n_mod,
        n_unchanged=n_unch,
    )
