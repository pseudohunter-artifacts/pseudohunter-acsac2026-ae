"""Track B Path B -- diff-based byte-level alignment of (benign, packed) APK pairs.

This is the **secondary ground-truth pathway** for Track B labels; it produces
payload byte intervals by comparing the bytes of a packed APK against the
benign APK it was derived from. Primary pathway is source injection
(`injected_packer_adapter`, Path A); the two are cross-validated per the
``cross_validate`` helper in that module.

See ``docs/workstreams/track_b/diff_alignment_spec.md`` for the full spec --
in particular:

* §2 three-stage algorithm: ZIP structural alignment -> entry classification
  (unchanged / new / byte-modified) -> byte-level diff
* §4 edge cases: renamed entries, 1-to-N merges, N-to-1 splits, fully
  rewritten ZIPs, whole-entry XOR
* §5 degenerate packer detection (payload_ratio > 0.95 flags
  ``needs_manual_review``)

This module deliberately stops short of producing ``SyntheticLabel`` objects.
The output is a neutral :class:`DiffReport` that a downstream converter
(B-b-2 ticket, to be added in a later commit) will consume along with
packer-specific metadata (``packer_name``, ``apk_id``, ``source_apk_id``) to
emit ``SyntheticLabel``. Separating these concerns lets the diff algorithm
and the labeling adapter evolve independently (they were started by two
parallel agents per ``tasks.md`` §3 parallelism plan).
"""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Block size for byte-level pre-scan. Entries are split into ``BLOCK_SIZE``
# chunks; identical chunks are skipped without running SequenceMatcher. This
# keeps Stage 3 tractable for multi-megabyte entries that are mostly
# unchanged (common when the packer rewrites only the DEX header). The size
# is not sensitive -- 4 KB matches the typical page size and strikes a
# balance between chunk-hash overhead and diff granularity.
BLOCK_SIZE = 4096

# Degenerate packer threshold; see spec §5.
DEGENERATE_PAYLOAD_RATIO_THRESHOLD = 0.95

# Branch codes stored in ``EntryMapping.branch`` (spec §2 Stage 2).
BRANCH_UNCHANGED = "unchanged"          # (a) same entry name, identical bytes
BRANCH_NEW_IN_PACKED = "new_in_packed"  # (b) entry only exists in packed APK
BRANCH_BYTE_MODIFIED = "byte_modified"  # (c) same entry, different bytes
BRANCH_RENAMED = "renamed"              # entry renamed but bytes identical
BRANCH_REMOVED = "removed"              # entry dropped from benign by packer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DiffRange:
    """A contiguous byte interval inside a *packed* entry that differs from
    (or is absent from) the benign counterpart.

    ``start`` / ``end`` are **object-local** offsets inside the packed entry
    -- compatible with :class:`SyntheticLabel.offset_start` /
    :class:`SyntheticLabel.offset_end`. Right-open (``end`` is exclusive).
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"DiffRange invalid: start={self.start} end={self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclasses.dataclass(frozen=True)
class EntryMapping:
    """Per-ZIP-entry alignment result."""

    packed_entry: Optional[str]
    benign_entry: Optional[str]
    branch: str
    packed_size: int
    benign_size: int
    packed_sha256: Optional[str]
    benign_sha256: Optional[str]
    diff_ranges: Tuple[DiffRange, ...]
    changed_bytes: int  # total bytes in diff_ranges (for payload_ratio calc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packed_entry": self.packed_entry,
            "benign_entry": self.benign_entry,
            "branch": self.branch,
            "packed_size": self.packed_size,
            "benign_size": self.benign_size,
            "packed_sha256": self.packed_sha256,
            "benign_sha256": self.benign_sha256,
            "diff_ranges": [r.to_dict() for r in self.diff_ranges],
            "changed_bytes": self.changed_bytes,
        }


@dataclasses.dataclass(frozen=True)
class DiffReport:
    """Top-level alignment result consumed by the adapter stage (B-b-2)."""

    benign_apk: str
    packed_apk: str
    benign_sha256: str
    packed_sha256: str
    entries: Tuple[EntryMapping, ...]
    payload_ratio: float
    degenerate_flag: bool
    alignment_failed: bool
    total_packed_bytes: int
    total_changed_bytes: int
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benign_apk": self.benign_apk,
            "packed_apk": self.packed_apk,
            "benign_sha256": self.benign_sha256,
            "packed_sha256": self.packed_sha256,
            "entries": [e.to_dict() for e in self.entries],
            "payload_ratio": self.payload_ratio,
            "degenerate_flag": self.degenerate_flag,
            "alignment_failed": self.alignment_failed,
            "total_packed_bytes": self.total_packed_bytes,
            "total_changed_bytes": self.total_changed_bytes,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align(benign_apk: Path, packed_apk: Path) -> DiffReport:
    """Align two APKs and return a neutral :class:`DiffReport`.

    See module docstring and ``diff_alignment_spec.md`` §2 for the algorithm.
    """
    benign_apk = Path(benign_apk)
    packed_apk = Path(packed_apk)
    benign_sha = _sha256_of_file(benign_apk)
    packed_sha = _sha256_of_file(packed_apk)

    benign_entries = _read_all_entries(benign_apk)
    packed_entries = _read_all_entries(packed_apk)

    notes: List[str] = []
    entry_mappings: List[EntryMapping] = []

    # Stage 1: Name-based alignment. Build a reverse index by content sha256
    # of benign entries so Stage 2 can detect renamed entries.
    benign_by_sha: Dict[str, List[str]] = {}
    for name, (data, sha) in benign_entries.items():
        benign_by_sha.setdefault(sha, []).append(name)

    matched_benign_entries: set[str] = set()

    # Stage 2: classify every packed entry.
    for packed_name, (packed_data, packed_entry_sha) in packed_entries.items():
        if packed_name in benign_entries:
            benign_data, benign_entry_sha = benign_entries[packed_name]
            matched_benign_entries.add(packed_name)
            if benign_entry_sha == packed_entry_sha:
                entry_mappings.append(
                    _make_unchanged_mapping(
                        packed_name, len(packed_data), packed_entry_sha
                    )
                )
            else:
                entry_mappings.append(
                    _make_byte_modified_mapping(
                        packed_name,
                        packed_data,
                        packed_entry_sha,
                        benign_data,
                        benign_entry_sha,
                    )
                )
        else:
            # Packed entry with no same-name counterpart in benign.
            # Try renamed detection via content sha256.
            renamed_matches = benign_by_sha.get(packed_entry_sha, [])
            renamed_matches = [n for n in renamed_matches if n not in matched_benign_entries]
            if renamed_matches:
                benign_name = renamed_matches[0]
                matched_benign_entries.add(benign_name)
                entry_mappings.append(
                    EntryMapping(
                        packed_entry=packed_name,
                        benign_entry=benign_name,
                        branch=BRANCH_RENAMED,
                        packed_size=len(packed_data),
                        benign_size=len(packed_data),
                        packed_sha256=packed_entry_sha,
                        benign_sha256=packed_entry_sha,
                        diff_ranges=(),
                        changed_bytes=0,
                    )
                )
            else:
                # Genuinely new entry in packed APK. We do NOT classify
                # payload vs loader here -- that's a packer-specific
                # decision left to the adapter (B-b-2) which has access to
                # the packer's labeling policy. We just report the full
                # range as potentially-changed.
                entry_mappings.append(
                    EntryMapping(
                        packed_entry=packed_name,
                        benign_entry=None,
                        branch=BRANCH_NEW_IN_PACKED,
                        packed_size=len(packed_data),
                        benign_size=0,
                        packed_sha256=packed_entry_sha,
                        benign_sha256=None,
                        diff_ranges=(DiffRange(0, len(packed_data)),) if packed_data else (),
                        changed_bytes=len(packed_data),
                    )
                )

    # Record benign entries that disappeared from the packed APK (packer
    # dropped them or merged into something else). They carry zero
    # changed_bytes from the packed-APK perspective but are useful for
    # spotcheck forensics.
    for benign_name, (benign_data, benign_entry_sha) in benign_entries.items():
        if benign_name in matched_benign_entries:
            continue
        entry_mappings.append(
            EntryMapping(
                packed_entry=None,
                benign_entry=benign_name,
                branch=BRANCH_REMOVED,
                packed_size=0,
                benign_size=len(benign_data),
                packed_sha256=None,
                benign_sha256=benign_entry_sha,
                diff_ranges=(),
                changed_bytes=0,
            )
        )

    # Degenerate detection (§5).
    total_packed_bytes = sum(m.packed_size for m in entry_mappings)
    total_changed_bytes = sum(m.changed_bytes for m in entry_mappings)
    payload_ratio = total_changed_bytes / total_packed_bytes if total_packed_bytes else 0.0
    degenerate = payload_ratio > DEGENERATE_PAYLOAD_RATIO_THRESHOLD
    if degenerate:
        notes.append(
            f"degenerate: payload_ratio={payload_ratio:.3f} exceeds "
            f"{DEGENERATE_PAYLOAD_RATIO_THRESHOLD:.2f}; "
            "alignment does not distinguish payload from benign."
        )

    # §4.4 full-rewrite detection: no entry mappings at all between the two
    # means either the ZIP is malformed or the packer rewrote everything
    # beyond recognition.
    alignment_failed = (
        not entry_mappings
        or (
            total_packed_bytes > 0
            and all(m.branch in {BRANCH_NEW_IN_PACKED, BRANCH_REMOVED} for m in entry_mappings)
            and not any(m.branch == BRANCH_UNCHANGED for m in entry_mappings)
            and not any(m.branch == BRANCH_RENAMED for m in entry_mappings)
        )
    )
    if alignment_failed:
        notes.append(
            "alignment_failed: no overlapping entries between benign and "
            "packed; packer likely rewrote ZIP structure beyond "
            "name-or-sha recognition."
        )

    return DiffReport(
        benign_apk=str(benign_apk),
        packed_apk=str(packed_apk),
        benign_sha256=benign_sha,
        packed_sha256=packed_sha,
        entries=tuple(entry_mappings),
        payload_ratio=payload_ratio,
        degenerate_flag=degenerate,
        alignment_failed=alignment_failed,
        total_packed_bytes=total_packed_bytes,
        total_changed_bytes=total_changed_bytes,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_all_entries(apk: Path) -> Dict[str, Tuple[bytes, str]]:
    """Return ``{entry_name: (bytes, sha256_hex)}``.

    Directory entries and empty names are skipped. This loads every entry
    into memory; the spec §3.3 budgets <30 MB APKs so that is acceptable.
    """
    entries: Dict[str, Tuple[bytes, str]] = {}
    with zipfile.ZipFile(apk) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename:
                continue
            with zf.open(info) as fh:
                data = fh.read()
            entries[info.filename] = (data, _sha256_of_bytes(data))
    return entries


def _make_unchanged_mapping(name: str, size: int, sha: str) -> EntryMapping:
    return EntryMapping(
        packed_entry=name,
        benign_entry=name,
        branch=BRANCH_UNCHANGED,
        packed_size=size,
        benign_size=size,
        packed_sha256=sha,
        benign_sha256=sha,
        diff_ranges=(),
        changed_bytes=0,
    )


def _make_byte_modified_mapping(
    name: str,
    packed_data: bytes,
    packed_sha: str,
    benign_data: bytes,
    benign_sha: str,
) -> EntryMapping:
    """Stage 3 byte diff for the (c) branch.

    Strategy: block-level pre-scan (O(n/BLOCK_SIZE) hashing) flags ranges
    where the two byte strings differ; we collect those ranges directly.
    Because any block-level difference marks that entire block's bytes as
    ``changed`` we err on the side of flagging a few extra bytes at the
    block boundaries. Downstream evaluation tolerates this because the
    payload is rarely smaller than a single 4 KB block.

    For the fully-XOR-ed case (every byte differs) we still produce a
    single contiguous range thanks to the merge pass at the end.
    """
    diff_ranges = _block_level_diff(packed_data, benign_data, BLOCK_SIZE)
    changed = sum(r.length for r in diff_ranges)
    return EntryMapping(
        packed_entry=name,
        benign_entry=name,
        branch=BRANCH_BYTE_MODIFIED,
        packed_size=len(packed_data),
        benign_size=len(benign_data),
        packed_sha256=packed_sha,
        benign_sha256=benign_sha,
        diff_ranges=tuple(diff_ranges),
        changed_bytes=changed,
    )


def _block_level_diff(packed: bytes, benign: bytes, block_size: int) -> List[DiffRange]:
    """Identify byte ranges in ``packed`` that differ from ``benign``.

    Emits maximally-merged, right-open ranges. Algorithm:

    1. **Block pre-scan** over the overlapping prefix: for every ``block_size``
       slice, if ``packed[slice] == benign[slice]`` skip it; otherwise
       refine the block's left and right byte boundaries to tighten the
       reported interval.
    2. **Tail append**: if ``packed`` is longer than ``benign``, every byte
       in the tail is by definition changed.
    3. **Merge** adjacent / overlapping intervals.

    When ``packed`` and ``benign`` are completely different (every block
    differs and every byte within differs -- e.g. whole-entry XOR), this
    collapses down to a single ``DiffRange(0, len(packed))``.
    """
    raw_intervals: List[Tuple[int, int]] = []
    overlap = min(len(packed), len(benign))

    offset = 0
    while offset < overlap:
        block_end = min(offset + block_size, overlap)
        if packed[offset:block_end] != benign[offset:block_end]:
            raw_intervals.extend(
                _refine_block_boundaries(packed, benign, offset, block_end)
            )
        offset = block_end

    # Tail append when packed is longer than benign.
    if len(packed) > overlap:
        raw_intervals.append((overlap, len(packed)))

    merged = _merge_intervals(raw_intervals)
    return [DiffRange(start=s, end=e) for s, e in merged]


def _refine_block_boundaries(
    packed: bytes, benign: bytes, start: int, end: int
) -> List[Tuple[int, int]]:
    """Refine which bytes inside ``[start, end)`` actually differ.

    Runs a simple left-and-right boundary search to tighten the range when
    only a small prefix/suffix of the block is changed (common for
    modified DEX headers). Returns a list of ``(sub_start, sub_end)``
    tuples inside ``[start, end)``; empty if nothing differs.
    """
    if packed[start:end] == benign[start:end]:
        return []
    # Tighten left boundary.
    left = start
    while left < end and left < len(benign) and packed[left] == benign[left]:
        left += 1
    if left >= end:
        return []
    # Tighten right boundary.
    right = end
    while right > left and right - 1 < len(benign) and packed[right - 1] == benign[right - 1]:
        right -= 1
    if right <= left:
        return []
    return [(left, right)]


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping / adjacent (end == next.start) intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged: List[Tuple[int, int]] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlap or touch
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "BLOCK_SIZE",
    "BRANCH_BYTE_MODIFIED",
    "BRANCH_NEW_IN_PACKED",
    "BRANCH_REMOVED",
    "BRANCH_RENAMED",
    "BRANCH_UNCHANGED",
    "DEGENERATE_PAYLOAD_RATIO_THRESHOLD",
    "DiffRange",
    "DiffReport",
    "EntryMapping",
    "align",
]
