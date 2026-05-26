"""Tests for ``android_packer.labeling.diff_alignment`` (Track B · Path B).

Covers the 5 unit-test categories mandated by
``docs/workstreams/track_b/diff_alignment_spec.md`` §6:

6.1 Branch coverage (unchanged / new / byte-modified / renamed)
6.2 Edge cases (whole-entry XOR, entry removal, size change)
6.3 Degenerate-packer detection
6.4 DiffReport schema stability
6.5 Performance (5 MB single-entry diff finishes fast; full-APK happy path)

The output is a neutral ``DiffReport``; the converter that turns it into
``SyntheticLabel`` is B-b-2's job and has its own test module.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Dict, Sequence

import pytest

from android_packer.labeling.diff_alignment import (
    BLOCK_SIZE,
    BRANCH_BYTE_MODIFIED,
    BRANCH_NEW_IN_PACKED,
    BRANCH_REMOVED,
    BRANCH_RENAMED,
    BRANCH_UNCHANGED,
    DEGENERATE_PAYLOAD_RATIO_THRESHOLD,
    DiffRange,
    DiffReport,
    EntryMapping,
    align,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_apk(path: Path, entries: Dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _make_pair(
    tmp_path: Path,
    benign_entries: Dict[str, bytes],
    packed_entries: Dict[str, bytes],
) -> tuple[Path, Path]:
    benign = _write_apk(tmp_path / "benign.apk", benign_entries)
    packed = _write_apk(tmp_path / "packed.apk", packed_entries)
    return benign, packed


def _find(report: DiffReport, packed_entry: str) -> EntryMapping:
    for m in report.entries:
        if m.packed_entry == packed_entry:
            return m
    raise AssertionError(f"no mapping for packed_entry={packed_entry!r}")


# ---------------------------------------------------------------------------
# 6.1 Branch coverage
# ---------------------------------------------------------------------------


def test_branch_unchanged_entry(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"dexmagic" + b"\x00" * 200},
        {"classes.dex": b"dexmagic" + b"\x00" * 200},
    )
    report = align(benign, packed)
    m = _find(report, "classes.dex")
    assert m.branch == BRANCH_UNCHANGED
    assert m.diff_ranges == ()
    assert m.changed_bytes == 0


def test_branch_new_in_packed_full_range(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"dex_original_bytes"},
        {
            "classes.dex": b"dex_original_bytes",
            "assets/encrypted.dat": b"\xde\xad\xbe\xef" * 50,
        },
    )
    report = align(benign, packed)
    m = _find(report, "assets/encrypted.dat")
    assert m.branch == BRANCH_NEW_IN_PACKED
    assert m.benign_entry is None
    assert len(m.diff_ranges) == 1
    assert m.diff_ranges[0].start == 0
    assert m.diff_ranges[0].end == 200
    assert m.changed_bytes == 200


def test_branch_byte_modified_prefix_only(tmp_path):
    """Header-only modification: only the first few bytes differ."""
    benign_bytes = b"dex\n035\x00" + bytes(range(100)) * 40  # 4 KB +
    packed_bytes = b"DEX!HACK" + benign_bytes[8:]
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    m = _find(report, "classes.dex")
    assert m.branch == BRANCH_BYTE_MODIFIED
    # Only the first 8 bytes differ; refinement should tighten to exactly that.
    assert len(m.diff_ranges) == 1
    assert m.diff_ranges[0].start == 0
    assert m.diff_ranges[0].end == 8
    assert m.changed_bytes == 8


def test_branch_byte_modified_multiple_ranges(tmp_path):
    """Two far-apart modifications produce two ranges."""
    base = bytearray(b"A" * 20_000)  # 20 KB
    mod = bytearray(base)
    mod[100:110] = b"X" * 10           # change in block 0
    mod[15_000:15_050] = b"Y" * 50     # change 3 blocks later
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": bytes(base)},
        {"classes.dex": bytes(mod)},
    )
    report = align(benign, packed)
    m = _find(report, "classes.dex")
    assert m.branch == BRANCH_BYTE_MODIFIED
    # Expect at least 2 distinct ranges because the gap between modifications
    # spans multiple identical blocks.
    assert len(m.diff_ranges) >= 2
    # Total changed is at least the 10 + 50 bytes of actual edits.
    assert m.changed_bytes >= 60


def test_branch_renamed_by_content_sha(tmp_path):
    """Entry renamed but identical bytes -> branch=renamed."""
    payload = b"exact same bytes" * 50
    benign, packed = _make_pair(
        tmp_path,
        {"assets/original.bin": payload},
        {"assets/renamed.bin": payload},
    )
    report = align(benign, packed)
    m = _find(report, "assets/renamed.bin")
    assert m.branch == BRANCH_RENAMED
    assert m.benign_entry == "assets/original.bin"
    assert m.changed_bytes == 0


def test_branch_removed_entries_reported(tmp_path):
    """Entry present in benign but dropped by packer -> branch=removed."""
    benign, packed = _make_pair(
        tmp_path,
        {
            "classes.dex": b"dex bytes",
            "resources.arsc": b"arsc bytes" * 30,
        },
        {"classes.dex": b"dex bytes"},
    )
    report = align(benign, packed)
    removed = [m for m in report.entries if m.branch == BRANCH_REMOVED]
    assert len(removed) == 1
    assert removed[0].benign_entry == "resources.arsc"
    assert removed[0].packed_entry is None


# ---------------------------------------------------------------------------
# 6.2 Edge cases
# ---------------------------------------------------------------------------


def test_whole_entry_xored_merges_to_single_range(tmp_path):
    benign_bytes = bytes(range(256)) * 40  # 10 KB
    # XOR every byte with 0xFF: all bytes differ, no identical block.
    packed_bytes = bytes(b ^ 0xFF for b in benign_bytes)
    benign, packed = _make_pair(
        tmp_path,
        {"assets/enc.dat": benign_bytes},
        {"assets/enc.dat": packed_bytes},
    )
    report = align(benign, packed)
    m = _find(report, "assets/enc.dat")
    assert m.branch == BRANCH_BYTE_MODIFIED
    # The whole-entry XOR must collapse into ONE diff range covering everything.
    assert len(m.diff_ranges) == 1
    assert m.diff_ranges[0].start == 0
    assert m.diff_ranges[0].end == len(benign_bytes)
    assert m.changed_bytes == len(benign_bytes)


def test_size_change_tail_is_marked_changed(tmp_path):
    benign_bytes = b"head" + b"\x00" * 1000
    packed_bytes = benign_bytes + b"\xff" * 500  # 500 extra bytes
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    m = _find(report, "classes.dex")
    assert m.branch == BRANCH_BYTE_MODIFIED
    # The tail [len(benign_bytes) .. len(packed_bytes)) must be in the diff.
    last_range = m.diff_ranges[-1]
    assert last_range.end == len(packed_bytes)
    assert last_range.start <= len(benign_bytes)


def test_renamed_and_byte_modified_coexist(tmp_path):
    benign_payload = b"classes" + b"\x00" * 2000
    modified_payload = b"CLASSES" + b"\x00" * 2000  # first 7 bytes differ
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_payload, "res.bin": b"resdata" * 20},
        {"classes.dex": modified_payload, "res-renamed.bin": b"resdata" * 20},
    )
    report = align(benign, packed)
    dex = _find(report, "classes.dex")
    assert dex.branch == BRANCH_BYTE_MODIFIED
    assert dex.changed_bytes == 7
    renamed = _find(report, "res-renamed.bin")
    assert renamed.branch == BRANCH_RENAMED
    assert renamed.benign_entry == "res.bin"


# ---------------------------------------------------------------------------
# 6.3 Degenerate packer detection
# ---------------------------------------------------------------------------


def test_degenerate_flag_on_whole_apk_replacement(tmp_path):
    """Packer replaces every byte -> payload_ratio > 0.95 -> degenerate."""
    benign_bytes = b"A" * 8000
    packed_bytes = b"B" * 8000
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    assert report.degenerate_flag is True
    assert report.payload_ratio > DEGENERATE_PAYLOAD_RATIO_THRESHOLD
    assert any("degenerate" in n for n in report.notes)


def test_non_degenerate_small_header_modification(tmp_path):
    """Only first 8 bytes of a 50 KB entry differ -> payload_ratio ~ 0.00016."""
    benign_bytes = b"header00" + b"\x00" * 50_000
    packed_bytes = b"HEADER00" + b"\x00" * 50_000
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    assert report.degenerate_flag is False
    assert report.payload_ratio < 0.01


def test_alignment_failed_when_zero_overlap(tmp_path):
    """Packer rewrote ZIP: benign and packed share no entry name or content."""
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"original dex bytes"},
        {"proxy.dex": b"completely different content that has no sha match"},
    )
    report = align(benign, packed)
    assert report.alignment_failed is True
    assert any("alignment_failed" in n for n in report.notes)


# ---------------------------------------------------------------------------
# 6.4 Schema stability
# ---------------------------------------------------------------------------


def test_diff_range_rejects_negative_or_inverted():
    with pytest.raises(ValueError):
        DiffRange(start=-1, end=10)
    with pytest.raises(ValueError):
        DiffRange(start=10, end=5)


def test_diff_report_to_dict_is_json_safe(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"abc"},
        {"classes.dex": b"xyz"},
    )
    report = align(benign, packed)
    import json
    # Must round-trip through json without raising.
    dumped = json.dumps(report.to_dict())
    reloaded = json.loads(dumped)
    assert reloaded["benign_sha256"] == report.benign_sha256
    assert reloaded["packed_sha256"] == report.packed_sha256
    assert reloaded["entries"][0]["packed_entry"] == "classes.dex"


def test_report_sha256_matches_file_contents(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"abc"},
        {"classes.dex": b"abc"},
    )
    report = align(benign, packed)
    assert report.benign_sha256 == hashlib.sha256(benign.read_bytes()).hexdigest()
    assert report.packed_sha256 == hashlib.sha256(packed.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 6.5 Performance
# ---------------------------------------------------------------------------


def test_large_entry_whole_xor_is_fast(tmp_path):
    """5 MB whole-entry XOR must finish in well under 10 seconds.

    This is the pathological case for naive O(n^2) diffs; our block-level
    algorithm should handle it in milliseconds.
    """
    size = 5 * 1024 * 1024  # 5 MB
    benign_bytes = bytes(size)  # all zeros
    packed_bytes = bytes(0xFF for _ in range(size))
    benign, packed = _make_pair(
        tmp_path,
        {"assets/big.bin": benign_bytes},
        {"assets/big.bin": packed_bytes},
    )
    start = time.monotonic()
    report = align(benign, packed)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"align() took {elapsed:.2f}s on 5 MB whole-XOR"
    m = _find(report, "assets/big.bin")
    # 5 MB whole-XOR collapses to a single range.
    assert len(m.diff_ranges) == 1
    assert m.diff_ranges[0].end == size


def test_medium_apk_full_pipeline_fast(tmp_path):
    """Simulated 2 MB APK with 6 entries (mix of unchanged / changed / new)
    finishes in < 2 seconds."""
    mb = 1024 * 1024
    benign_entries = {
        "classes.dex": b"dex_header\x00\x00" + b"\x00" * (mb - 12),
        "classes2.dex": b"dex2\x00\x00\x00\x00" + b"\x11" * (mb - 8),
        "resources.arsc": b"arsc_header" + b"\x22" * 500,
        "assets/config.json": b'{"version": 1}',
        "lib/arm64-v8a/libnative.so": b"ELF\x7f" + b"\x33" * 4000,
        "AndroidManifest.xml": b"<manifest/>",
    }
    packed_entries = dict(benign_entries)
    # Mutate a small prefix of classes.dex to simulate packer stamping.
    packed_entries["classes.dex"] = b"DEX!HACK!!!!" + b"\x00" * (mb - 12)
    # Add an encrypted container.
    packed_entries["assets/enc.dat"] = b"\xde\xad" * (mb // 2)

    benign, packed = _make_pair(tmp_path, benign_entries, packed_entries)
    start = time.monotonic()
    report = align(benign, packed)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"align() took {elapsed:.2f}s on simulated APK"

    # Sanity checks on branch distribution.
    branches = {m.branch for m in report.entries}
    assert BRANCH_UNCHANGED in branches
    assert BRANCH_BYTE_MODIFIED in branches
    assert BRANCH_NEW_IN_PACKED in branches


# ---------------------------------------------------------------------------
# 6.6 Regression slots (reserved for future spotcheck findings)
# ---------------------------------------------------------------------------


def test_empty_apks_align_without_error(tmp_path):
    """Regression: zero-entry APK pair must return a valid (non-crashing) report."""
    benign = _write_apk(tmp_path / "benign.apk", {})
    packed = _write_apk(tmp_path / "packed.apk", {})
    report = align(benign, packed)
    assert report.entries == ()
    # alignment_failed semantics: no entries at all -> True (degenerate edge).
    assert report.alignment_failed is True


def test_only_directory_entries_are_skipped(tmp_path):
    """APKs with only directory entries should be treated as empty."""
    benign = tmp_path / "benign.apk"
    with zipfile.ZipFile(benign, "w") as zf:
        zf.writestr(zipfile.ZipInfo("assets/"), b"")
    packed = tmp_path / "packed.apk"
    with zipfile.ZipFile(packed, "w") as zf:
        zf.writestr(zipfile.ZipInfo("assets/"), b"")
    report = align(benign, packed)
    assert report.entries == ()


def test_changed_bytes_never_exceeds_packed_size(tmp_path):
    """Invariant: changed_bytes <= packed_size for every mapping."""
    benign, packed = _make_pair(
        tmp_path,
        {"a.bin": b"xxxxxxxxxxxx", "b.bin": b"yyyyyy"},
        {"a.bin": b"XXXXXXXXXXXX", "c.bin": b"totally new bytes"},
    )
    report = align(benign, packed)
    for m in report.entries:
        assert m.changed_bytes <= m.packed_size
