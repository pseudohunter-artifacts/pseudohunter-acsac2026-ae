"""Tests for ``android_packer.labeling.diff_adapter`` (Track B · B-b-3).

Covers the three policy modes (all-payload, no-payload, rule-based), all
five DiffReport branches, payload_sha256 computation with and without
packed_apk_path, and cross-validated schema compatibility with the
existing ``build_training_labels`` pipeline.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Dict

import pytest

from android_packer.labeling.alignment import build_training_labels
from android_packer.labeling.diff_adapter import (
    NEW_ENTRY_LOADER,
    NEW_ENTRY_PAYLOAD,
    NEW_ENTRY_UNKNOWN,
    AllNewEntriesArePayload,
    NewEntryPolicy,
    NewEntryRule,
    NoNewEntriesArePayload,
    RuleBasedNewEntryPolicy,
    diff_report_to_synthetic_labels,
    new_entry_rules_from_spec,
)
from android_packer.labeling.diff_alignment import (
    BRANCH_BYTE_MODIFIED,
    BRANCH_NEW_IN_PACKED,
    BRANCH_REMOVED,
    BRANCH_RENAMED,
    BRANCH_UNCHANGED,
    DiffRange,
    DiffReport,
    EntryMapping,
    align,
)
from android_packer.labeling.synthetic import HIDDEN_EXECUTABLE_PAYLOAD


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_apk(path: Path, entries: Dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _make_pair(
    tmp_path: Path, benign: Dict[str, bytes], packed: Dict[str, bytes]
) -> tuple[Path, Path]:
    return (
        _write_apk(tmp_path / "benign.apk", benign),
        _write_apk(tmp_path / "packed.apk", packed),
    )


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


def test_all_new_entries_are_payload_policy_returns_payload():
    policy = AllNewEntriesArePayload()
    entry = EntryMapping(
        packed_entry="assets/x.dat",
        benign_entry=None,
        branch=BRANCH_NEW_IN_PACKED,
        packed_size=10,
        benign_size=0,
        packed_sha256="a" * 64,
        benign_sha256=None,
        diff_ranges=(DiffRange(0, 10),),
        changed_bytes=10,
    )
    assert policy.classify(entry) == NEW_ENTRY_PAYLOAD


def test_no_new_entries_are_payload_policy_returns_loader():
    policy = NoNewEntriesArePayload()
    entry = EntryMapping(
        packed_entry="lib/libloader.so",
        benign_entry=None,
        branch=BRANCH_NEW_IN_PACKED,
        packed_size=10,
        benign_size=0,
        packed_sha256="a" * 64,
        benign_sha256=None,
        diff_ranges=(DiffRange(0, 10),),
        changed_bytes=10,
    )
    assert policy.classify(entry) == NEW_ENTRY_LOADER


def test_rule_based_policy_first_match_wins():
    rules = [
        NewEntryRule(r"^lib/.*/libjiagu.*\.so$", NEW_ENTRY_PAYLOAD),
        NewEntryRule(r"^lib/.*/libshell.*\.so$", NEW_ENTRY_LOADER),
    ]
    policy = RuleBasedNewEntryPolicy(rules)

    def _entry(name: str) -> EntryMapping:
        return EntryMapping(
            packed_entry=name,
            benign_entry=None,
            branch=BRANCH_NEW_IN_PACKED,
            packed_size=10,
            benign_size=0,
            packed_sha256="a" * 64,
            benign_sha256=None,
            diff_ranges=(DiffRange(0, 10),),
            changed_bytes=10,
        )

    assert policy.classify(_entry("lib/armeabi-v7a/libjiagu.so")) == NEW_ENTRY_PAYLOAD
    assert policy.classify(_entry("lib/arm64-v8a/libshellx.so")) == NEW_ENTRY_LOADER
    assert policy.classify(_entry("lib/x86_64/libunknown.so")) == NEW_ENTRY_UNKNOWN


def test_new_entry_rule_rejects_invalid_classification():
    with pytest.raises(ValueError, match="classification"):
        NewEntryRule(object_path_regex="^.*$", classification="typo")


# ---------------------------------------------------------------------------
# Branch coverage in the adapter
# ---------------------------------------------------------------------------


def test_byte_modified_entry_emits_one_label_per_range(tmp_path):
    base = bytearray(b"A" * 20_000)
    mod = bytearray(base)
    mod[100:110] = b"X" * 10
    mod[15_000:15_050] = b"Y" * 50
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": bytes(base)},
        {"classes.dex": bytes(mod)},
    )
    report = align(benign, packed)

    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="s1_cvvt",
        apk_id="packed-abc",
        source_apk_id="benign-abc",
        packed_apk_path=packed,
    )
    # Two modifications, two diff ranges -> two labels.
    assert len(labels) == 2
    for lbl in labels:
        assert lbl.apk_id == "packed-abc"
        assert lbl.object_path == "classes.dex"
        assert lbl.label == HIDDEN_EXECUTABLE_PAYLOAD
        assert lbl.transform_family == "packer_s1_cvvt"
        assert lbl.source_apk_id == "benign-abc"
        assert lbl.source_object_path == "classes.dex"


def test_new_in_packed_entry_emitted_only_under_payload_policy(tmp_path):
    # Pad the unchanged entries so that payload_ratio stays well under 0.95
    # and the report is NOT flagged as degenerate (which would cause the
    # adapter to return []).
    unchanged_padding = b"U" * 100_000
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": unchanged_padding},
        {
            "classes.dex": unchanged_padding,
            "assets/encrypted.dat": b"\xde\xad\xbe\xef" * 500,  # 2 KB
            "lib/armeabi-v7a/libshellx.so": b"loader native code",
        },
    )
    report = align(benign, packed)
    assert report.degenerate_flag is False

    # Rule policy: libshellx* = loader, assets/encrypted.dat = payload
    rules = [
        NewEntryRule(r"^lib/.*/libshell.*\.so$", NEW_ENTRY_LOADER),
        NewEntryRule(r"^assets/encrypted\.dat$", NEW_ENTRY_PAYLOAD),
    ]
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="cs1_360_jiagu",
        apk_id="apk1",
        source_apk_id="ben1",
        new_entry_policy=RuleBasedNewEntryPolicy(rules),
        packed_apk_path=packed,
    )
    # Only assets/encrypted.dat emits a payload label; libshellx is loader; classes.dex unchanged.
    assert len(labels) == 1
    assert labels[0].object_path == "assets/encrypted.dat"
    assert labels[0].offset_start == 0
    assert labels[0].offset_end == 2000
    assert labels[0].transform_family == "packer_cs1_360_jiagu"


def test_unchanged_renamed_removed_branches_emit_no_labels(tmp_path):
    payload = b"same bytes" * 20
    benign, packed = _make_pair(
        tmp_path,
        {
            "classes.dex": b"dex",
            "assets/identical.bin": payload,
            "res.bin": b"removed in packed",
        },
        {
            "classes.dex": b"dex",                 # unchanged
            "assets/identical-renamed.bin": payload,  # renamed
            # res.bin missing in packed -> branch=removed
        },
    )
    report = align(benign, packed)
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="s5_timscriptov",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
    )
    assert labels == []


def test_degenerate_report_returns_empty_by_default(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"A" * 8000},
        {"classes.dex": b"B" * 8000},  # every byte differs -> payload_ratio 1.0
    )
    report = align(benign, packed)
    assert report.degenerate_flag is True

    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="cs1_360_jiagu",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
    )
    assert labels == []


def test_degenerate_report_opt_in_still_emits_labels(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"A" * 8000},
        {"classes.dex": b"B" * 8000},
    )
    report = align(benign, packed)
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="cs1_360_jiagu",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
        include_when_degenerate=True,
    )
    assert len(labels) == 1
    assert labels[0].offset_start == 0
    assert labels[0].offset_end == 8000


def test_alignment_failed_always_returns_empty(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"orig"},
        {"shell.dex": b"different entry with no sha match"},
    )
    report = align(benign, packed)
    assert report.alignment_failed is True
    assert diff_report_to_synthetic_labels(
        report,
        packer_name="s1_cvvt",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
    ) == []


# ---------------------------------------------------------------------------
# payload_sha256 behaviour
# ---------------------------------------------------------------------------


def test_payload_sha256_is_range_slice_when_bytes_available(tmp_path):
    benign_bytes = b"\x00" * 100
    packed_bytes = b"\x00" * 50 + b"\xff" * 50
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="s1_cvvt",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
    )
    assert len(labels) == 1
    lbl = labels[0]
    # sha should be exactly sha256(packed_bytes[lbl.offset_start:lbl.offset_end])
    expected = hashlib.sha256(packed_bytes[lbl.offset_start : lbl.offset_end]).hexdigest()
    assert lbl.payload_sha256 == expected


def test_payload_sha256_falls_back_to_entry_sha_when_path_missing(tmp_path):
    padding = b"U" * 100_000
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"abc", "large.bin": padding},
        {"classes.dex": b"xyz", "large.bin": padding},
    )
    report = align(benign, packed)
    assert report.degenerate_flag is False
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="s1_cvvt",
        apk_id="apk",
        source_apk_id="ben",
        # no packed_apk_path; adapter falls back to EntryMapping.packed_sha256
    )
    # classes.dex is byte_modified (all 3 bytes differ) -> one label.
    dex_labels = [l for l in labels if l.object_path == "classes.dex"]
    assert len(dex_labels) == 1
    # Fallback: sha must match the whole-entry sha from the DiffReport.
    report_entry = next(e for e in report.entries if e.packed_entry == "classes.dex")
    assert dex_labels[0].payload_sha256 == report_entry.packed_sha256


def test_missing_packed_apk_path_raises_when_file_not_found(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"a"},
        {"classes.dex": b"b"},
    )
    report = align(benign, packed)
    with pytest.raises(FileNotFoundError):
        diff_report_to_synthetic_labels(
            report,
            packer_name="s1_cvvt",
            apk_id="apk",
            source_apk_id="ben",
            packed_apk_path=tmp_path / "does_not_exist.apk",
        )


# ---------------------------------------------------------------------------
# Schema parity with Track A pipeline
# ---------------------------------------------------------------------------


def test_labels_consumed_by_build_training_labels_without_error(tmp_path):
    """End-to-end: DiffReport -> SyntheticLabel -> build_training_labels."""
    benign_bytes = b"A" * 1000
    packed_bytes = b"A" * 500 + b"B" * 500
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": benign_bytes},
        {"classes.dex": packed_bytes},
    )
    report = align(benign, packed)
    synth = diff_report_to_synthetic_labels(
        report,
        packer_name="s1_cvvt",
        apk_id="packed-apk",
        source_apk_id="benign-apk",
        packed_apk_path=packed,
    )
    assert len(synth) == 1
    # Build a minimal region list that references the same object/offsets
    # so build_training_labels has something to join against.
    regions = [
        {
            "apk_id": "packed-apk",
            "object_id": "obj-classes-dex",
            "region_id": "r1",
            "object_path": "classes.dex",
            "object_type": "dex",
            "offset_start": 500,
            "offset_end": 1000,
            "size": 500,
            "sha256": "c" * 64,
            "entropy": 7.8,
            "printable_ratio": 0.1,
        }
    ]
    labels = build_training_labels(
        regions=regions,
        synthetic_labels=[l.to_dict() for l in synth],
    )
    # Smoke: the region should end up labelled payload.
    assert len(labels.region_labels) == 1
    region_label = labels.region_labels[0]
    label_value = getattr(region_label, "label", None)
    assert label_value == HIDDEN_EXECUTABLE_PAYLOAD


def test_transform_family_override_is_honoured(tmp_path):
    benign, packed = _make_pair(
        tmp_path,
        {"classes.dex": b"a" * 1000},
        {"classes.dex": b"b" * 1000},
    )
    report = align(benign, packed)
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name="custom",
        apk_id="apk",
        source_apk_id="ben",
        packed_apk_path=packed,
        transform_family="packer_cvvt_experimental",
        include_when_degenerate=True,  # this tiny APK degenerates
    )
    assert labels
    assert all(l.transform_family == "packer_cvvt_experimental" for l in labels)


# ---------------------------------------------------------------------------
# Bridge from commercial_rule_engine
# ---------------------------------------------------------------------------


def test_new_entry_rules_from_spec_classifies_native_stub_as_payload():
    """Regression for the B-g-1 adapter fix: native_stub == payload under
    the PackerGrind-informed stance."""
    from android_packer.labeling.commercial_rule_engine import load_rule_file

    rule_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "data"
        / "track_b_commercial_rules"
        / "cs1_360_jiagu.yaml"
    )
    if not rule_path.exists():  # pragma: no cover - defensive
        pytest.skip("cs1_360_jiagu.yaml not available")
    spec = load_rule_file(rule_path)
    rules = new_entry_rules_from_spec(spec)
    assert rules, "expected at least one new-entry rule from cs1_360_jiagu.yaml"

    # Find the lib/*/libjiagu*.so rule and confirm it maps to payload.
    native_rules = [
        r for r in rules if "libjiagu" in r.object_path_regex.lower()
    ]
    assert native_rules, "expected at least one libjiagu rule"
    assert all(r.classification == NEW_ENTRY_PAYLOAD for r in native_rules), (
        "libjiagu native stubs must be classified as payload under the "
        "2026-04-30 corrected stance"
    )
