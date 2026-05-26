"""Tests for ``android_packer.labeling.injected_packer_adapter``.

Aligned with ``docs/workstreams/track_b/labeling_injection_spec.md`` section 7:
at least 15 tests covering schema parsing, conversion to SyntheticLabel,
cross-validation, write/read round-trip, and Track A schema compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_packer.labeling import build_training_labels
from android_packer.labeling.injected_packer_adapter import (
    CrossValidationResult,
    InjectLabelRecord,
    InjectLabelSchemaError,
    InjectedEntryRecord,
    cross_validate,
    load_synthetic_labels,
    parse_inject_labels,
    to_synthetic_labels,
    write_inject_labels,
)
from android_packer.labeling.synthetic import (
    HIDDEN_EXECUTABLE_PAYLOAD,
    SyntheticLabel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, *rows: dict) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return path


def _minimal_payload_entry(**overrides) -> dict:
    base = {
        "object_path": "assets/encrypted.dat",
        "offset_start": 0,
        "offset_end": 48236,
        "label": HIDDEN_EXECUTABLE_PAYLOAD,
        "payload_kind": "encrypted_dex",
        "transform_family": "packer_s1_cvvt_apkprotect",
        "source_object_path": "classes.dex",
        "source_offset_start": 0,
        "source_offset_end": 48236,
        "payload_sha256": "a" * 64,
        "injection_point": "encodeSingleDex",
    }
    base.update(overrides)
    return base


def _minimal_record(**overrides) -> dict:
    base = {
        "apk_id": "packed_sha256_xxx",
        "source_apk_id": "benign_sha256_yyy",
        "packer_name": "s1_cvvt_apkprotect",
        "packer_commit": "651e73eddb",
        "label_source": "source_injected",
        "timestamp_utc": "2026-04-30T16:42:51Z",
        "entries": [_minimal_payload_entry()],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 7.1 Schema parsing
# ---------------------------------------------------------------------------


def test_parse_jsonl_minimal(tmp_path):
    """One well-formed record parses to exactly one InjectLabelRecord."""
    path = _write_jsonl(tmp_path / "inject_labels.jsonl", _minimal_record())
    records = parse_inject_labels(path)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, InjectLabelRecord)
    assert record.packer_name == "s1_cvvt_apkprotect"
    assert record.label_source == "source_injected"
    assert len(record.entries) == 1
    entry = record.entries[0]
    assert isinstance(entry, InjectedEntryRecord)
    assert entry.object_path == "assets/encrypted.dat"
    assert entry.payload_kind == "encrypted_dex"


def test_parse_jsonl_multi_entries(tmp_path):
    """One record can carry payload + loader entries; both are parsed."""
    payload_entry = _minimal_payload_entry()
    loader_entry = _minimal_payload_entry(
        object_path="classes.dex",
        offset_start=0,
        offset_end=12000,
        label="benign_loader",
        payload_kind="shim",
        source_object_path=None,
        source_offset_start=None,
        source_offset_end=None,
    )
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[payload_entry, loader_entry]),
    )
    records = parse_inject_labels(path)
    assert len(records[0].entries) == 2
    assert records[0].entries[1].payload_kind == "shim"
    assert records[0].entries[1].label == "benign_loader"


def test_parse_jsonl_schema_missing_required(tmp_path):
    """Missing required field triggers InjectLabelSchemaError with context."""
    bad_entry = _minimal_payload_entry()
    del bad_entry["offset_end"]
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[bad_entry]),
    )
    with pytest.raises(InjectLabelSchemaError, match="offset_end"):
        parse_inject_labels(path)


def test_parse_jsonl_payload_kind_enum(tmp_path):
    """Unknown payload_kind is rejected with a helpful message."""
    bad = _minimal_payload_entry(payload_kind="totally_bogus_kind")
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[bad]),
    )
    with pytest.raises(InjectLabelSchemaError, match="unknown payload_kind"):
        parse_inject_labels(path)


def test_parse_jsonl_rejects_bad_range(tmp_path):
    """offset_end < offset_start is a semantic error."""
    bad = _minimal_payload_entry(offset_start=100, offset_end=50)
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[bad]),
    )
    with pytest.raises(InjectLabelSchemaError, match="offset_end.*offset_start"):
        parse_inject_labels(path)


def test_parse_jsonl_requires_packer_prefix(tmp_path):
    """transform_family must begin with ``packer_`` for Track B labels."""
    bad = _minimal_payload_entry(transform_family="xor")
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[bad]),
    )
    with pytest.raises(InjectLabelSchemaError, match="transform_family"):
        parse_inject_labels(path)


def test_parse_jsonl_invalid_json(tmp_path):
    """Non-JSON lines produce an error that references the line number."""
    path = tmp_path / "inject_labels.jsonl"
    path.write_text("{this is not json}\n", encoding="utf-8")
    with pytest.raises(InjectLabelSchemaError, match="invalid JSON"):
        parse_inject_labels(path)


def test_parse_jsonl_skips_blank_lines(tmp_path):
    """Blank lines are tolerated (common in hand-edited files)."""
    path = tmp_path / "inject_labels.jsonl"
    record = _minimal_record()
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(record) + "\n")
        fh.write("   \n")
    records = parse_inject_labels(path)
    assert len(records) == 1


def test_parse_jsonl_file_not_found(tmp_path):
    """Missing file surfaces a FileNotFoundError, not a schema error."""
    with pytest.raises(FileNotFoundError):
        parse_inject_labels(tmp_path / "does_not_exist.jsonl")


def test_parse_jsonl_rule_based_label_source(tmp_path):
    """Commercial rule engine uses label_source=rule_based; also valid."""
    record = _minimal_record(label_source="rule_based", packer_name="cs1_360_jiagu")
    path = _write_jsonl(tmp_path / "inject_labels.jsonl", record)
    records = parse_inject_labels(path)
    assert records[0].label_source == "rule_based"


def test_parse_jsonl_invalid_label_source(tmp_path):
    """Unknown label_source values are rejected."""
    record = _minimal_record(label_source="made_up_value")
    path = _write_jsonl(tmp_path / "inject_labels.jsonl", record)
    with pytest.raises(InjectLabelSchemaError, match="label_source"):
        parse_inject_labels(path)


# ---------------------------------------------------------------------------
# 7.2 Conversion to SyntheticLabel
# ---------------------------------------------------------------------------


def test_to_synthetic_labels_drops_loader_entries(tmp_path):
    """Loader regions must not become SyntheticLabel (Track A convention)."""
    payload = _minimal_payload_entry()
    loader = _minimal_payload_entry(
        object_path="classes.dex",
        label="benign_loader",
        payload_kind="shim",
        source_object_path=None,
        source_offset_start=None,
        source_offset_end=None,
    )
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[payload, loader]),
    )
    labels = load_synthetic_labels(path)
    assert len(labels) == 1
    assert labels[0].label == HIDDEN_EXECUTABLE_PAYLOAD
    assert labels[0].transform_family.startswith("packer_")


def test_to_synthetic_labels_fields_round_trip(tmp_path):
    """Essential fields flow through unchanged to SyntheticLabel."""
    entry = _minimal_payload_entry(
        object_path="assets/ciphered.dat",
        offset_start=16,
        offset_end=4096,
        source_object_path="classes2.dex",
        source_offset_start=0,
        source_offset_end=4080,
    )
    path = _write_jsonl(
        tmp_path / "inject_labels.jsonl",
        _minimal_record(entries=[entry]),
    )
    [label] = load_synthetic_labels(path)
    assert label.object_path == "assets/ciphered.dat"
    assert label.offset_start == 16
    assert label.offset_end == 4096
    assert label.source_object_path == "classes2.dex"
    assert label.source_offset_start == 0
    assert label.source_offset_end == 4080
    assert label.apk_id == "packed_sha256_xxx"
    assert label.source_apk_id == "benign_sha256_yyy"


def test_payload_without_sha256_is_error():
    """payload_sha256 is mandatory for payload entries (integrity anchor)."""
    entry = InjectedEntryRecord(
        object_path="assets/enc.dat",
        offset_start=0,
        offset_end=100,
        label=HIDDEN_EXECUTABLE_PAYLOAD,
        payload_kind="encrypted_dex",
        transform_family="packer_test",
        payload_sha256=None,
    )
    with pytest.raises(InjectLabelSchemaError, match="payload_sha256"):
        entry.to_synthetic_label(apk_id="a", source_apk_id="b")


# ---------------------------------------------------------------------------
# 7.3 Integration with build_training_labels (Track A schema parity)
# ---------------------------------------------------------------------------


def test_injected_labels_feed_build_training_labels():
    """SyntheticLabel produced by adapter flows through the Track A pipeline."""
    payload = _minimal_payload_entry(object_path="assets/enc.dat", offset_start=0, offset_end=100)
    record = InjectLabelRecord(
        apk_id="pkd",
        source_apk_id="bng",
        packer_name="s1",
        packer_commit=None,
        label_source="source_injected",
        timestamp_utc=None,
        entries=(
            InjectedEntryRecord(**{k: v for k, v in payload.items()}),
        ),
    )
    labels = to_synthetic_labels([record])
    regions = [
        {
            "apk_id": "pkd",
            "object_id": "pkd:001",
            "region_id": "pkd:001:0",
            "object_path": "assets/enc.dat",
            "object_type": "asset_blob",
            "offset_start": 0,
            "offset_end": 100,
            "size": 100,
            "sha256": "b" * 64,
            "entropy": 7.9,
            "printable_ratio": 0.05,
        }
    ]
    training = build_training_labels(
        regions,
        [lbl.to_dict() for lbl in labels],
        min_overlap_bytes=1,
        min_overlap_ratio=0.0,
    )
    assert len(training.region_labels) == 1
    assert training.region_labels[0].label == HIDDEN_EXECUTABLE_PAYLOAD
    assert training.region_labels[0].label_id == 1


# ---------------------------------------------------------------------------
# 7.4 Cross-validation (Path A vs Path B)
# ---------------------------------------------------------------------------


def _make_label(object_path: str, start: int, end: int) -> SyntheticLabel:
    return SyntheticLabel(
        apk_id="pkd",
        object_path=object_path,
        offset_start=start,
        offset_end=end,
        label=HIDDEN_EXECUTABLE_PAYLOAD,
        transform_family="packer_x",
        payload_sha256="c" * 64,
        source_apk_id="bng",
    )


def test_cross_validate_identical_ranges_is_solid():
    a = [_make_label("x", 0, 100)]
    b = [_make_label("x", 0, 100)]
    result = cross_validate(a, b)
    assert result.iou == 1.0
    assert result.verdict == "solid"
    assert result.is_solid is True


def test_cross_validate_partial_mismatch():
    """IoU in [0.5, 0.9) triggers partial_mismatch verdict."""
    # intersection=75 (from 25 to 100), union=100 (from 0 to 100), IoU=0.75
    a = [_make_label("x", 0, 100)]
    b = [_make_label("x", 25, 100)]
    result = cross_validate(a, b)
    assert abs(result.iou - 0.75) < 1e-6
    assert result.verdict == "partial_mismatch"


def test_cross_validate_no_overlap_is_low_confidence():
    a = [_make_label("x", 0, 100)]
    b = [_make_label("x", 500, 600)]
    result = cross_validate(a, b)
    assert result.iou == 0.0
    assert result.verdict == "low_confidence"
    assert result.needs_manual_review is True


def test_cross_validate_per_object_breakdown():
    """Per-object IoU helps pinpoint which entry disagrees."""
    a = [_make_label("a", 0, 100), _make_label("b", 0, 100)]
    b = [_make_label("a", 0, 100), _make_label("b", 0, 50)]
    result = cross_validate(a, b)
    assert result.per_object_iou["a"] == 1.0
    assert abs(result.per_object_iou["b"] - 0.5) < 1e-6


def test_cross_validate_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        cross_validate([], [], solid_threshold=0.5, review_threshold=0.8)


def test_cross_validate_merges_overlapping_ranges():
    """Overlapping A ranges should be merged before IoU (no double-counting)."""
    a = [_make_label("x", 0, 50), _make_label("x", 30, 100)]  # merges to [0, 100]
    b = [_make_label("x", 0, 100)]
    assert cross_validate(a, b).iou == 1.0


# ---------------------------------------------------------------------------
# 7.5 Write round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_round_trip(tmp_path):
    entry = InjectedEntryRecord(
        object_path="assets/enc.dat",
        offset_start=0,
        offset_end=48236,
        label=HIDDEN_EXECUTABLE_PAYLOAD,
        payload_kind="encrypted_dex",
        transform_family="packer_s5_timscriptov",
        source_object_path="classes.dex",
        source_offset_start=0,
        source_offset_end=48236,
        payload_sha256="d" * 64,
    )
    record = InjectLabelRecord(
        apk_id="pkd_abc",
        source_apk_id="bng_def",
        packer_name="s5_timscriptov_apkprotector",
        packer_commit="0c51f48f79",
        label_source="source_injected",
        timestamp_utc="2026-04-30T17:00:00Z",
        entries=(entry,),
    )
    out = tmp_path / "inject_labels.jsonl"
    write_inject_labels(out, [record])
    assert out.exists()
    loaded = parse_inject_labels(out)
    assert len(loaded) == 1
    assert loaded[0].apk_id == "pkd_abc"
    assert loaded[0].packer_commit == "0c51f48f79"
    assert loaded[0].entries[0].payload_sha256 == "d" * 64


def test_write_drops_none_fields(tmp_path):
    """Optional None fields must be absent from JSON, not written as null."""
    entry = InjectedEntryRecord(
        object_path="x",
        offset_start=0,
        offset_end=1,
        label=HIDDEN_EXECUTABLE_PAYLOAD,
        payload_kind="encrypted_dex",
        transform_family="packer_y",
        payload_sha256="e" * 64,
        # source_* / injection_point left as None
    )
    record = InjectLabelRecord(
        apk_id="p",
        source_apk_id="b",
        packer_name="y",
        packer_commit=None,  # should be absent
        label_source="source_injected",
        timestamp_utc=None,
        entries=(entry,),
    )
    out = tmp_path / "inject_labels.jsonl"
    write_inject_labels(out, [record])
    payload = json.loads(out.read_text(encoding="utf-8").strip())
    assert "packer_commit" not in payload
    assert "timestamp_utc" not in payload
    entry_payload = payload["entries"][0]
    assert "source_offset_start" not in entry_payload
    assert "injection_point" not in entry_payload
