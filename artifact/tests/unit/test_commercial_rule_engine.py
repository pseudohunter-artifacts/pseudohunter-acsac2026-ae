"""Tests for ``android_packer.labeling.commercial_rule_engine``.

Covers:
  * YAML schema validation (required keys / regex / enum-fields)
  * End-to-end rule application on a synthetic ZIP that simulates a
    commercial-packer layout
  * Round-trip: rule-engine output is consumable by the existing Path A
    adapter (schema parity with open-source packers)
  * cs1_360_jiagu.yaml (the checked-in first rule file) parses and all
    four rules match plausible filenames
"""

from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest

from android_packer.labeling.commercial_rule_engine import (
    CommercialRuleSchemaError,
    apply_rules_to_apk,
    load_rule_file,
    run_commercial_rule_engine,
)
from android_packer.labeling.injected_packer_adapter import (
    load_synthetic_labels,
    parse_inject_labels,
)
from android_packer.labeling.synthetic import HIDDEN_EXECUTABLE_PAYLOAD


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_rule_file(tmp_path: Path) -> Path:
    content = textwrap.dedent(
        """\
        packer_id: cs_demo
        packer_version: "1.0"
        gen_level: Gen2
        references:
          - title: "Demo paper"
            venue: "arXiv 2026"
        rules:
          - rule_id: "demo_lib_so"
            match:
              object_path_regex: "^lib/.*/libdemo\\\\.so$"
            emit:
              label: hidden_executable_payload
              payload_kind: native_stub
              transform_family: packer_cs_demo
              offset_start: 0
              offset_end: __file_size__
        """
    )
    p = tmp_path / "cs_demo.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _make_fake_packed_apk(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    apk = tmp_path / "packed.apk"
    with zipfile.ZipFile(apk, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return apk


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_load_rule_file_happy_path(tmp_path):
    spec = load_rule_file(_write_minimal_rule_file(tmp_path))
    assert spec.packer_id == "cs_demo"
    assert len(spec.rules) == 1
    assert spec.rules[0].rule_id == "demo_lib_so"


def test_load_rule_file_requires_references(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "packer_id: x\npacker_version: '1'\ngen_level: Gen1\nreferences: []\nrules: [{}]\n",
        encoding="utf-8",
    )
    with pytest.raises(CommercialRuleSchemaError, match="references"):
        load_rule_file(p)


def test_load_rule_file_rejects_bad_regex(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            packer_id: x
            packer_version: "1"
            gen_level: Gen1
            references: [{title: t, venue: v}]
            rules:
              - rule_id: r
                match: {object_path_regex: "["}
                emit:
                  label: hidden_executable_payload
                  payload_kind: encrypted_dex
                  transform_family: packer_x
                  offset_start: 0
                  offset_end: 10
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommercialRuleSchemaError, match="invalid regex"):
        load_rule_file(p)


def test_load_rule_file_rejects_unknown_label(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            packer_id: x
            packer_version: "1"
            gen_level: Gen1
            references: [{title: t, venue: v}]
            rules:
              - rule_id: r
                match: {object_path_regex: ".*"}
                emit:
                  label: bogus_label
                  payload_kind: encrypted_dex
                  transform_family: packer_x
                  offset_start: 0
                  offset_end: 10
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommercialRuleSchemaError, match="emit.label"):
        load_rule_file(p)


def test_load_rule_file_transform_family_prefix(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            packer_id: x
            packer_version: "1"
            gen_level: Gen1
            references: [{title: t, venue: v}]
            rules:
              - rule_id: r
                match: {object_path_regex: ".*"}
                emit:
                  label: hidden_executable_payload
                  payload_kind: encrypted_dex
                  transform_family: not_starting_with_packer
                  offset_start: 0
                  offset_end: 10
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommercialRuleSchemaError, match="transform_family"):
        load_rule_file(p)


def test_load_rule_file_rejects_duplicate_rule_id(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            packer_id: x
            packer_version: "1"
            gen_level: Gen1
            references: [{title: t, venue: v}]
            rules:
              - rule_id: r1
                match: {object_path_regex: ".*"}
                emit: {label: hidden_executable_payload, payload_kind: encrypted_dex, transform_family: packer_x, offset_start: 0, offset_end: 10}
              - rule_id: r1
                match: {object_path_regex: ".*"}
                emit: {label: hidden_executable_payload, payload_kind: encrypted_dex, transform_family: packer_x, offset_start: 0, offset_end: 10}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommercialRuleSchemaError, match="duplicate rule_id"):
        load_rule_file(p)


# ---------------------------------------------------------------------------
# Engine: end-to-end apply
# ---------------------------------------------------------------------------


def test_apply_rules_matches_entries(tmp_path):
    spec = load_rule_file(_write_minimal_rule_file(tmp_path))
    apk = _make_fake_packed_apk(
        tmp_path,
        {
            "lib/arm64-v8a/libdemo.so": b"\x00" * 128,
            "classes.dex": b"\xde\xad\xbe\xef" * 64,
        },
    )
    result = apply_rules_to_apk(apk, spec, apk_id="p1", source_apk_id="b1")
    assert len(result.record.entries) == 1
    entry = result.record.entries[0]
    assert entry.object_path == "lib/arm64-v8a/libdemo.so"
    assert entry.offset_end == 128
    assert entry.label == HIDDEN_EXECUTABLE_PAYLOAD
    assert entry.payload_sha256 is not None
    assert result.matched_entries == ["demo_lib_so:lib/arm64-v8a/libdemo.so"]
    assert "classes.dex" in result.unmatched_entries


def test_apply_rules_emits_record_consumable_by_path_a_adapter(tmp_path):
    """Rule-engine output must flow through parse_inject_labels() unchanged."""
    spec_path = _write_minimal_rule_file(tmp_path)
    apk = _make_fake_packed_apk(
        tmp_path,
        {"lib/arm64-v8a/libdemo.so": b"\x01" * 64},
    )
    out_jsonl = tmp_path / "inject_labels.jsonl"
    summary = run_commercial_rule_engine(
        apk,
        spec_path,
        out_jsonl,
        apk_id="packed_sha_x",
        source_apk_id="benign_sha_y",
    )
    assert summary["entries_emitted"] == 1

    records = parse_inject_labels(out_jsonl)
    assert len(records) == 1
    assert records[0].label_source == "rule_based"
    assert records[0].entries[0].transform_family == "packer_cs_demo"

    labels = load_synthetic_labels(out_jsonl)
    assert len(labels) == 1
    assert labels[0].label == HIDDEN_EXECUTABLE_PAYLOAD


def test_apply_rules_no_hash_when_disabled(tmp_path):
    spec = load_rule_file(_write_minimal_rule_file(tmp_path))
    apk = _make_fake_packed_apk(tmp_path, {"lib/x86/libdemo.so": b"a" * 16})
    result = apply_rules_to_apk(
        apk, spec, apk_id="p", source_apk_id="b", compute_sha256=False
    )
    # payload_sha256 stays None; engine users should only use this for
    # rule-file iteration, not final label production.
    assert result.record.entries[0].payload_sha256 is None


def test_apply_rules_multi_rule_entry_hit(tmp_path):
    """A single entry can be tagged by multiple rules (rare but valid)."""
    content = textwrap.dedent(
        """\
        packer_id: cs_multi
        packer_version: "1"
        gen_level: Gen1
        references: [{title: t, venue: v}]
        rules:
          - rule_id: r_generic
            match: {object_path_regex: "^assets/.*"}
            emit: {label: hidden_executable_payload, payload_kind: encrypted_dex, transform_family: packer_cs_multi, offset_start: 0, offset_end: __file_size__}
          - rule_id: r_specific
            match: {object_path_regex: "^assets/libjiagu\\\\.so$"}
            emit: {label: hidden_executable_payload, payload_kind: native_stub, transform_family: packer_cs_multi, offset_start: 0, offset_end: __file_size__}
        """
    )
    p = tmp_path / "multi.yaml"
    p.write_text(content, encoding="utf-8")
    apk = _make_fake_packed_apk(tmp_path, {"assets/libjiagu.so": b"\0" * 32})
    result = apply_rules_to_apk(apk, load_rule_file(p), apk_id="p", source_apk_id="b")
    # Both rules fire for the same entry.
    assert len(result.record.entries) == 2
    kinds = {e.payload_kind for e in result.record.entries}
    assert kinds == {"encrypted_dex", "native_stub"}


# ---------------------------------------------------------------------------
# Checked-in cs1_360_jiagu.yaml
# ---------------------------------------------------------------------------


def test_cs1_360_jiagu_rule_file_parses():
    rule_path = _REPO_ROOT / "configs" / "data" / "track_b_commercial_rules" / "cs1_360_jiagu.yaml"
    spec = load_rule_file(rule_path)
    assert spec.packer_id == "cs1_360_jiagu"
    # Dual-era rules (v1 lib/<abi>/ + v2 assets/ + .jgapp marker + shim) -> >= 4
    assert len(spec.rules) >= 4
    assert len(spec.references) >= 1
    rule_ids = {r.rule_id for r in spec.rules}
    assert "360_v1_lock_libjiagu_lib_abi" in rule_ids
    assert "360_v2_assets_libjiagu_abi_family" in rule_ids
    assert "360_shell_classes_dex" in rule_ids


def test_cs1_360_jiagu_regex_matches_typical_paths(tmp_path):
    """The three primary 360 regexes match the paths PackerGrind documents."""
    rule_path = _REPO_ROOT / "configs" / "data" / "track_b_commercial_rules" / "cs1_360_jiagu.yaml"
    spec = load_rule_file(rule_path)
    apk = _make_fake_packed_apk(
        tmp_path,
        {
            "lib/arm64-v8a/libjiagu_a64.so": b"\x01" * 128,
            "lib/armeabi-v7a/libjiagu.so": b"\x02" * 128,
            "assets/libjiagu_a64.so": b"\x03" * 256,
            "classes.dex": b"\x04" * 64,  # shell loader
            "AndroidManifest.xml": b"<manifest/>",  # should stay unmatched
        },
    )
    result = apply_rules_to_apk(apk, spec, apk_id="p", source_apk_id="b")

    object_paths = [e.object_path for e in result.record.entries]
    assert "lib/arm64-v8a/libjiagu_a64.so" in object_paths
    assert "lib/armeabi-v7a/libjiagu.so" in object_paths
    assert "assets/libjiagu_a64.so" in object_paths
    assert "classes.dex" in object_paths  # matched by shell rule
    assert "AndroidManifest.xml" in result.unmatched_entries

    # Loader classes.dex must be labeled benign_loader, not payload.
    loader_entries = [e for e in result.record.entries if e.object_path == "classes.dex"]
    assert len(loader_entries) == 1
    assert loader_entries[0].label == "benign_loader"


def test_cs1_benign_loader_does_not_produce_synthetic_label(tmp_path):
    """benign_loader regions must NOT become SyntheticLabel (positive class purity)."""
    rule_path = _REPO_ROOT / "configs" / "data" / "track_b_commercial_rules" / "cs1_360_jiagu.yaml"
    apk = _make_fake_packed_apk(
        tmp_path,
        {
            "classes.dex": b"\x00" * 64,  # only the loader; no payloads
        },
    )
    out = tmp_path / "labels.jsonl"
    summary = run_commercial_rule_engine(
        apk, rule_path, out, apk_id="p", source_apk_id="b"
    )
    assert summary["entries_emitted"] == 1  # loader recorded in JSONL
    labels = load_synthetic_labels(out)
    # but zero SyntheticLabel positives because the only entry is benign_loader
    assert labels == []
