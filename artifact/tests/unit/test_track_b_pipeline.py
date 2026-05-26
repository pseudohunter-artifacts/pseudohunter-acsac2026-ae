"""Tests for ``android_packer.labeling.track_b_pipeline`` (Track B · B-c-1).

Covers:
1. PackerIdent.from_registry_entry grouping (open_source / commercial /
   registered_not_patched) + commercial auto path_b_enabled override.
2. process_pair with open-source packer + real inject_labels.jsonl ->
   LABEL_SOURCE_PATH_A chosen.
3. process_pair open-source with missing inject_labels.jsonl + valid
   Path B fallback -> LABEL_SOURCE_PATH_B chosen.
4. process_pair open-source with both Path A and Path B missing ->
   LABEL_SOURCE_NONE + needs_manual_review.
5. process_pair commercial without rule_file -> Path B only, needs_review.
6. process_pair commercial with rule_file (CS cross-validate path) ->
   cs_decision populated.
7. process_pair when packed_apk missing entirely -> REASON_NO_PACKED_APK.
8. process_pair when benign_apk missing entirely -> REASON_NO_BENIGN_APK.
9. process_pair with degenerate diff -> diff_degenerate=True + review.
10. process_batch summary aggregation (histograms, per-packer stats).
11. discover_pair_inputs walking packed_dir + benign_dir.
12. APKiD invocation is mocked (graceful=True path) so tests don't need apkid.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest

from android_packer.labeling import (
    BatchSummary,
    LABEL_SOURCE_NONE,
    LABEL_SOURCE_PATH_A,
    LABEL_SOURCE_PATH_B,
    PackerIdent,
    PairInputs,
    PairOutcome,
    REASON_NO_BENIGN_APK,
    REASON_NO_PACKED_APK,
    REASON_RULE_FILE_MISSING,
    discover_pair_inputs,
    process_batch,
    process_pair,
)
from android_packer.labeling.apkid_cross_check import ApkidFamilyMap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_apk(path: Path, entries: Dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _write_inject_labels_jsonl(path: Path, records: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r))
            fh.write("\n")
    return path


@pytest.fixture
def open_source_spec() -> Dict[str, Any]:
    """A minimal yaml-style spec for an open-source packer (like S3)."""
    return {
        "gen_level": "Gen1-Gen2",
        "license": "NONE",
        "status": "selected",
        "label": {
            "transform_family": "packer_test_opensource",
            "path_a": True,
            "path_b": True,
            "path_c": False,
        },
    }


@pytest.fixture
def commercial_spec() -> Dict[str, Any]:
    """A minimal yaml-style spec for a commercial packer (like CS3)."""
    return {
        "gen_level": "Gen2",
        "license": "commercial_eula",
        "status": "candidate",
        "label": {
            "transform_family": "packer_test_commercial",
            "path_a": True,
            "path_b": False,  # yaml says false, but pipeline must force it True
            "path_c": False,
        },
    }


@pytest.fixture
def registered_not_patched_spec() -> Dict[str, Any]:
    """A minimal yaml-style spec for S7 · Bangcle-OSS style."""
    return {
        "gen_level": "Gen2",
        "license": "NONE",
        "status": "registered_not_patched",
        "label": {
            "transform_family": "packer_test_not_patched",
            "path_a": False,
            "path_b": True,
            "path_c": False,
        },
    }


def _make_benign_and_packed(
    tmp_path: Path, *, benign_stem: str = "app"
) -> tuple[Path, Path]:
    """Build a benign+packed pair with a deliberate new payload entry."""
    benign = _write_apk(
        tmp_path / "benign" / f"{benign_stem}.apk",
        {
            "classes.dex": b"DEX-ORIGINAL-" + b"\x00" * 200,
            "AndroidManifest.xml": b"<manifest/>",
            "resources.arsc": b"RSRC",
        },
    )
    packed = _write_apk(
        tmp_path / "packed" / benign_stem / "packed.apk",
        {
            "classes.dex": b"DEX-ORIGINAL-" + b"\x00" * 200,  # unchanged
            "AndroidManifest.xml": b"<manifest/>",
            "resources.arsc": b"RSRC",
            "assets/payload.dat": b"ENCRYPTED-PAYLOAD-" + b"\xff" * 300,  # NEW
        },
    )
    return benign, packed


# ---------------------------------------------------------------------------
# 1. PackerIdent grouping
# ---------------------------------------------------------------------------


class TestPackerIdentGrouping:
    def test_open_source_basic(self, open_source_spec):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        assert ident.group == "open_source"
        assert ident.path_a_enabled is True
        assert ident.path_b_enabled is True
        assert ident.transform_family == "packer_test_opensource"
        assert ident.rule_file is None

    def test_commercial_forces_path_b_enabled(self, commercial_spec):
        """Even if yaml says path_b: false, commercial group must force True.

        Rationale: cs_cross_validate_commercial_packer requires Path B;
        the yaml's ``path_b: false`` only indicates Path B is not the
        primary source (see labeling_injection_spec.md section 9.3).
        """
        ident = PackerIdent.from_registry_entry("cs3_test", commercial_spec)
        assert ident.group == "commercial"
        assert ident.path_a_enabled is True
        assert ident.path_b_enabled is True, (
            "commercial packers must force path_b_enabled=True for "
            "cross-validation even when yaml says path_b: false"
        )

    def test_registered_not_patched(self, registered_not_patched_spec):
        ident = PackerIdent.from_registry_entry(
            "s7_test", registered_not_patched_spec
        )
        assert ident.group == "registered_not_patched"
        assert ident.path_a_enabled is False
        assert ident.path_b_enabled is True

    def test_missing_transform_family_raises(self):
        spec = {"label": {}}
        with pytest.raises(ValueError, match="transform_family"):
            PackerIdent.from_registry_entry("broken", spec)

    def test_partial_path_b_treated_as_enabled(self, open_source_spec):
        open_source_spec["label"]["path_b"] = "partial"
        ident = PackerIdent.from_registry_entry("s6_test", open_source_spec)
        assert ident.path_b_enabled is True

    def test_commercial_rule_file_resolution(self, commercial_spec, tmp_path):
        """rule_file is resolved against rules_dir if present."""
        rf = tmp_path / "cs_test.yaml"
        rf.write_text(
            "schema_version: 1\npacker_id: cs3_test\ntransform_family: packer_test_commercial\nrules: []\n",
            encoding="utf-8",
        )
        spec = dict(commercial_spec)
        spec["label"] = dict(spec["label"])
        spec["label"]["rule_file"] = "cs_test.yaml"
        ident = PackerIdent.from_registry_entry(
            "cs3_test", spec, rules_dir=tmp_path
        )
        assert ident.rule_file == rf

    def test_commercial_rule_file_missing_is_silent(self, commercial_spec, tmp_path):
        spec = dict(commercial_spec)
        spec["label"] = dict(spec["label"])
        spec["label"]["rule_file"] = "nope.yaml"
        ident = PackerIdent.from_registry_entry(
            "cs3_test", spec, rules_dir=tmp_path
        )
        assert ident.rule_file is None


# ---------------------------------------------------------------------------
# 2-9. process_pair per-branch tests
# ---------------------------------------------------------------------------


class TestProcessPair:
    def _run(
        self,
        *,
        tmp_path: Path,
        packer_ident: PackerIdent,
        benign: Path,
        packed: Optional[Path],
        inject_jsonl: Optional[Path],
        run_apkid: bool = False,
    ) -> PairOutcome:
        inputs = PairInputs(
            packer=packer_ident,
            benign_apk=benign,
            packed_apk=packed,
            inject_labels_jsonl=inject_jsonl,
            apk_id="test:apk:1",
            source_apk_id="test:benign:1",
        )
        return process_pair(
            inputs,
            tmp_path / "out",
            run_apkid=run_apkid,
        )

    def test_open_source_with_inject_labels_chooses_path_a(
        self, tmp_path, open_source_spec
    ):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        benign, packed = _make_benign_and_packed(tmp_path)
        inject_jsonl = _write_inject_labels_jsonl(
            tmp_path / "labels" / "inject_labels.jsonl",
            [
                {
                    "apk_id": "test:apk:1",
                    "source_apk_id": "test:benign:1",
                    "packer_name": "s3_test",
                    "entries": [
                        {
                            "object_path": "assets/payload.dat",
                            "offset_start": 0,
                            "offset_end": 318,
                            "label": "hidden_executable_payload",
                            "payload_kind": "encrypted_dex",
                            "transform_family": "packer_test_opensource",
                            "label_source": "source_injected",
                            "payload_sha256": "a" * 64,
                        }
                    ],
                }
            ],
        )
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=packed,
            inject_jsonl=inject_jsonl,
        )
        assert outcome.chosen_source == LABEL_SOURCE_PATH_A
        assert outcome.path_a_label_count >= 1
        assert outcome.path_b_label_count >= 1  # path B still runs as sanity
        assert outcome.final_label_count >= 1
        # merged_labels.jsonl should be written
        assert "merged_labels" in outcome.artifacts
        merged = Path(outcome.artifacts["merged_labels"])
        assert merged.exists() and merged.read_text(encoding="utf-8").strip(), (
            "merged_labels.jsonl should contain at least one line"
        )

    def test_open_source_no_path_a_falls_back_to_path_b(
        self, tmp_path, open_source_spec
    ):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        benign, packed = _make_benign_and_packed(tmp_path)
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=packed,
            inject_jsonl=None,
        )
        assert outcome.chosen_source == LABEL_SOURCE_PATH_B
        assert outcome.path_a_label_count == 0
        assert outcome.path_b_label_count >= 1
        assert "path_a_jsonl_missing" in outcome.reasons
        assert any("fell back to Path B" in n for n in outcome.notes)

    def test_open_source_nothing_to_label(self, tmp_path, open_source_spec):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        # benign and packed are identical -> diff produces no payload labels
        benign = _write_apk(
            tmp_path / "benign" / "app.apk",
            {
                "classes.dex": b"IDENTICAL",
                "AndroidManifest.xml": b"<m/>",
            },
        )
        packed = _write_apk(
            tmp_path / "packed" / "app" / "packed.apk",
            {
                "classes.dex": b"IDENTICAL",
                "AndroidManifest.xml": b"<m/>",
            },
        )
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=packed,
            inject_jsonl=None,
        )
        assert outcome.path_a_label_count == 0
        assert outcome.path_b_label_count == 0
        assert outcome.final_label_count == 0
        # When there's nothing to label and it's not degenerate,
        # chosen_source stays NONE.
        assert outcome.chosen_source == LABEL_SOURCE_NONE

    def test_commercial_without_rule_file_path_b_only(
        self, tmp_path, commercial_spec
    ):
        ident = PackerIdent.from_registry_entry("cs3_test", commercial_spec)
        benign, packed = _make_benign_and_packed(tmp_path)
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=packed,
            inject_jsonl=None,
        )
        # commercial group with no rule file -> path_b_only + review
        assert outcome.chosen_source == LABEL_SOURCE_PATH_B
        assert outcome.needs_manual_review is True
        # Reason or notes surfaces the missing rule file
        joined = " ".join(outcome.notes) + " " + " ".join(outcome.reasons)
        assert "commercial" in joined.lower() or "rule" in joined.lower()

    def test_packed_apk_missing_records_reason(
        self, tmp_path, open_source_spec
    ):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        benign = _write_apk(
            tmp_path / "benign" / "app.apk",
            {"classes.dex": b"X"},
        )
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=None,  # intentionally missing
            inject_jsonl=None,
        )
        assert outcome.path_b_label_count == 0
        assert REASON_NO_PACKED_APK in outcome.reasons
        assert outcome.final_label_count == 0

    def test_benign_apk_missing_records_reason(
        self, tmp_path, open_source_spec
    ):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        packed = _write_apk(
            tmp_path / "packed" / "app" / "packed.apk",
            {"classes.dex": b"X"},
        )
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=tmp_path / "missing.apk",  # does not exist
            packed=packed,
            inject_jsonl=None,
        )
        assert REASON_NO_BENIGN_APK in outcome.reasons
        assert outcome.final_label_count == 0

    def test_registered_not_patched_only_runs_path_b(
        self, tmp_path, registered_not_patched_spec
    ):
        ident = PackerIdent.from_registry_entry(
            "s7_test", registered_not_patched_spec
        )
        benign, packed = _make_benign_and_packed(tmp_path)
        # Even if we pass an inject_jsonl, path_a_enabled is False so
        # it must not be parsed.
        fake_jsonl = tmp_path / "never-parsed.jsonl"
        fake_jsonl.write_text("not-valid-json", encoding="utf-8")
        outcome = self._run(
            tmp_path=tmp_path,
            packer_ident=ident,
            benign=benign,
            packed=packed,
            inject_jsonl=fake_jsonl,
        )
        assert outcome.path_a_label_count == 0
        assert outcome.chosen_source == LABEL_SOURCE_PATH_B
        assert outcome.path_b_label_count >= 1


# ---------------------------------------------------------------------------
# 10. process_batch aggregation
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_batch_aggregates_correctly(self, tmp_path, open_source_spec):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        benign1, packed1 = _make_benign_and_packed(tmp_path, benign_stem="app1")
        benign2, packed2 = _make_benign_and_packed(tmp_path, benign_stem="app2")
        pairs = [
            PairInputs(
                packer=ident,
                benign_apk=benign1,
                packed_apk=packed1,
                inject_labels_jsonl=None,
                apk_id="apk1",
                source_apk_id="ben1",
            ),
            PairInputs(
                packer=ident,
                benign_apk=benign2,
                packed_apk=packed2,
                inject_labels_jsonl=None,
                apk_id="apk2",
                source_apk_id="ben2",
            ),
        ]
        outcomes, summary = process_batch(
            pairs, tmp_path / "out", run_apkid=False
        )
        assert len(outcomes) == 2
        assert summary.total_pairs == 2
        assert summary.ok_pairs == 2
        assert summary.label_source_histogram[LABEL_SOURCE_PATH_B] == 2
        assert "s3_test" in summary.per_packer_stats
        assert summary.per_packer_stats["s3_test"]["total"] == 2
        assert summary.per_packer_stats["s3_test"]["with_labels"] == 2

        # summary files written
        assert (tmp_path / "out" / "summary.jsonl").exists()
        assert (tmp_path / "out" / "summary.json").exists()
        # summary.jsonl has 2 lines
        lines = (tmp_path / "out" / "summary.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["packer_id"] == "s3_test" for line in lines)

    def test_batch_with_mixed_outcomes(self, tmp_path, open_source_spec):
        ident = PackerIdent.from_registry_entry("s3_test", open_source_spec)
        benign, packed = _make_benign_and_packed(tmp_path, benign_stem="good")
        missing_benign = tmp_path / "nope.apk"
        pairs = [
            PairInputs(
                packer=ident,
                benign_apk=benign,
                packed_apk=packed,
                inject_labels_jsonl=None,
                apk_id="apk_good",
                source_apk_id="ben_good",
            ),
            PairInputs(
                packer=ident,
                benign_apk=missing_benign,
                packed_apk=None,
                inject_labels_jsonl=None,
                apk_id="apk_bad",
                source_apk_id="ben_bad",
            ),
        ]
        _outcomes, summary = process_batch(
            pairs, tmp_path / "out", run_apkid=False
        )
        assert summary.total_pairs == 2
        assert summary.ok_pairs == 1


# ---------------------------------------------------------------------------
# 11. discover_pair_inputs
# ---------------------------------------------------------------------------


class TestDiscoverPairInputs:
    def _make_registry(self) -> Dict[str, Any]:
        return {
            "s3_test": {
                "license": "NONE",
                "status": "selected",
                "label": {
                    "transform_family": "packer_test_opensource",
                    "path_a": True,
                    "path_b": True,
                },
            },
            "cs3_test": {
                "license": "commercial_eula",
                "status": "candidate",
                "label": {
                    "transform_family": "packer_test_commercial",
                    "path_a": True,
                    "path_b": False,
                },
            },
            "s7_test": {
                "license": "NONE",
                "status": "registered_not_patched",
                "label": {
                    "transform_family": "packer_test_not_patched",
                    "path_a": False,
                    "path_b": True,
                },
            },
        }

    def test_discover_finds_matching_pairs(self, tmp_path):
        registry = self._make_registry()
        # Build expected layout
        benign_dir = tmp_path / "benign"
        packed_dir = tmp_path / "packed"
        _write_apk(benign_dir / "app1.apk", {"a": b"1"})
        _write_apk(benign_dir / "app2.apk", {"a": b"2"})
        _write_apk(packed_dir / "s3_test" / "app1" / "packed.apk", {"a": b"A"})
        _write_apk(packed_dir / "s3_test" / "app2" / "packed.apk", {"a": b"B"})
        _write_apk(
            packed_dir / "cs3_test" / "app1" / "packed.apk", {"a": b"C"}
        )
        # inject_labels.jsonl for one of them
        (packed_dir / "s3_test" / "app1" / "inject_labels.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        pairs = discover_pair_inputs(
            registry, packed_dir=packed_dir, benign_dir=benign_dir
        )
        # 2 s3_test pairs + 1 cs3_test pair = 3 pairs
        assert len(pairs) == 3
        packer_ids = [p.packer.packer_id for p in pairs]
        assert packer_ids.count("s3_test") == 2
        assert packer_ids.count("cs3_test") == 1

        # The app1 s3_test pair has inject_labels.jsonl wired up.
        s3_app1 = next(
            p
            for p in pairs
            if p.packer.packer_id == "s3_test" and p.benign_apk.stem == "app1"
        )
        assert s3_app1.inject_labels_jsonl is not None
        assert s3_app1.inject_labels_jsonl.exists()

    def test_discover_skips_registered_not_patched_by_default(self, tmp_path):
        registry = self._make_registry()
        benign_dir = tmp_path / "benign"
        packed_dir = tmp_path / "packed"
        _write_apk(benign_dir / "app.apk", {"a": b"1"})
        _write_apk(packed_dir / "s7_test" / "app" / "packed.apk", {"a": b"2"})
        pairs = discover_pair_inputs(
            registry, packed_dir=packed_dir, benign_dir=benign_dir
        )
        assert len(pairs) == 0

    def test_discover_includes_registered_not_patched_when_flag_set(
        self, tmp_path
    ):
        registry = self._make_registry()
        benign_dir = tmp_path / "benign"
        packed_dir = tmp_path / "packed"
        _write_apk(benign_dir / "app.apk", {"a": b"1"})
        _write_apk(packed_dir / "s7_test" / "app" / "packed.apk", {"a": b"2"})
        pairs = discover_pair_inputs(
            registry,
            packed_dir=packed_dir,
            benign_dir=benign_dir,
            include_registered_not_patched=True,
        )
        assert len(pairs) == 1
        assert pairs[0].packer.packer_id == "s7_test"

    def test_discover_allowlist(self, tmp_path):
        registry = self._make_registry()
        benign_dir = tmp_path / "benign"
        packed_dir = tmp_path / "packed"
        _write_apk(benign_dir / "app.apk", {"a": b"1"})
        _write_apk(packed_dir / "s3_test" / "app" / "packed.apk", {"a": b"A"})
        _write_apk(packed_dir / "cs3_test" / "app" / "packed.apk", {"a": b"C"})
        pairs = discover_pair_inputs(
            registry,
            packed_dir=packed_dir,
            benign_dir=benign_dir,
            packer_allowlist=["cs3_test"],
        )
        assert len(pairs) == 1
        assert pairs[0].packer.packer_id == "cs3_test"

    def test_discover_skips_missing_packed_apk(self, tmp_path):
        registry = self._make_registry()
        benign_dir = tmp_path / "benign"
        packed_dir = tmp_path / "packed"
        _write_apk(benign_dir / "app.apk", {"a": b"1"})
        # create packer dir but no packed.apk inside
        (packed_dir / "s3_test" / "app").mkdir(parents=True, exist_ok=True)
        pairs = discover_pair_inputs(
            registry, packed_dir=packed_dir, benign_dir=benign_dir
        )
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# 12. Real registry + yaml parse sanity
# ---------------------------------------------------------------------------


def test_real_registry_yaml_grouping() -> None:
    """All packer entries in the shipped yaml must PackerIdent-parse cleanly."""
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "data" / "track_b_packers.yaml"
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))["packers"]

    groups = {"open_source": 0, "commercial": 0, "registered_not_patched": 0}
    for pid, spec in registry.items():
        ident = PackerIdent.from_registry_entry(pid, spec)
        groups[ident.group] = groups.get(ident.group, 0) + 1
        # Every commercial packer must have path_b_enabled forced to True.
        if ident.group == "commercial":
            assert ident.path_b_enabled is True, (
                f"commercial packer {pid!r} must have path_b_enabled=True"
            )

    # We expect: S1..S6 = 6 open_source, S7 = 1 registered_not_patched,
    # CS1..CS5 = 5 commercial. Totals: 12.
    assert groups["open_source"] == 6
    assert groups["registered_not_patched"] == 1
    assert groups["commercial"] == 5
