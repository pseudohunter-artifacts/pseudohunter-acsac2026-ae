"""Tests for ``android_packer.labeling.apkid_cross_check`` (Track B · B-g-4).

Covers:
1. parse_apkid_json OK / empty / malformed JSON / non-dict top-level /
   files not a list / matches not a dict / hits not a list
2. ApkidFamilyMap case-insensitive + substring fallback + None lookup
3. apkid_family_map_from_dict schema validation (non-dict, wrong
   schema_version, non-dict mappings, non-str key/value)
4. cross_check for every agreement value:
   - SOLID (expected matches detected)
   - MISMATCH (expected packer, detected different packer)
   - NO_APKID_DETECTION (expected packer, apkid found nothing)
   - NO_EXPECTATION (benign, apkid found nothing)
   - APKID_FALSE_POSITIVE (benign, apkid found packer)
   - APKID_FAILED (apkid run itself failed)
5. cross_check with unmapped hits (note but still SOLID if a mapped hit
   also happens to agree)
6. run_apkid graceful mode when CLI is missing
7. write_apkid_reports_jsonl round-trip + atomic replace
8. load_apkid_family_map against the real shipped YAML file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_packer.labeling.apkid_cross_check import (
    AGREEMENT_APKID_FAILED,
    AGREEMENT_APKID_FALSE_POSITIVE,
    AGREEMENT_MISMATCH,
    AGREEMENT_NO_APKID_DETECTION,
    AGREEMENT_NO_EXPECTATION,
    AGREEMENT_SOLID,
    APKID_CATEGORY_PACKER,
    APKID_CATEGORY_PROTECTOR,
    ApkidCrossCheckReport,
    ApkidError,
    ApkidFamilyMap,
    ApkidFamilyMapError,
    ApkidMatch,
    ApkidResult,
    apkid_family_map_from_dict,
    cross_check,
    cross_check_apk,
    load_apkid_family_map,
    parse_apkid_json,
    run_apkid,
    write_apkid_reports_jsonl,
)


# ---------------------------------------------------------------------------
# parse_apkid_json
# ---------------------------------------------------------------------------


def test_parse_apkid_json_ok() -> None:
    payload = {
        "apkid_version": "3.1.0",
        "rules_sha256": "deadbeef",
        "files": [
            {
                "filename": "a.apk",
                "matches": {
                    "packer": ["Bangcle"],
                    "compiler": ["r8"],
                },
            },
            {
                "filename": "a.apk!classes.dex",
                "matches": {
                    "anti_vm": ["Build.FINGERPRINT check"],
                },
            },
        ],
    }
    r = parse_apkid_json(json.dumps(payload))
    assert r.apkid_version == "3.1.0"
    assert r.rules_sha256 == "deadbeef"
    # 1 packer + 1 compiler + 1 anti_vm = 3 matches
    assert len(r.matches) == 3
    assert any(m.is_packer_like() for m in r.matches)
    packer_like = r.packer_like_matches()
    assert len(packer_like) == 1
    assert packer_like[0].category == APKID_CATEGORY_PACKER
    assert packer_like[0].hit == "Bangcle"
    # raw_json is retained verbatim
    assert "Bangcle" in r.raw_json


def test_parse_apkid_json_empty_raises() -> None:
    with pytest.raises(ApkidError, match="empty"):
        parse_apkid_json("")
    with pytest.raises(ApkidError, match="empty"):
        parse_apkid_json("   \n\t  ")


def test_parse_apkid_json_malformed() -> None:
    with pytest.raises(ApkidError, match="not valid JSON"):
        parse_apkid_json("{not valid")


def test_parse_apkid_json_non_dict_top_level() -> None:
    with pytest.raises(ApkidError, match="not an object"):
        parse_apkid_json("[1, 2, 3]")


def test_parse_apkid_json_files_not_list() -> None:
    with pytest.raises(ApkidError, match="'files' is not a list"):
        parse_apkid_json(json.dumps({"files": {"a": "b"}}))


def test_parse_apkid_json_file_missing_filename() -> None:
    with pytest.raises(ApkidError, match="empty filename"):
        parse_apkid_json(json.dumps({"files": [{"matches": {"packer": ["x"]}}]}))


def test_parse_apkid_json_matches_not_dict() -> None:
    with pytest.raises(ApkidError, match="matches is not an object"):
        parse_apkid_json(
            json.dumps({"files": [{"filename": "a.apk", "matches": ["nope"]}]})
        )


def test_parse_apkid_json_hits_not_list() -> None:
    with pytest.raises(ApkidError, match="is not a list"):
        parse_apkid_json(
            json.dumps(
                {"files": [{"filename": "a.apk", "matches": {"packer": "nope"}}]}
            )
        )


def test_parse_apkid_json_empty_matches() -> None:
    r = parse_apkid_json(
        json.dumps(
            {
                "apkid_version": "3.1.0",
                "rules_sha256": "xx",
                "files": [{"filename": "a.apk", "matches": {}}],
            }
        )
    )
    assert r.matches == ()
    assert r.packer_like_matches() == ()


# ---------------------------------------------------------------------------
# ApkidFamilyMap
# ---------------------------------------------------------------------------


def test_family_map_case_insensitive_and_substring() -> None:
    fm = apkid_family_map_from_dict(
        {
            "schema_version": 1,
            "mappings": {
                "Bangcle": "packer_cs3_bangcle",
                "Qihoo 360": "packer_cs1_360_jiagu",
            },
        }
    )
    # exact (case-insensitive)
    assert fm.lookup("Bangcle") == "packer_cs3_bangcle"
    assert fm.lookup("bangcle") == "packer_cs3_bangcle"
    assert fm.lookup("BANGCLE") == "packer_cs3_bangcle"
    # substring fallback
    assert fm.lookup("Bangcle v2 (new)") == "packer_cs3_bangcle"
    assert fm.lookup("Qihoo 360 Packer") == "packer_cs1_360_jiagu"
    # no match
    assert fm.lookup("DexProtector") is None
    assert fm.lookup("") is None
    assert fm.lookup(" ") is None


def test_family_map_schema_errors() -> None:
    with pytest.raises(ApkidFamilyMapError, match="top-level must be a mapping"):
        apkid_family_map_from_dict([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ApkidFamilyMapError, match="unsupported schema_version"):
        apkid_family_map_from_dict({"schema_version": 2, "mappings": {}})
    with pytest.raises(ApkidFamilyMapError, match="'mappings' must be a dict"):
        apkid_family_map_from_dict({"schema_version": 1, "mappings": []})
    with pytest.raises(ApkidFamilyMapError, match="must be non-empty str"):
        apkid_family_map_from_dict(
            {"schema_version": 1, "mappings": {"": "x"}}
        )
    with pytest.raises(ApkidFamilyMapError, match="must be non-empty str"):
        apkid_family_map_from_dict(
            {"schema_version": 1, "mappings": {"a": ""}}
        )


def test_load_real_shipped_family_map() -> None:
    """Smoke test the YAML file we actually ship with the repo."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "data" / "apkid_family_map.yaml"
    assert path.exists(), f"shipped family map missing: {path}"
    fm = load_apkid_family_map(path)
    # Spot-check a few CS mappings we know about.
    assert fm.lookup("Bangcle") == "packer_cs3_bangcle"
    assert fm.lookup("Ijiami") == "packer_cs2_ijiami"
    assert fm.lookup("DexProtector") == "packer_cs5_dexprotector"
    # Substring works: "Qihoo 360 Packer" must map via "Qihoo 360" key.
    assert fm.lookup("Qihoo 360 Packer") == "packer_cs1_360_jiagu"


def test_load_family_map_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_apkid_family_map(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# cross_check decisions (all 6 agreement values)
# ---------------------------------------------------------------------------


def _make_result(*, hits_by_category: dict, filename: str = "a.apk") -> ApkidResult:
    """Helper to build an ApkidResult without JSON round-trip."""
    matches = []
    for cat, hits in hits_by_category.items():
        for h in hits:
            matches.append(ApkidMatch(filename=filename, category=cat, hit=h))
    return ApkidResult(
        apkid_version="3.1.0",
        rules_sha256="x",
        matches=tuple(matches),
        raw_json="",
    )


@pytest.fixture
def cs_family_map() -> ApkidFamilyMap:
    return apkid_family_map_from_dict(
        {
            "schema_version": 1,
            "mappings": {
                "Bangcle": "packer_cs3_bangcle",
                "Ijiami": "packer_cs2_ijiami",
                "Qihoo 360": "packer_cs1_360_jiagu",
            },
        }
    )


def test_cross_check_solid(cs_family_map: ApkidFamilyMap) -> None:
    r = _make_result(hits_by_category={APKID_CATEGORY_PACKER: ["Bangcle"]})
    rep = cross_check(
        r,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_SOLID
    assert rep.needs_manual_review is False
    assert rep.has_packer_hit is True
    assert rep.has_protector_hit is False
    assert rep.detected_families == ("packer_cs3_bangcle",)


def test_cross_check_solid_via_protector_category(
    cs_family_map: ApkidFamilyMap,
) -> None:
    r = _make_result(hits_by_category={APKID_CATEGORY_PROTECTOR: ["Bangcle"]})
    rep = cross_check(
        r,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_SOLID
    assert rep.has_packer_hit is False
    assert rep.has_protector_hit is True


def test_cross_check_mismatch(cs_family_map: ApkidFamilyMap) -> None:
    # we expected Bangcle but APKiD says Ijiami
    r = _make_result(hits_by_category={APKID_CATEGORY_PACKER: ["Ijiami"]})
    rep = cross_check(
        r,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_MISMATCH
    assert rep.needs_manual_review is True
    assert "packer_cs2_ijiami" in rep.detected_families


def test_cross_check_mismatch_with_unmapped_hit(
    cs_family_map: ApkidFamilyMap,
) -> None:
    r = _make_result(hits_by_category={APKID_CATEGORY_PACKER: ["UnknownPacker"]})
    rep = cross_check(
        r,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=cs_family_map,
    )
    # unmapped hit is NOT in detected_families but is in notes
    assert rep.agreement == AGREEMENT_MISMATCH
    assert rep.detected_families == ()
    assert any("UnknownPacker" in n for n in rep.notes)


def test_cross_check_no_apkid_detection(cs_family_map: ApkidFamilyMap) -> None:
    # we expected a packer but APKiD didn't fire any packer/protector rules
    r = _make_result(hits_by_category={"compiler": ["r8"], "anti_vm": ["x"]})
    rep = cross_check(
        r,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_NO_APKID_DETECTION
    assert rep.needs_manual_review is True
    assert rep.has_packer_hit is False
    assert rep.has_protector_hit is False


def test_cross_check_no_expectation(cs_family_map: ApkidFamilyMap) -> None:
    # benign apk, apkid found no packers
    r = _make_result(hits_by_category={"compiler": ["r8"]})
    rep = cross_check(
        r,
        apk_id="benign1",
        apk_path="benign1.apk",
        expected_family=None,
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_NO_EXPECTATION
    assert rep.needs_manual_review is False
    # treat empty-string expected the same as None
    rep2 = cross_check(
        r,
        apk_id="benign1",
        apk_path="benign1.apk",
        expected_family="",
        family_map=cs_family_map,
    )
    assert rep2.agreement == AGREEMENT_NO_EXPECTATION


def test_cross_check_apkid_false_positive(cs_family_map: ApkidFamilyMap) -> None:
    # benign apk, but apkid says packer
    r = _make_result(hits_by_category={APKID_CATEGORY_PACKER: ["Bangcle"]})
    rep = cross_check(
        r,
        apk_id="benign_but_detected",
        apk_path="x.apk",
        expected_family=None,
        family_map=cs_family_map,
    )
    assert rep.agreement == AGREEMENT_APKID_FALSE_POSITIVE
    assert rep.needs_manual_review is True
    assert "Bangcle" in " ".join(rep.notes)


def test_cross_check_apkid_failed_with_expected() -> None:
    # run_apkid returned an ApkidResult with error set, no hits
    failed = ApkidResult(
        apkid_version="",
        rules_sha256="",
        matches=(),
        raw_json="",
        error="apkid CLI not found",
    )
    rep = cross_check(
        failed,
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        family_map=None,
    )
    assert rep.agreement == AGREEMENT_APKID_FAILED
    assert rep.needs_manual_review is True
    # Notes mention both the raw error and the "expected but did not run".
    joined = " ".join(rep.notes)
    assert "apkid CLI not found" in joined
    assert "packer_cs3_bangcle" in joined


def test_cross_check_empty_apk_id_raises() -> None:
    r = _make_result(hits_by_category={})
    with pytest.raises(ValueError, match="apk_id must be non-empty"):
        cross_check(
            r,
            apk_id="",
            apk_path="a.apk",
            expected_family=None,
            family_map=None,
        )


# ---------------------------------------------------------------------------
# run_apkid (subprocess wrapper) -- graceful vs strict
# ---------------------------------------------------------------------------


def test_run_apkid_missing_apk_graceful(tmp_path: Path) -> None:
    r = run_apkid(tmp_path / "missing.apk", graceful=True)
    assert r.error is not None
    assert "does not exist" in r.error
    assert r.matches == ()


def test_run_apkid_missing_apk_strict(tmp_path: Path) -> None:
    with pytest.raises(ApkidError, match="does not exist"):
        run_apkid(tmp_path / "missing.apk", graceful=False)


def test_run_apkid_cli_missing_graceful(tmp_path: Path) -> None:
    # create a valid APK path but specify a bogus CLI
    apk = tmp_path / "dummy.apk"
    apk.write_bytes(b"PK\x03\x04")  # just any bytes; we're mocking the CLI call
    r = run_apkid(apk, apkid_cmd="definitely_not_installed_xyz123", graceful=True)
    assert r.error is not None
    assert "not found" in r.error.lower() or "system cannot find" in r.error.lower()


def test_run_apkid_cli_missing_strict(tmp_path: Path) -> None:
    apk = tmp_path / "dummy.apk"
    apk.write_bytes(b"PK\x03\x04")
    with pytest.raises(ApkidError):
        run_apkid(apk, apkid_cmd="definitely_not_installed_xyz123", graceful=False)


# ---------------------------------------------------------------------------
# write_apkid_reports_jsonl
# ---------------------------------------------------------------------------


def test_write_apkid_reports_jsonl_roundtrip(tmp_path: Path) -> None:
    r1 = ApkidCrossCheckReport(
        apk_id="apk1",
        apk_path="apk1.apk",
        expected_family="packer_cs3_bangcle",
        detected_families=("packer_cs3_bangcle",),
        agreement=AGREEMENT_SOLID,
        needs_manual_review=False,
        has_packer_hit=True,
        has_protector_hit=False,
        apkid_result=ApkidResult(
            apkid_version="3.1.0",
            rules_sha256="x",
            matches=(
                ApkidMatch(filename="apk1.apk", category="packer", hit="Bangcle"),
            ),
            raw_json="",
        ),
        notes=(),
    )
    r2 = ApkidCrossCheckReport(
        apk_id="apk2",
        apk_path="apk2.apk",
        expected_family=None,
        detected_families=(),
        agreement=AGREEMENT_NO_EXPECTATION,
        needs_manual_review=False,
        has_packer_hit=False,
        has_protector_hit=False,
        apkid_result=ApkidResult(
            apkid_version="3.1.0",
            rules_sha256="x",
            matches=(),
            raw_json="",
        ),
        notes=(),
    )
    out = tmp_path / "reports" / "apkid.jsonl"
    written = write_apkid_reports_jsonl([r1, r2], out)
    assert written == out
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed1 = json.loads(lines[0])
    parsed2 = json.loads(lines[1])
    assert parsed1["apk_id"] == "apk1"
    assert parsed1["agreement"] == AGREEMENT_SOLID
    assert parsed1["apkid_result"]["matches"][0]["hit"] == "Bangcle"
    assert parsed2["expected_family"] is None
    assert parsed2["agreement"] == AGREEMENT_NO_EXPECTATION
    # Atomic write leaves no stale .tmp file.
    assert not (out.parent / (out.name + ".tmp")).exists()


def test_cross_check_apk_full_flow_graceful(tmp_path: Path) -> None:
    """End-to-end: missing APK -> graceful -> APKID_FAILED agreement."""
    rep = cross_check_apk(
        tmp_path / "nope.apk",
        apk_id="nope",
        expected_family="packer_cs3_bangcle",
        family_map=None,
        apkid_cmd="definitely_not_installed_xyz123",
        graceful=True,
    )
    assert rep.agreement == AGREEMENT_APKID_FAILED
    assert rep.needs_manual_review is True
