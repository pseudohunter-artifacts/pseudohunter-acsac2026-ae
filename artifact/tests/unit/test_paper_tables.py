"""Tests for ``android_packer.experiments.paper_tables``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_packer.experiments.paper_tables import (
    EMPTY_CELL,
    ApkidAgreementCounts,
    BaselineCell,
    PaperTablesBundle,
    aggregate_apkid,
    aggregate_per_packer_auroc,
    build_paper_tables,
    collect_track_a_baselines,
    read_track_a_summary,
    read_track_b_summary_jsonl,
    render_apkid_agreement_table,
    render_per_packer_table,
    render_region_level_table,
    write_tables_bundle,
)


# ---------------------------------------------------------------------------
# Track A reader
# ---------------------------------------------------------------------------


def test_read_track_a_summary_missing(tmp_path: Path) -> None:
    r = read_track_a_summary(tmp_path / "nope.json")
    assert r == {"f1": None, "auroc": None}


def test_read_track_a_summary_dict_top(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"f1": 0.77, "auroc": 0.92}), encoding="utf-8")
    assert read_track_a_summary(p) == {"f1": 0.77, "auroc": 0.92}


def test_read_track_a_summary_nested(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"region": {"f1": 0.5, "auroc": 0.8}}), encoding="utf-8"
    )
    assert read_track_a_summary(p) == {"f1": 0.5, "auroc": 0.8}


def test_read_track_a_summary_double_nested(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"metrics": {"region": {"f1": 0.1, "auroc": 0.2}}}),
        encoding="utf-8",
    )
    assert read_track_a_summary(p) == {"f1": 0.1, "auroc": 0.2}


def test_read_track_a_summary_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text("not json", encoding="utf-8")
    assert read_track_a_summary(p) == {"f1": None, "auroc": None}


def test_read_track_a_summary_top_level_not_dict(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert read_track_a_summary(p) == {"f1": None, "auroc": None}


# ---------------------------------------------------------------------------
# Track A collect
# ---------------------------------------------------------------------------


def test_collect_track_a_baselines_mixed(tmp_path: Path) -> None:
    root = tmp_path / "baselines"
    # one baseline with data, one without
    (root / "entropy" / "same_set").mkdir(parents=True)
    (root / "entropy" / "same_set" / "summary.json").write_text(
        json.dumps({"f1": 0.42, "auroc": 0.71}), encoding="utf-8"
    )
    cells = collect_track_a_baselines(
        root, methods=["entropy", "ngram_logreg"], train_modes=["same_set"]
    )
    assert len(cells) == 2
    assert cells[0].method == "entropy"
    assert cells[0].f1 == 0.42
    assert cells[0].auroc == 0.71
    assert cells[0].note is None
    assert cells[1].method == "ngram_logreg"
    assert cells[1].f1 is None
    assert cells[1].note == "no report yet"


# ---------------------------------------------------------------------------
# Track B readers / aggregators
# ---------------------------------------------------------------------------


def test_read_track_b_summary_jsonl_missing(tmp_path: Path) -> None:
    assert read_track_b_summary_jsonl(tmp_path / "nope.jsonl") == []


def test_read_track_b_summary_jsonl_skips_bad_lines(tmp_path: Path) -> None:
    p = tmp_path / "summary.jsonl"
    p.write_text(
        '{"ok": true}\nnot json\n{"another": 1}\n', encoding="utf-8"
    )
    rows = read_track_b_summary_jsonl(p)
    assert rows == [{"ok": True}, {"another": 1}]


def test_aggregate_apkid_histogram() -> None:
    rows = [
        {"apkid_agreement": "solid"},
        {"apkid_agreement": "solid"},
        {"apkid_agreement": "mismatch"},
        {"apkid_agreement": "no_apkid_detection"},
        {"apkid_agreement": "no_expectation"},
        {"apkid_agreement": "apkid_false_positive"},
        {"apkid_agreement": "apkid_failed"},
        {"apkid_agreement": None},  # should be ignored
        {},  # missing agreement — ignored
    ]
    counts = aggregate_apkid(rows)
    assert counts.solid == 2
    assert counts.mismatch == 1
    assert counts.no_detection == 1
    assert counts.no_expectation == 1
    assert counts.false_positive == 1
    assert counts.apkid_failed == 1
    assert counts.total() == 7


def test_aggregate_per_packer_auroc_empty() -> None:
    assert aggregate_per_packer_auroc([]) == {}


def test_aggregate_per_packer_auroc_averages() -> None:
    rows = [
        {"packer_id": "p1", "region_auroc": 0.8},
        {"packer_id": "p1", "region_auroc": 0.6},
        {"packer_id": "p2", "auroc": 0.9},  # fallback key
        {"packer_id": "p3"},  # no auroc -> None
    ]
    result = aggregate_per_packer_auroc(rows)
    assert abs(result["p1"] - 0.7) < 1e-9
    assert result["p2"] == 0.9
    assert result["p3"] is None


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_region_level_table() -> None:
    cells = [
        BaselineCell("entropy", "same_set", 0.42, 0.71, None),
        BaselineCell(
            "ngram_logreg",
            "holdout_transform",
            None,
            None,
            "no report yet",
        ),
    ]
    md = render_region_level_table(cells)
    assert "§5.2" in md
    assert "`entropy`" in md and "0.420" in md and "0.710" in md
    assert "`ngram_logreg`" in md and EMPTY_CELL in md
    assert "no report yet" in md


def test_render_per_packer_table_empty() -> None:
    md = render_per_packer_table(["entropy"], {})
    assert "no track b pairs" in md.lower()


def test_render_per_packer_table_data() -> None:
    md = render_per_packer_table(
        ["entropy", "ngram_logreg"],
        {
            "entropy": {"p1": 0.6, "p2": None},
            "ngram_logreg": {"p1": 0.8, "p2": 0.7},
        },
    )
    assert "p1" in md and "p2" in md
    assert "0.600" in md and "0.800" in md
    assert EMPTY_CELL in md  # ngram/p2 is None for entropy? no — entropy has p2 None
    # ngram_logreg row must include 0.700
    assert "0.700" in md


def test_render_apkid_agreement_table() -> None:
    counts = ApkidAgreementCounts(solid=3, mismatch=1, no_detection=2)
    md = render_apkid_agreement_table(counts)
    assert "§5.4" in md
    assert "solid" in md
    assert "total" in md
    # 3 + 1 + 2 = 6
    assert "**6**" in md


def test_render_apkid_agreement_table_zero() -> None:
    counts = ApkidAgreementCounts()
    md = render_apkid_agreement_table(counts)
    assert "total" in md
    # No division by zero even when total is 0
    assert EMPTY_CELL in md


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_build_paper_tables_end_to_end(tmp_path: Path) -> None:
    # Fake Track A
    baselines = tmp_path / "baselines"
    (baselines / "entropy" / "same_set").mkdir(parents=True)
    (baselines / "entropy" / "same_set" / "summary.json").write_text(
        json.dumps({"f1": 0.5, "auroc": 0.8}), encoding="utf-8"
    )
    # Fake Track B summary.jsonl
    track_b = tmp_path / "labeling"
    track_b.mkdir()
    with (track_b / "summary.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"packer_id": "s3", "apkid_agreement": "solid"}) + "\n")
        fh.write(json.dumps({"packer_id": "cs1", "apkid_agreement": "mismatch"}) + "\n")

    bundle = build_paper_tables(
        baselines_root=baselines,
        methods=["entropy", "ngram_logreg"],
        train_modes=["same_set"],
        track_b_summary_jsonl=track_b / "summary.jsonl",
    )
    assert isinstance(bundle, PaperTablesBundle)
    md = bundle.combined_markdown()
    assert "§5.2" in md and "§5.3" in md and "§5.4" in md
    assert "entropy" in md
    # packers seen
    assert "s3" in md and "cs1" in md
    # apkid counts rendered
    assert "solid" in md and "mismatch" in md


def test_write_tables_bundle(tmp_path: Path) -> None:
    bundle = PaperTablesBundle(
        region_level_md="A",
        per_packer_md="B",
        apkid_md="C",
    )
    paths = write_tables_bundle(bundle, tmp_path / "out")
    assert paths["md"].exists()
    content = paths["md"].read_text(encoding="utf-8")
    assert content == "A\n\nB\n\nC"
