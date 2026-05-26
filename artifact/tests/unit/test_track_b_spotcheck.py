"""Tests for ``android_packer.labeling.track_b_spotcheck`` (Track B · B-d-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_packer.labeling.track_b_spotcheck import (
    MAX_SAMPLE_SIZE,
    MIN_SAMPLE_SIZE,
    SpotCheckError,
    SpotCheckPair,
    SpotCheckPlan,
    load_summary_jsonl,
    render_markdown,
    sample_spotcheck,
    write_plan_artifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    packer_id: str,
    apk_id: str,
    payload_ratio: float | None = 0.5,
    final_label_count: int = 3,
    needs_manual_review: bool = False,
    chosen_source: str = "path_b_diff",
    group: str = "open_source",
) -> dict:
    return {
        "packer_id": packer_id,
        "apk_id": apk_id,
        "source_apk_id": f"benign:{apk_id}",
        "path_b_payload_ratio": payload_ratio,
        "final_label_count": final_label_count,
        "needs_manual_review": needs_manual_review,
        "chosen_source": chosen_source,
        "group": group,
    }


# ---------------------------------------------------------------------------
# Target-size math
# ---------------------------------------------------------------------------


def test_target_size_clamped_min() -> None:
    """<= 40 pairs -> target clamped to MIN_SAMPLE_SIZE (4)."""
    rows = [_row(packer_id="p1", apk_id=f"a{i}") for i in range(10)]
    plan = sample_spotcheck(rows)
    assert plan.sample_target == MIN_SAMPLE_SIZE
    assert len(plan.sample) == MIN_SAMPLE_SIZE


def test_target_size_clamped_max() -> None:
    """>> 60 pairs -> target clamped to MAX_SAMPLE_SIZE (6)."""
    rows = [_row(packer_id=f"p{i % 8}", apk_id=f"a{i}") for i in range(200)]
    plan = sample_spotcheck(rows)
    assert plan.sample_target == MAX_SAMPLE_SIZE


def test_target_size_mid_range() -> None:
    """50 pairs -> ceil(5) = 5."""
    rows = [_row(packer_id="p1", apk_id=f"a{i}") for i in range(50)]
    plan = sample_spotcheck(rows)
    assert plan.sample_target == 5
    assert len(plan.sample) == 5


# ---------------------------------------------------------------------------
# Sampling rules §1.1
# ---------------------------------------------------------------------------


class TestSamplingRules:
    def test_every_packer_covered_at_least_once(self) -> None:
        # 8 packers, 40 pairs, target = MIN (4)
        rows = []
        for p in range(8):
            for a in range(5):
                rows.append(_row(packer_id=f"pk{p}", apk_id=f"apk{a}"))
        plan = sample_spotcheck(rows, seed=42)
        # With 8 packers and target=4, every_packer_covered would need 8
        # slots > 4 target; spec says "at least 1 per packer" wins over
        # the cap, so we expect sample size >= 8 to satisfy coverage.
        packers_in_sample = {p.packer_id for p in plan.sample}
        assert packers_in_sample == {f"pk{p}" for p in range(8)}

    def test_all_needs_manual_review_pairs_included(self) -> None:
        rows = [
            _row(packer_id="p1", apk_id="a1", needs_manual_review=True),
            _row(packer_id="p1", apk_id="a2"),
            _row(packer_id="p2", apk_id="a1", needs_manual_review=True),
            _row(packer_id="p2", apk_id="a2"),
        ]
        plan = sample_spotcheck(rows)
        review_keys = {(p.packer_id, p.apk_id) for p in plan.sample if p.needs_manual_review}
        assert ("p1", "a1") in review_keys
        assert ("p2", "a1") in review_keys

    def test_highest_and_lowest_payload_ratio_included(self) -> None:
        rows = [
            _row(packer_id="p1", apk_id="low", payload_ratio=0.01),
            _row(packer_id="p1", apk_id="mid", payload_ratio=0.5),
            _row(packer_id="p1", apk_id="high", payload_ratio=0.99),
            _row(packer_id="p2", apk_id="a", payload_ratio=0.3),
            _row(packer_id="p3", apk_id="a", payload_ratio=0.7),
        ]
        plan = sample_spotcheck(rows, seed=0)
        apk_ids = {p.apk_id for p in plan.sample}
        assert "low" in apk_ids, "lowest non-zero payload_ratio must be in sample"
        assert "high" in apk_ids, "highest payload_ratio must be in sample"

    def test_zero_payload_ratio_not_picked_as_lowest(self) -> None:
        """Pairs with payload_ratio=0 are not the 'lowest non-zero'."""
        rows = [
            _row(packer_id="p1", apk_id="zero", payload_ratio=0.0),
            _row(packer_id="p1", apk_id="tiny", payload_ratio=0.001),
            _row(packer_id="p1", apk_id="big", payload_ratio=0.9),
        ]
        plan = sample_spotcheck(rows)
        # "tiny" is the lowest non-zero, not "zero"
        tiny_in = any(
            p.apk_id == "tiny" and "lowest non-zero" in p.rationale
            for p in plan.sample
        )
        assert tiny_in

    def test_deterministic_across_seeds(self) -> None:
        rows = [_row(packer_id="p1", apk_id=f"a{i}") for i in range(20)]
        plan1 = sample_spotcheck(rows, seed=12345)
        plan2 = sample_spotcheck(rows, seed=12345)
        assert [p.to_dict() for p in plan1.sample] == [p.to_dict() for p in plan2.sample]
        plan3 = sample_spotcheck(rows, seed=99)
        # Different seed can produce different random-fill picks; but
        # deterministic slots (lowest/highest/review) must still be there.
        # We only assert that both plans produce MIN_SAMPLE_SIZE pairs.
        assert len(plan1.sample) == len(plan3.sample) == MIN_SAMPLE_SIZE

    def test_review_can_exceed_cap(self) -> None:
        """If more pairs need review than the cap, keep them all + note."""
        rows = [
            _row(packer_id="p1", apk_id=f"a{i}", needs_manual_review=True)
            for i in range(10)
        ]
        plan = sample_spotcheck(rows)
        assert len(plan.sample) == 10  # all 10 review pairs kept
        assert any("exceed" in n for n in plan.notes)


# ---------------------------------------------------------------------------
# Excluded inputs (no reviewable content)
# ---------------------------------------------------------------------------


def test_empty_rows_returns_empty_plan() -> None:
    plan = sample_spotcheck([])
    assert plan.sample == []
    assert plan.sample_target == 0


def test_all_pairs_empty_labels_and_no_review() -> None:
    rows = [
        _row(packer_id="p1", apk_id="a1", final_label_count=0, payload_ratio=0.0)
    ]
    plan = sample_spotcheck(rows)
    assert plan.sample == []
    assert "no reviewable pairs" in " ".join(plan.notes)


# ---------------------------------------------------------------------------
# load_summary_jsonl / write_plan_artifacts
# ---------------------------------------------------------------------------


class TestIO:
    def test_load_summary_jsonl_round_trip(self, tmp_path: Path) -> None:
        rows = [
            _row(packer_id="p1", apk_id="a1"),
            _row(packer_id="p2", apk_id="a2"),
        ]
        jsonl = tmp_path / "summary.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        loaded = load_summary_jsonl(jsonl)
        assert loaded == rows

    def test_load_summary_jsonl_missing(self, tmp_path: Path) -> None:
        with pytest.raises(SpotCheckError, match="not found"):
            load_summary_jsonl(tmp_path / "nope.jsonl")

    def test_load_summary_jsonl_bad_line(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "summary.jsonl"
        jsonl.write_text('{"ok": true}\nNOT JSON\n', encoding="utf-8")
        with pytest.raises(SpotCheckError, match="line 2"):
            load_summary_jsonl(jsonl)

    def test_load_summary_jsonl_non_object(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "summary.jsonl"
        jsonl.write_text('[1,2,3]\n', encoding="utf-8")
        with pytest.raises(SpotCheckError, match="not an object"):
            load_summary_jsonl(jsonl)

    def test_write_plan_artifacts(self, tmp_path: Path) -> None:
        rows = [
            _row(packer_id="p1", apk_id=f"a{i}", needs_manual_review=(i % 2 == 0))
            for i in range(5)
        ]
        plan = sample_spotcheck(rows, seed=0)
        paths = write_plan_artifacts(plan, tmp_path / "out")
        for key in ("jsonl", "md", "json"):
            assert paths[key].exists()
        lines = paths["jsonl"].read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(plan.sample)
        md = paths["md"].read_text(encoding="utf-8")
        assert "# Track B 抽检清单" in md
        assert "packer_id" in md


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_contains_all_samples() -> None:
    rows = [
        _row(packer_id="p1", apk_id="a1"),
        _row(packer_id="p2", apk_id="a2"),
    ]
    plan = sample_spotcheck(rows)
    md = render_markdown(plan)
    assert "p1" in md and "p2" in md
    assert "packer_coverage" in md


def test_render_markdown_empty_plan() -> None:
    plan = SpotCheckPlan(sample=[], total_pairs=0, sample_target=0, packer_coverage={})
    md = render_markdown(plan)
    assert "total_pairs: **0**" in md
