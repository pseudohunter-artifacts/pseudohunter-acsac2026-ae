"""Track B spot-check sampler (B-d-1).

Given the ``summary.jsonl`` produced by ``track_b_pipeline.process_batch``,
sample 4-6 (packer, benign APK) pairs for 10% human label audit, following
the rules in ``docs/workstreams/track_b/spotcheck_protocol.md`` §1:

* each ``selected`` packer covered **at least once**
* total sample size = ``ceil(total_pairs * 0.10)`` clamped to ``[4, 6]``
* must include:
    - the pair with the highest ``path_b_payload_ratio``
      (near-degenerate boundary)
    - the pair with the lowest non-zero ``path_b_payload_ratio``
      (possible payload miss)
    - **every** pair with ``needs_manual_review=true`` (up to the cap)
    - otherwise random fill to reach the target size

Deterministic: accepts a ``seed`` so that sampling results are
reproducible across agent sessions and CI.

Writes ``spotcheck_sample.jsonl`` (one row per sampled pair) plus
``spotcheck_sample.md`` (Markdown table ready to paste into
``docs/validation/track_b_spotcheck.md``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


MIN_SAMPLE_SIZE = 4
MAX_SAMPLE_SIZE = 6
DEFAULT_RATIO = 0.10


class SpotCheckError(RuntimeError):
    """Raised when sampling inputs are invalid beyond recovery."""


@dataclass(frozen=True)
class SpotCheckPair:
    """A single sampled pair + the rationale for including it."""

    packer_id: str
    apk_id: str
    source_apk_id: str
    path_b_payload_ratio: Optional[float]
    final_label_count: int
    needs_manual_review: bool
    chosen_source: str
    group: str
    rationale: str
    raw: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(dataclasses.asdict(self))
        # drop the raw blob when serialising to keep files small
        d.pop("raw", None)
        return d


@dataclass
class SpotCheckPlan:
    """Full sampling plan for one batch run."""

    sample: List[SpotCheckPair]
    total_pairs: int
    sample_target: int
    packer_coverage: Dict[str, int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample": [p.to_dict() for p in self.sample],
            "total_pairs": self.total_pairs,
            "sample_target": self.sample_target,
            "packer_coverage": dict(self.packer_coverage),
            "notes": list(self.notes),
        }


def load_summary_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read a ``summary.jsonl`` produced by ``process_batch``."""
    if not path.exists():
        raise SpotCheckError(f"summary.jsonl not found: {path}")
    rows: List[Dict[str, object]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SpotCheckError(f"{path}: line {i + 1} not valid JSON: {e}") from e
        if not isinstance(row, dict):
            raise SpotCheckError(
                f"{path}: line {i + 1} not an object, got {type(row).__name__}"
            )
        rows.append(row)
    return rows


def _to_pair(row: Mapping[str, object], rationale: str) -> SpotCheckPair:
    return SpotCheckPair(
        packer_id=str(row.get("packer_id", "")),
        apk_id=str(row.get("apk_id", "")),
        source_apk_id=str(row.get("source_apk_id", "")),
        path_b_payload_ratio=_f_or_none(row.get("path_b_payload_ratio")),
        final_label_count=int(row.get("final_label_count", 0)),
        needs_manual_review=bool(row.get("needs_manual_review", False)),
        chosen_source=str(row.get("chosen_source", "")),
        group=str(row.get("group", "")),
        rationale=rationale,
        raw=row,
    )


def _f_or_none(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _target_size(total: int, ratio: float = DEFAULT_RATIO) -> int:
    if total <= 0:
        return 0
    raw = math.ceil(total * ratio)
    return max(MIN_SAMPLE_SIZE, min(MAX_SAMPLE_SIZE, raw))


def sample_spotcheck(
    rows: Sequence[Mapping[str, object]],
    *,
    ratio: float = DEFAULT_RATIO,
    seed: int = 0,
) -> SpotCheckPlan:
    """Apply the §1 sampling rules to ``rows``.

    ``rows`` is typically the parsed output of ``load_summary_jsonl``.
    """
    if not rows:
        return SpotCheckPlan(
            sample=[], total_pairs=0, sample_target=0, packer_coverage={}
        )

    # Defensive: only sample pairs whose pipeline actually ran and produced
    # something we can inspect; pairs with final_label_count==0 and no
    # review flag are excluded because there's literally nothing to review.
    reviewable = [
        r
        for r in rows
        if int(r.get("final_label_count", 0)) > 0
        or bool(r.get("needs_manual_review", False))
    ]
    if not reviewable:
        return SpotCheckPlan(
            sample=[],
            total_pairs=len(rows),
            sample_target=0,
            packer_coverage={},
            notes=[
                "no reviewable pairs in summary.jsonl "
                "(all had final_label_count=0 and needs_manual_review=false)"
            ],
        )

    target = _target_size(len(reviewable), ratio)
    if target <= 0:
        return SpotCheckPlan(
            sample=[], total_pairs=len(rows), sample_target=0, packer_coverage={}
        )

    notes: List[str] = []
    chosen: Dict[tuple, SpotCheckPair] = {}  # key = (packer_id, apk_id)

    def _add(row: Mapping[str, object], rationale: str) -> None:
        key = (row.get("packer_id", ""), row.get("apk_id", ""))
        if key in chosen:
            # Merge rationale rather than dup the row
            old = chosen[key]
            chosen[key] = dataclasses.replace(
                old, rationale=old.rationale + "; " + rationale
            )
            return
        chosen[key] = _to_pair(row, rationale)

    # 1. Every needs_manual_review pair (up to the cap; if more than the cap,
    #    keep them all and note that the target is exceeded).
    review_rows = [r for r in reviewable if r.get("needs_manual_review")]
    for r in review_rows:
        _add(r, "needs_manual_review=true")
    if len(chosen) > target:
        notes.append(
            f"needs_manual_review pairs ({len(review_rows)}) exceed the "
            f"sample target ({target}); keeping all of them so audit is "
            "comprehensive"
        )

    # 2. Highest and lowest non-zero payload_ratio, if different from #1.
    ratio_rows = [
        r
        for r in reviewable
        if _f_or_none(r.get("path_b_payload_ratio")) not in (None, 0.0)
    ]
    if ratio_rows:
        sorted_by_ratio = sorted(
            ratio_rows, key=lambda r: _f_or_none(r.get("path_b_payload_ratio")) or 0.0
        )
        low = sorted_by_ratio[0]
        high = sorted_by_ratio[-1]
        _add(low, "lowest non-zero payload_ratio (possible payload miss)")
        if high is not low:
            _add(high, "highest payload_ratio (degenerate boundary)")

    # 3. Per-packer coverage: every packer should have at least one entry.
    packers_seen = {p.packer_id for p in chosen.values()}
    all_packers = sorted({str(r.get("packer_id", "")) for r in reviewable})
    for pid in all_packers:
        if pid in packers_seen:
            continue
        # Pick the first reviewable row for this packer (deterministic)
        first = next(r for r in reviewable if r.get("packer_id") == pid)
        _add(first, f"packer {pid!r} coverage")

    # 4. Random fill up to target (if still under).
    if len(chosen) < target:
        need = target - len(chosen)
        rng = random.Random(seed)
        pool = [
            r
            for r in reviewable
            if (r.get("packer_id"), r.get("apk_id")) not in chosen
        ]
        rng.shuffle(pool)
        for r in pool[:need]:
            _add(r, "random fill")

    sample = list(chosen.values())
    sample.sort(key=lambda p: (p.packer_id, p.apk_id))
    coverage: Dict[str, int] = {}
    for p in sample:
        coverage[p.packer_id] = coverage.get(p.packer_id, 0) + 1

    return SpotCheckPlan(
        sample=sample,
        total_pairs=len(rows),
        sample_target=target,
        packer_coverage=coverage,
        notes=notes,
    )


def render_markdown(plan: SpotCheckPlan) -> str:
    """Render the plan as a Markdown block for docs/validation/*."""
    out: List[str] = []
    out.append("# Track B 抽检清单")
    out.append("")
    out.append(
        f"- total_pairs: **{plan.total_pairs}** "
        f"- sample_target: **{plan.sample_target}** "
        f"- actual_sampled: **{len(plan.sample)}**"
    )
    cov = ", ".join(f"{k}={v}" for k, v in sorted(plan.packer_coverage.items()))
    out.append(f"- packer_coverage: {cov or '(empty)'}")
    if plan.notes:
        out.append("")
        out.append("**Notes**:")
        for n in plan.notes:
            out.append(f"- {n}")
    out.append("")
    out.append("| # | packer_id | apk_id | payload_ratio | final_labels | review? | chosen_source | rationale |")
    out.append("|---|---|---|---|---|---|---|---|")
    for i, p in enumerate(plan.sample, start=1):
        ratio_str = (
            f"{p.path_b_payload_ratio:.3f}"
            if p.path_b_payload_ratio is not None
            else "n/a"
        )
        out.append(
            f"| {i} | `{p.packer_id}` | `{p.apk_id}` | {ratio_str} | "
            f"{p.final_label_count} | {'YES' if p.needs_manual_review else 'no'} "
            f"| `{p.chosen_source}` | {p.rationale} |"
        )
    out.append("")
    return "\n".join(out)


def write_plan_artifacts(plan: SpotCheckPlan, out_dir: Path) -> Dict[str, Path]:
    """Write spotcheck_sample.{jsonl,md,json} under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "spotcheck_sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for p in plan.sample:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    md_path = out_dir / "spotcheck_sample.md"
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    json_path = out_dir / "spotcheck_sample.json"
    json_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return {"jsonl": jsonl_path, "md": md_path, "json": json_path}
