"""Paper §5 table generation helpers.

Reads the aggregate reports produced by Track A (baselines run summary)
and Track B (track_b_pipeline.process_batch summary.jsonl) and renders
three Markdown tables ready to paste into ``docs/results_matrix.md`` or
the paper's §5:

* ``§5.2 region-level`` — method × train_mode × F1 × AUROC
* ``§5.3 per-packer AUROC`` — method × packer
* ``§5.4 APKiD baseline`` — agreement histogram on Track B

The tables are **schema-only** until real numbers are produced; cells
for which no input report exists are rendered as ``—``. This keeps the
tables re-runnable at any time during Week 11 to see current
progress without manually hand-tracking fill progress.

Input assumptions (none strictly required; missing inputs become ``—``):

1. Track A baselines: one JSON per (baseline, train_mode) under
   ``outputs/experiments/baselines/<baseline>/<train_mode>/summary.json``.
2. Track B labeling summary: ``outputs/experiments/track_b/labeling/summary.jsonl``.
3. Track B APKiD reports: per-pair ``apkid_report.json`` files discovered
   via the ``track_b_pipeline.discover_apkid_reports`` helper (created here).

No schema validation beyond best-effort key lookup -- this module is
purely a reporting layer and should never fail the main pipeline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EMPTY_CELL = "—"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineCell:
    """One row of the region-level table."""

    method: str
    train_mode: str
    f1: Optional[float]
    auroc: Optional[float]
    note: Optional[str] = None


@dataclass(frozen=True)
class PerPackerCell:
    """One (method, packer) cell in the per-packer AUROC table."""

    method: str
    packer_id: str
    auroc: Optional[float]


@dataclass
class ApkidAgreementCounts:
    """Aggregate APKiD agreement histogram."""

    solid: int = 0
    mismatch: int = 0
    no_detection: int = 0
    no_expectation: int = 0
    false_positive: int = 0
    apkid_failed: int = 0

    def record(self, agreement: str) -> None:
        if agreement == "solid":
            self.solid += 1
        elif agreement == "mismatch":
            self.mismatch += 1
        elif agreement == "no_apkid_detection":
            self.no_detection += 1
        elif agreement == "no_expectation":
            self.no_expectation += 1
        elif agreement == "apkid_false_positive":
            self.false_positive += 1
        elif agreement == "apkid_failed":
            self.apkid_failed += 1

    def total(self) -> int:
        return (
            self.solid
            + self.mismatch
            + self.no_detection
            + self.no_expectation
            + self.false_positive
            + self.apkid_failed
        )

    def to_dict(self) -> dict:
        return {
            "solid": self.solid,
            "mismatch": self.mismatch,
            "no_apkid_detection": self.no_detection,
            "no_expectation": self.no_expectation,
            "apkid_false_positive": self.false_positive,
            "apkid_failed": self.apkid_failed,
        }


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _f_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def read_track_a_summary(path: Path) -> Dict[str, Optional[float]]:
    """Best-effort extract F1 / AUROC from one Track A baseline summary.

    Returns a dict with keys ``f1`` and ``auroc``; missing keys -> None.
    Accepts a few plausible schemas (``metrics.region.f1``,
    ``region.f1``, bare ``f1``) so that this stays robust against
    past/future aggregation schema tweaks.
    """
    if not path.exists():
        return {"f1": None, "auroc": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"f1": None, "auroc": None}
    if not isinstance(data, dict):
        return {"f1": None, "auroc": None}

    def _dig(d: Mapping, *keys: str) -> Optional[float]:
        cur: object = d
        for k in keys:
            if isinstance(cur, Mapping) and k in cur:
                cur = cur[k]
            else:
                return None
        return _f_or_none(cur)

    for path_tuple in (
        ("region", "f1"),
        ("metrics", "region", "f1"),
        ("metrics", "f1"),
        ("f1",),
    ):
        v = _dig(data, *path_tuple)
        if v is not None:
            f1 = v
            break
    else:
        f1 = None

    for path_tuple in (
        ("region", "auroc"),
        ("metrics", "region", "auroc"),
        ("metrics", "auroc"),
        ("auroc",),
    ):
        v = _dig(data, *path_tuple)
        if v is not None:
            auroc = v
            break
    else:
        auroc = None

    return {"f1": f1, "auroc": auroc}


def collect_track_a_baselines(
    root: Path, methods: Sequence[str], train_modes: Sequence[str]
) -> List[BaselineCell]:
    """Expected layout: ``root/<method>/<train_mode>/summary.json``."""
    out: List[BaselineCell] = []
    for m in methods:
        for mode in train_modes:
            p = root / m / mode / "summary.json"
            metrics = read_track_a_summary(p)
            out.append(
                BaselineCell(
                    method=m,
                    train_mode=mode,
                    f1=metrics["f1"],
                    auroc=metrics["auroc"],
                    note=None if p.exists() else "no report yet",
                )
            )
    return out


def read_track_b_summary_jsonl(path: Path) -> List[Dict]:
    """Tolerate missing file -> empty list."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def aggregate_apkid(
    track_b_summary_rows: Sequence[Mapping],
) -> ApkidAgreementCounts:
    """Histogram APKiD agreement values straight from the batch summary."""
    counts = ApkidAgreementCounts()
    for row in track_b_summary_rows:
        agreement = row.get("apkid_agreement")
        if isinstance(agreement, str):
            counts.record(agreement)
    return counts


def aggregate_per_packer_auroc(
    track_b_summary_rows: Sequence[Mapping],
) -> Dict[str, Optional[float]]:
    """Group-by packer average of ``metrics.region.auroc`` when present.

    Track B pipeline's summary does not currently include AUROC (that's
    the downstream baseline's job). So until Week 11, this returns only
    the set of packer_ids seen, each mapped to None. The function is
    still useful as a schema-setter for the Markdown table.
    """
    out: Dict[str, List[float]] = defaultdict(list)
    for row in track_b_summary_rows:
        pid = row.get("packer_id")
        if not isinstance(pid, str):
            continue
        auroc = row.get("region_auroc") or row.get("auroc")
        v = _f_or_none(auroc)
        if v is not None:
            out[pid].append(v)
        else:
            out.setdefault(pid, [])
    # average
    return {
        pid: (sum(vals) / len(vals)) if vals else None
        for pid, vals in out.items()
    }


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return EMPTY_CELL
    return f"{v:.{digits}f}"


def render_region_level_table(cells: Sequence[BaselineCell]) -> str:
    out = [
        "### §5.2 Region-level baselines",
        "",
        "| Method | Train mode | F1 | AUROC | Note |",
        "|---|---|---|---|---|",
    ]
    for c in cells:
        note = c.note or ""
        out.append(
            f"| `{c.method}` | `{c.train_mode}` | {_fmt(c.f1)} | {_fmt(c.auroc)} | {note} |"
        )
    return "\n".join(out) + "\n"


def render_per_packer_table(
    methods: Sequence[str],
    per_packer_auroc_by_method: Mapping[str, Mapping[str, Optional[float]]],
) -> str:
    """Render method × packer AUROC table.

    ``per_packer_auroc_by_method[method][packer_id] = auroc or None``
    """
    all_packers = sorted(
        {p for d in per_packer_auroc_by_method.values() for p in d.keys()}
    )
    if not all_packers:
        return "### §5.3 Per-packer AUROC\n\n_(no Track B pairs discovered yet)_\n"
    out = ["### §5.3 Per-packer AUROC (Track B)", ""]
    header = "| Method \\ Packer | " + " | ".join(all_packers) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_packers)) + "|"
    out.append(header)
    out.append(sep)
    for m in methods:
        cells = [_fmt(per_packer_auroc_by_method.get(m, {}).get(p)) for p in all_packers]
        out.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_apkid_agreement_table(counts: ApkidAgreementCounts) -> str:
    total = counts.total()
    denom = total if total else 1
    out = [
        "### §5.4 APKiD third-party baseline (Track B)",
        "",
        "| Agreement | Count | % |",
        "|---|---|---|",
    ]
    for label, val in counts.to_dict().items():
        pct = f"{(val / denom * 100):.1f}%" if total else EMPTY_CELL
        out.append(f"| `{label}` | {val} | {pct} |")
    out.append(f"| **total** | **{total}** | **100.0%** |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PaperTablesBundle:
    region_level_md: str
    per_packer_md: str
    apkid_md: str

    def combined_markdown(self) -> str:
        return "\n\n".join([self.region_level_md, self.per_packer_md, self.apkid_md])


def build_paper_tables(
    *,
    baselines_root: Path,
    methods: Sequence[str],
    train_modes: Sequence[str],
    track_b_summary_jsonl: Path,
) -> PaperTablesBundle:
    """Produce all three tables in one call."""
    region_cells = collect_track_a_baselines(baselines_root, methods, train_modes)
    track_b_rows = read_track_b_summary_jsonl(track_b_summary_jsonl)

    # Per-packer AUROC — we currently have data per packer but not per method,
    # so emit one row per method with the same packer_id set for now.
    per_packer = aggregate_per_packer_auroc(track_b_rows)
    per_packer_by_method = {m: per_packer for m in methods}

    apkid_counts = aggregate_apkid(track_b_rows)

    return PaperTablesBundle(
        region_level_md=render_region_level_table(region_cells),
        per_packer_md=render_per_packer_table(methods, per_packer_by_method),
        apkid_md=render_apkid_agreement_table(apkid_counts),
    )


def write_tables_bundle(
    bundle: PaperTablesBundle, out_dir: Path, *, stem: str = "paper_tables"
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(bundle.combined_markdown(), encoding="utf-8")
    return {"md": md_path}
