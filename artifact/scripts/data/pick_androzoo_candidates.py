#!/usr/bin/env python3
"""Pick an AndroZoo A2 candidate list from the PackerGrind scout output.

Context
-------
``scripts/data/find_packergrind_in_androzoo.py`` scans the 3.55 GB AndroZoo
master CSV once and writes all (pkg_name, sha256) rows that match the 42
PackerGrind-resolvable package names into::

    outputs/experiments/androzoo_scout/packergrind_matches.jsonl

This follow-on tool reads that JSONL and selects a small, reviewable
candidate set for **operator-approved download** via
``scripts/data/fetch_androzoo.py --sha256-file ... --execute``. It does
NOT call the AndroZoo API itself; all it does is pick + format.

Selection strategy
------------------
The default recipe mirrors what the paper needs for a Track B v1 real-world
case study (§5) without blowing past the academic-download etiquette:

1. **Group by pkg_name**; keep at most ``--per-pkg`` rows per group.
2. **Prefer rows with VirusTotal detections** (``vt_detection > 0``) -- they
   are the most interesting for a "real Gen-2 packing" story. Fall back to
   ``vt_detection == 0`` only if a package has no flagged commits.
3. **Prefer newer dex_date** inside each (pkg_name, vt_detection > 0) group
   so we avoid 2015-era toolchain artefacts.
4. **Cap at --limit samples overall** (default 30) so the caller can
   sanity-check before kicking off a real download.
5. **Deduplicate by sha256**.

Outputs
-------
* ``outputs/experiments/androzoo_scout/candidates.sha256.txt`` -- one sha256
  per line, ready to feed to ``fetch_androzoo.py --sha256-file``.
* ``outputs/experiments/androzoo_scout/candidates.jsonl`` -- the selected
  rows with their full metadata, so reviewers can eyeball them.
* ``outputs/experiments/androzoo_scout/candidates.summary.json`` -- the
  aggregate stats (n_total_matches, n_selected, per-package breakdown,
  vt_detection histogram) plus the exact CLI recipe to invoke
  ``fetch_androzoo.py`` next.

Usage
-----
::

    python scripts/data/pick_androzoo_candidates.py
    python scripts/data/pick_androzoo_candidates.py --limit 20 --per-pkg 1
    python scripts/data/pick_androzoo_candidates.py --min-vt 3  # only interesting malware
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_int(v: Any, default: int = 0) -> int:
    """AndroZoo CSV columns arrive as strings; parse safely."""
    try:
        if v is None or v == "":
            return default
        return int(v)
    except (ValueError, TypeError):
        return default


def _parse_dex_date(v: Any) -> Optional[datetime]:
    """Best-effort ISO-ish parser. Rows with 1981-01-01 sentinel are treated
    as unknown and sort to the bottom of "newest first" orderings."""
    if not v or not isinstance(v, str):
        return None
    try:
        # AndroZoo uses "YYYY-MM-DD HH:MM:SS[.ffffff]"
        head = v.split(".", 1)[0]
        dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        if dt.year <= 1990:  # sentinel for "unknown"
            return None
        return dt
    except Exception:
        return None


def _load_matches(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[FATAL] matches file missing: {path}", file=sys.stderr)
        print("        Run scripts/data/find_packergrind_in_androzoo.py first.", file=sys.stderr)
        sys.exit(2)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_candidates(
    rows: List[Dict[str, Any]],
    *,
    per_pkg: int,
    limit: int,
    min_vt: int,
) -> List[Dict[str, Any]]:
    """Return up to ``limit`` selected rows."""
    # Bucket by pkg_name. Inside each bucket, sort by (vt_detection desc,
    # dex_date desc) so the "most interesting, most recent" floats to the top.
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r.get("pkg_name", "")].append(r)

    for pkg, group in buckets.items():
        group.sort(
            key=lambda r: (
                -_parse_int(r.get("vt_detection", 0)),
                -(d.timestamp() if (d := _parse_dex_date(r.get("dex_date"))) else 0),
            )
        )

    # Select up to per_pkg * |pkgs|; seen set dedupes by sha256.
    selected: List[Dict[str, Any]] = []
    seen: set = set()
    for pkg in sorted(buckets):
        for r in buckets[pkg][:per_pkg]:
            sha = (r.get("sha256") or "").lower()
            if not sha or sha in seen:
                continue
            if _parse_int(r.get("vt_detection", 0)) < min_vt:
                # Below the VT threshold; skip. min_vt=0 keeps everything.
                continue
            seen.add(sha)
            selected.append(r)

    # Sort overall picks newest-first so the summary & txt list are stable.
    selected.sort(
        key=lambda r: -(d.timestamp() if (d := _parse_dex_date(r.get("dex_date"))) else 0)
    )
    if limit and len(selected) > limit:
        selected = selected[:limit]
    return selected


def build_summary(
    all_rows: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    *,
    limit: int,
    per_pkg: int,
    min_vt: int,
    out_dir: Path,
) -> Dict[str, Any]:
    vt_hist = Counter(_parse_int(r.get("vt_detection", 0)) for r in all_rows)
    per_pkg_counts = Counter(r.get("pkg_name", "") for r in all_rows)
    per_pkg_selected = Counter(r.get("pkg_name", "") for r in selected)

    # Collapse vt_hist into buckets for readability.
    def _bucket(v: int) -> str:
        if v == 0:
            return "0"
        if v <= 2:
            return "1-2"
        if v <= 5:
            return "3-5"
        if v <= 10:
            return "6-10"
        return "11+"
    bucketed: Dict[str, int] = {}
    for v, n in vt_hist.items():
        bucketed[_bucket(v)] = bucketed.get(_bucket(v), 0) + n

    sha_file = out_dir / "candidates.sha256.txt"
    jsonl_file = out_dir / "candidates.jsonl"
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": {
            "matches_total": len(all_rows),
            "matches_unique_pkgs": len(per_pkg_counts),
        },
        "selection": {
            "limit": limit,
            "per_pkg": per_pkg,
            "min_vt": min_vt,
            "selected_total": len(selected),
            "selected_unique_pkgs": len(per_pkg_selected),
        },
        "vt_detection_histogram_all": bucketed,
        "top10_pkgs_by_corpus_size": dict(per_pkg_counts.most_common(10)),
        "selected_per_pkg": dict(per_pkg_selected),
        "artefacts": {
            "sha256_txt": str(sha_file),
            "selected_jsonl": str(jsonl_file),
        },
        "next_step_cli": (
            "# Operator-approved download (reads this sha256 list):\n"
            f"python scripts\\data\\fetch_androzoo.py "
            f"--sha256-file \"{sha_file}\" "
            "--execute "
            "--sleep-sec 0.5 "
            "--out-dir data\\androzoo\\apks "
            "--json-summary outputs\\experiments\\androzoo_scout\\fetch.json"
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_matches = repo_root / "outputs" / "experiments" / "androzoo_scout" / "packergrind_matches.jsonl"
    default_out_dir = repo_root / "outputs" / "experiments" / "androzoo_scout"

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip())
    ap.add_argument("--matches-jsonl", type=Path, default=default_matches)
    ap.add_argument("--out-dir", type=Path, default=default_out_dir)
    ap.add_argument("--per-pkg", type=int, default=1,
                    help="Maximum rows per pkg_name (default 1, i.e. 1 APK per PackerGrind app).")
    ap.add_argument("--limit", type=int, default=30,
                    help="Overall cap on selected samples (default 30).")
    ap.add_argument("--min-vt", type=int, default=0,
                    help="Minimum vt_detection count to select (default 0 = keep all).")
    args = ap.parse_args(argv)

    rows = _load_matches(args.matches_jsonl)
    selected = select_candidates(
        rows,
        per_pkg=args.per_pkg,
        limit=args.limit,
        min_vt=args.min_vt,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sha_file = args.out_dir / "candidates.sha256.txt"
    jsonl_file = args.out_dir / "candidates.jsonl"
    summary_file = args.out_dir / "candidates.summary.json"

    sha_file.write_text(
        "\n".join((r.get("sha256") or "").lower() for r in selected) + "\n",
        encoding="utf-8",
    )
    with jsonl_file.open("w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = build_summary(
        rows, selected,
        limit=args.limit, per_pkg=args.per_pkg, min_vt=args.min_vt,
        out_dir=args.out_dir,
    )
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] matches read:    {len(rows):5d} rows from {args.matches_jsonl.name}")
    print(f"[OK] selected:        {len(selected):5d} rows")
    print(f"[OK] sha256 list  ->  {sha_file}")
    print(f"[OK] jsonl        ->  {jsonl_file}")
    print(f"[OK] summary      ->  {summary_file}")
    print()
    print("Selected (pkg_name / vt_detection / dex_date / sha256[:12]):")
    for r in selected:
        pkg = r.get("pkg_name", "?")
        vt = _parse_int(r.get("vt_detection", 0))
        dd = r.get("dex_date", "?")
        sha = (r.get("sha256") or "?")[:12].lower()
        print(f"  {pkg:38}  vt={vt:3d}  {dd:23}  {sha}")
    print()
    print("NEXT STEP (operator-approved; actually downloads APKs from AndroZoo):")
    print(f"  {summary['next_step_cli']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
