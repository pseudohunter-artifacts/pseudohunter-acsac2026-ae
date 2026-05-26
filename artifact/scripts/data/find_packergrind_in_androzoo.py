"""Find AndroZoo entries whose pkg_name matches the 43 PackerGrind apps.

Reads ``data/real_world/packergrind/origins_manifest.json`` to get the
set of 43 package names PackerGrind tested (and we were able to resolve
on F-Droid), then scans the AndroZoo master CSV (gzipped or not) for
every matching row. Writes:

  outputs/experiments/androzoo_scout/packergrind_matches.jsonl
    One row per (pkg_name, sha256) AndroZoo entry, carrying the raw
    CSV columns we care about (dex_date, apk_size, vt_detection,
    markets, added) for later curation.

  outputs/experiments/androzoo_scout/packergrind_matches_by_pkg.json
    {pkg_name: {count, by_year: {YYYY: N}, example_sha256s: [...]}}
    for quick at-a-glance review.

Usage::

    python scripts/data/find_packergrind_in_androzoo.py
    python scripts/data/find_packergrind_in_androzoo.py \\
        --androzoo-csv data/androzoo/latest_with-added-date.csv.gz

We DO NOT call the AndroZoo API from here (no downloads). Downstream
callers can feed the produced sha256 list to
``scripts/data/fetch_androzoo.py --sha256-file ... --execute`` when
they decide what to pull.

AndroZoo CSV schema (after header, gzip-or-plain):
  sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,vt_detection,
  vt_scan_date,dex_size,markets,added
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS_MANIFEST = REPO_ROOT / "data" / "real_world" / "packergrind" / "origins_manifest.json"
DEFAULT_ANDROZOO_CSV = REPO_ROOT / "data" / "androzoo" / "latest_with-added-date.csv.gz"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "androzoo_scout"


def load_pkg_names(manifest_path: Path) -> List[str]:
    """Pull the 43 usable PackerGrind package names out of the origins manifest."""

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    usable_statuses = {"downloaded", "skipped_already_ok"}
    names: List[str] = []
    for row in doc.get("results", []):
        if row.get("status") not in usable_statuses:
            continue
        pkg = row.get("package_name") or row.get("canonical_package_name")
        if pkg:
            names.append(pkg)
    # Preserve order, drop duplicates defensively.
    seen: Set[str] = set()
    unique: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _open_csv(csv_path: Path) -> TextIO:
    """Open an AndroZoo CSV, transparently handling .gz."""

    if csv_path.suffix == ".gz":
        return io.TextIOWrapper(
            gzip.open(csv_path, "rb"), encoding="utf-8", newline=""
        )
    return open(csv_path, "r", encoding="utf-8", newline="")


def scan_androzoo_csv(
    csv_path: Path,
    target_pkgs: Set[str],
    *,
    progress_every: int = 1_000_000,
    stop_after_rows: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Stream the CSV and collect rows whose pkg_name is in target_pkgs."""

    if not csv_path.exists():
        raise FileNotFoundError(f"AndroZoo CSV not found: {csv_path}")

    matches: List[Dict[str, str]] = []
    with _open_csv(csv_path) as fp:
        reader = csv.DictReader(fp)
        expected_keys = {
            "sha256", "pkg_name", "dex_date", "apk_size",
            "vt_detection", "markets", "added",
        }
        missing = expected_keys - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"AndroZoo CSV missing expected columns: {sorted(missing)}; "
                f"got fieldnames={reader.fieldnames!r}"
            )
        for i, row in enumerate(reader, start=1):
            if stop_after_rows is not None and i > stop_after_rows:
                break
            pkg = row.get("pkg_name")
            if pkg in target_pkgs:
                matches.append({k: row.get(k, "") for k in expected_keys})
            if i % progress_every == 0:
                print(
                    f"[scan] {i:>12,} rows read, {len(matches):>6} matches so far",
                    flush=True,
                )
    return matches


def summarise_matches(matches: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Group matches by pkg_name with year histogram + sample hashes."""

    by_pkg: Dict[str, Dict] = defaultdict(
        lambda: {"count": 0, "by_year": defaultdict(int), "example_sha256s": []}
    )
    for row in matches:
        pkg = row["pkg_name"]
        by_pkg[pkg]["count"] += 1
        dex_date = row.get("dex_date", "")
        year = dex_date[:4] if dex_date else "unknown"
        by_pkg[pkg]["by_year"][year] += 1
        if len(by_pkg[pkg]["example_sha256s"]) < 5:
            by_pkg[pkg]["example_sha256s"].append(row["sha256"])

    # Normalise defaultdicts -> dicts so json.dumps works predictably.
    return {
        pkg: {
            "count": d["count"],
            "by_year": dict(sorted(d["by_year"].items())),
            "example_sha256s": d["example_sha256s"],
        }
        for pkg, d in sorted(by_pkg.items())
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origins-manifest", type=Path, default=DEFAULT_ORIGINS_MANIFEST)
    p.add_argument("--androzoo-csv", type=Path, default=DEFAULT_ANDROZOO_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the 43 target packages + CSV file info; do not scan.",
    )
    p.add_argument(
        "--stop-after-rows",
        type=int,
        default=None,
        help="For smoke: stop scanning after N CSV rows.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    pkg_names = load_pkg_names(args.origins_manifest)
    print(f"loaded {len(pkg_names)} target package names from {args.origins_manifest}")
    for pn in pkg_names[:5]:
        print(f"  e.g. {pn}")

    if not args.androzoo_csv.exists():
        print(
            f"[WARN] AndroZoo CSV not found at {args.androzoo_csv}",
            file=sys.stderr,
        )
        if not args.dry_run:
            print(
                "[ABORT] run the download runner first (see "
                "docs/references/dataset_download_guide.md \u00a73.2), or pass "
                "--androzoo-csv <path> to point at an existing copy.",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        size_hint = "(missing)"
        if args.androzoo_csv.exists():
            size_hint = f"{args.androzoo_csv.stat().st_size / (1024*1024):.1f} MB"
        print(f"[dry-run] AndroZoo CSV: {args.androzoo_csv}  size={size_hint}")
        print(f"[dry-run] would scan for {len(pkg_names)} package names")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matches_path = args.out_dir / "packergrind_matches.jsonl"
    summary_path = args.out_dir / "packergrind_matches_by_pkg.json"
    meta_path = args.out_dir / "_meta.json"

    t0 = datetime.now()
    matches = scan_androzoo_csv(
        args.androzoo_csv,
        set(pkg_names),
        stop_after_rows=args.stop_after_rows,
    )
    elapsed = (datetime.now() - t0).total_seconds()

    with matches_path.open("w", encoding="utf-8") as fp:
        for row in matches:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_pkg = summarise_matches(matches)
    summary_path.write_text(
        json.dumps(by_pkg, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "androzoo_csv": str(args.androzoo_csv),
        "androzoo_csv_size_bytes": args.androzoo_csv.stat().st_size,
        "origins_manifest": str(args.origins_manifest),
        "target_pkg_count": len(pkg_names),
        "match_row_count": len(matches),
        "covered_pkg_count": len(by_pkg),
        "elapsed_s": round(elapsed, 1),
        "stop_after_rows": args.stop_after_rows,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(f"[done] matches -> {matches_path}  ({len(matches):,} rows)")
    print(f"[done] summary -> {summary_path}  ({len(by_pkg)} / {len(pkg_names)} packages covered)")
    print(f"[done] meta    -> {meta_path}  (elapsed {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
