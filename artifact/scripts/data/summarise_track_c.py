"""Track C labels summariser.

Produces the aggregate counters + per-family / per-packer cross-tabs that
go into the paper's §4-Data table. Re-run any time ``labels.jsonl`` is
updated (e.g. after a new labelling stage run or after corpus expansion).

Usage::

    python scripts/data/summarise_track_c.py
    python scripts/data/summarise_track_c.py \\
        --labels outputs/experiments/track_c/labels.jsonl \\
        --out-json outputs/experiments/track_c/summary.json

Exit codes::

    0 = success (a summary was written)
    2 = labels.jsonl missing / unreadable
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_rows(labels_jsonl: Path) -> List[Dict[str, Any]]:
    if not labels_jsonl.exists():
        print(f"[FATAL] labels file missing: {labels_jsonl}", file=sys.stderr)
        sys.exit(2)
    rows: List[Dict[str, Any]] = []
    for line in labels_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source = dict(collections.Counter(r["source"] for r in rows))
    is_packed = collections.Counter(
        (r.get("labels") or {}).get("is_packed_probed") for r in rows
    )
    native = collections.Counter(
        (r.get("labels") or {}).get("has_native_libs") for r in rows
    )
    assets_dex = collections.Counter(
        (r.get("labels") or {}).get("has_assets_dex") for r in rows
    )

    # Packer-family histogram (from mapped suspected_packer lists)
    pack_fams: "collections.Counter[str]" = collections.Counter()
    pack_fams_by_source: Dict[str, "collections.Counter[str]"] = collections.defaultdict(collections.Counter)
    for r in rows:
        if not (r.get("labels") or {}).get("is_packed_probed"):
            continue
        for f in (r.get("labels") or {}).get("suspected_packer") or []:
            pack_fams[f] += 1
            pack_fams_by_source[r["source"]][f] += 1

    # Cross-tab is_packed * has_assets_dex (Gen-2 signature check)
    xtab: "collections.Counter[tuple]" = collections.Counter()
    for r in rows:
        lbl = r.get("labels") or {}
        xtab[(lbl.get("is_packed_probed"), lbl.get("has_assets_dex"))] += 1

    # Per malware-family rollup for the §4 table
    packed_by_fam: Dict[str, Dict[str, int]] = {}
    for r in rows:
        fam = r["family"]
        lbl = r.get("labels") or {}
        packed_by_fam.setdefault(fam, {"total": 0, "packed": 0, "native": 0, "assets_dex": 0})
        packed_by_fam[fam]["total"] += 1
        if lbl.get("is_packed_probed") is True:
            packed_by_fam[fam]["packed"] += 1
        if lbl.get("has_native_libs") is True:
            packed_by_fam[fam]["native"] += 1
        if lbl.get("has_assets_dex") is True:
            packed_by_fam[fam]["assets_dex"] += 1
    # Sort families by packed-count desc then name
    family_rollup = [
        dict(family=fam, **counts)
        for fam, counts in sorted(
            packed_by_fam.items(),
            key=lambda kv: (-kv[1]["packed"], kv[0]),
        )
    ]

    total = len(rows)
    packed_count = is_packed.get(True, 0)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": {
            "samples": total,
            "families": len(packed_by_fam),
            "packed": packed_count,
            "packed_rate": round(packed_count / total, 4) if total else None,
            "has_native_libs": native.get(True, 0),
            "has_assets_dex": assets_dex.get(True, 0),
        },
        "by_source": by_source,
        "is_packed_probed": {str(k): v for k, v in is_packed.items()},
        "has_native_libs": {str(k): v for k, v in native.items()},
        "has_assets_dex": {str(k): v for k, v in assets_dex.items()},
        "packer_family_hist": dict(pack_fams.most_common()),
        "packer_family_by_source": {
            src: dict(fams.most_common()) for src, fams in pack_fams_by_source.items()
        },
        "is_packed_x_assets_dex": {
            f"packed={k[0]} / assets_dex={k[1]}": v
            for k, v in sorted(xtab.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))
        },
        "per_family_rollup": family_rollup,
    }


def pretty_print(summary: Dict[str, Any]) -> None:
    t = summary["totals"]
    print(f"=== Track C summary ({t['samples']} samples / {t['families']} families) ===")
    print()
    print(f"  packed               = {t['packed']:5d}  ({(t['packed_rate'] or 0) * 100:.1f}%)")
    print(f"  has_native_libs      = {t['has_native_libs']:5d}")
    print(f"  has_assets_dex       = {t['has_assets_dex']:5d}")
    print()
    print("By source:")
    for s, n in sorted(summary["by_source"].items()):
        print(f"  {s:10}  {n}")
    print()
    if summary["packer_family_hist"]:
        print(f"Packer families detected ({len(summary['packer_family_hist'])}):")
        for f, n in summary["packer_family_hist"].items():
            print(f"  {f:45}  {n}")
    print()
    print("Top-10 malware families with packed samples:")
    top = [r for r in summary["per_family_rollup"] if r["packed"] > 0][:10]
    for entry in top:
        print(f"  {entry['family']:35}  total={entry['total']:3}  packed={entry['packed']:3}")


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_labels = repo_root / "outputs" / "experiments" / "track_c" / "labels.jsonl"
    default_out = repo_root / "outputs" / "experiments" / "track_c" / "summary.json"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip())
    p.add_argument("--labels", type=Path, default=default_labels,
                   help=f"Path to labels.jsonl (default: {default_labels}).")
    p.add_argument("--out-json", type=Path, default=default_out,
                   help=f"Where to write the summary JSON (default: {default_out}).")
    p.add_argument("--no-print", action="store_true",
                   help="Skip the human-readable stdout rendering (JSON-only mode).")
    args = p.parse_args(argv)

    rows = _load_rows(args.labels)
    summary = build_summary(rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] wrote summary: {args.out_json}")
    if not args.no_print:
        print()
        pretty_print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
