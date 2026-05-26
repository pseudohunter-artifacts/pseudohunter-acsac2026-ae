"""Build a normalized PackerGrind app inventory.

INPUT  .feature_list.txt  (output of `git -C adaptiveunpacker ls-tree -r HEAD features`)
OUTPUT data/real_world/packergrind/app_list.json  (structured index)
OUTPUT data/real_world/packergrind/app_list.csv   (flat view for spreadsheets)
OUTPUT data/real_world/packergrind/app_list.md    (human-readable summary)

Schema of app_list.json::

    {
      "source": "TheSillyStever/adaptiveunpacker (PackerGrind feature dir)",
      "commit": "2c26cc78a139c56f0829437abe5d1b99e5d0eaff",
      "total_entries": 691,
      "packers": ["Ali", "Baidu", "Bangcle", "Ijiami", "Qihoo", "Tencent"],
      "api_levels": ["15", "16", "18"],
      "by_app": {
         "2048_release": {
             "coverage": {"Ali": ["15", "16"], "Baidu": ["15", "16"], ...},
             "api_levels": ["15", "16", "18"],
             "packer_count": 6,
             "n_rows": 18
         },
         ...
      },
      "by_packer": {"Ali": {"apps": [...], "n_rows": 83}, ...}
    }

This is the source-of-truth list used by ``fetch_fdroid_origins.py`` and
``build_track_b_from_fdroid.py``. The actual app name canonicalization (strip
``_release``, ``_debug_unaligned``, ``_debug``, ``_aligned`` suffixes) is
done here so downstream scripts don't repeat that logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Packer codename suffix that PackerGrind appends to each feature filename.
# Verified by scanning the actual feature list shipped in the upstream repo:
#   _al = Ali (AliProtect)
#   _bd = Baidu
#   _bb = Bangcle  (note: ``_bb`` not ``_bc``; upstream abbreviation)
#   _ij = Ijiami
#   _qh = Qihoo360
#   _tx = Tencent (Legu)  (note: ``_tx`` = tencent-xiaomi; upstream abbreviation)
SUFFIX_TO_PACKER = {
    "al": "Ali",
    "bd": "Baidu",
    "bb": "Bangcle",
    "ij": "Ijiami",
    "qh": "Qihoo",
    "tx": "Tencent",
}

# PackerGrind's "Packer" directory name mirrors the canonical packer label.
# These are the values we expect to see as the second path component.
PACKER_DIR_CANONICAL = {
    "Ali": "Ali",
    "Baidu": "Baidu",
    "Bangcle": "Bangcle",
    "Ijiami": "Ijiami",
    "Qihoo": "Qihoo",
    "Tencent": "Tencent",
}

# Canonicalize "<app>_release", "<app>_debug_unaligned", etc. -> "<app>".
# The trailing ``_signed`` was already stripped by the upstream regex.
BUILD_SUFFIX_RE = re.compile(
    r"(_release|_debug_unaligned|_debug|_aligned|_unaligned)+$",
    re.IGNORECASE,
)

# Matches:  features/<api>/<Packer>/<app_base>_<packer_code>_signed.json
ROW_RE = re.compile(
    r"features/(\d+)/([^/]+)/(.+)_(al|bd|bb|ij|qh|tx)_signed\.json\s*$"
)


def canonicalize_app(raw: str) -> str:
    """Strip build-variant suffixes so multiple variants fold into one app.

    Example::
        "2048_release"                 -> "2048"
        "aRevelation_debug_unaligned"  -> "aRevelation"
        "bjtrainer_release"            -> "bjtrainer"
    """
    prev = None
    cur = raw
    # Iteratively strip multiple trailing suffix segments.
    while prev != cur:
        prev = cur
        cur = BUILD_SUFFIX_RE.sub("", cur)
    return cur or raw


def parse_feature_list(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = ROW_RE.search(line)
            if not m:
                continue
            api, packer_dir, app_raw, suffix = m.groups()
            packer_from_dir = PACKER_DIR_CANONICAL.get(packer_dir)
            packer_from_code = SUFFIX_TO_PACKER.get(suffix)
            if packer_from_dir is None or packer_from_code is None:
                continue
            if packer_from_dir != packer_from_code:
                # Consistency check: dir name vs suffix code must agree.
                # Skip the row and emit a note rather than poisoning the index.
                rows.append(
                    {
                        "api": api,
                        "packer": packer_from_dir,
                        "packer_code": packer_from_code,
                        "app_raw": app_raw,
                        "app": canonicalize_app(app_raw),
                        "inconsistent": "dir != suffix",
                    }
                )
                continue
            rows.append(
                {
                    "api": api,
                    "packer": packer_from_dir,
                    "packer_code": packer_from_code,
                    "app_raw": app_raw,
                    "app": canonicalize_app(app_raw),
                    "inconsistent": "",
                }
            )
    return rows


def build_indexes(rows: List[Dict[str, str]]) -> Dict:
    apps: Dict[str, Dict] = defaultdict(
        lambda: {"coverage": defaultdict(set), "api_levels": set(), "packer_count": 0, "n_rows": 0, "raw_names": set()}
    )
    packers: Dict[str, Dict] = defaultdict(lambda: {"apps": set(), "n_rows": 0})
    api_set = set()
    packer_set = set()

    for r in rows:
        if r.get("inconsistent"):
            continue
        app = r["app"]
        apps[app]["coverage"][r["packer"]].add(r["api"])
        apps[app]["api_levels"].add(r["api"])
        apps[app]["n_rows"] += 1
        apps[app]["raw_names"].add(r["app_raw"])

        packers[r["packer"]]["apps"].add(app)
        packers[r["packer"]]["n_rows"] += 1

        api_set.add(r["api"])
        packer_set.add(r["packer"])

    # Finalize: convert sets to sorted lists, compute derived counters.
    by_app = {}
    for name, info in apps.items():
        cov = {pk: sorted(info["coverage"][pk]) for pk in sorted(info["coverage"])}
        by_app[name] = {
            "coverage": cov,
            "api_levels": sorted(info["api_levels"]),
            "packer_count": len(cov),
            "n_rows": info["n_rows"],
            "raw_names": sorted(info["raw_names"]),
        }
    by_packer = {}
    for pk, info in packers.items():
        by_packer[pk] = {
            "apps": sorted(info["apps"]),
            "n_apps": len(info["apps"]),
            "n_rows": info["n_rows"],
        }

    return {
        "packers": sorted(packer_set),
        "api_levels": sorted(api_set),
        "n_unique_apps": len(by_app),
        "n_rows": sum(v["n_rows"] for v in by_app.values()),
        "by_app": by_app,
        "by_packer": by_packer,
    }


def render_markdown(indexes: Dict, inconsistencies: int) -> str:
    lines = []
    lines.append("# PackerGrind feature-dir inventory\n")
    lines.append(
        "Source: `TheSillyStever/adaptiveunpacker` (PackerGrind, "
        "ICSE'17 + TSE'22)\n"
    )
    lines.append(
        f"- Packers: {', '.join(indexes['packers'])}\n"
        f"- Android API levels covered: {', '.join(indexes['api_levels'])}\n"
        f"- Unique canonical apps: {indexes['n_unique_apps']}\n"
        f"- Total feature rows: {indexes['n_rows']}\n"
        f"- Inconsistent rows skipped (packer-dir vs suffix): {inconsistencies}\n"
    )

    lines.append("\n## Coverage by packer\n")
    lines.append("| Packer | unique apps | feature rows |")
    lines.append("|---|---:|---:|")
    for pk in indexes["packers"]:
        row = indexes["by_packer"][pk]
        lines.append(f"| {pk} | {row['n_apps']} | {row['n_rows']} |")

    lines.append("\n## Top 25 apps by packer coverage\n")
    lines.append("| App | #packers | API levels |")
    lines.append("|---|---:|---|")
    top = sorted(
        indexes["by_app"].items(),
        key=lambda kv: (-kv[1]["packer_count"], kv[0]),
    )[:25]
    for name, info in top:
        lines.append(
            f"| `{name}` | {info['packer_count']} | {', '.join(info['api_levels'])} |"
        )

    lines.append("\n## All unique apps (alphabetical)\n")
    lines.append(", ".join(f"`{k}`" for k in sorted(indexes["by_app"])))
    lines.append("\n")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-list",
        type=Path,
        default=Path(".feature_list.txt"),
        help="Path to the ``git ls-tree`` dump of PackerGrind features/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real_world/packergrind"),
        help="Output directory for app_list.{json,csv,md}.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="2c26cc78a139c56f0829437abe5d1b99e5d0eaff",
        help="Upstream commit SHA to record alongside the inventory.",
    )
    args = parser.parse_args(argv)

    if not args.feature_list.exists():
        print(f"[ABORT] feature-list not found: {args.feature_list}", file=sys.stderr)
        return 2

    rows = parse_feature_list(args.feature_list)
    inconsistent = sum(1 for r in rows if r.get("inconsistent"))
    indexes = build_indexes(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "app_list.json"
    payload = {
        "source": "TheSillyStever/adaptiveunpacker (PackerGrind feature dir)",
        "commit": args.commit,
        "inconsistent_rows_skipped": inconsistent,
        **indexes,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.out_dir / "app_list.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["app_canonical", "packer", "api", "app_raw"])
        for r in rows:
            if r.get("inconsistent"):
                continue
            w.writerow([r["app"], r["packer"], r["api"], r["app_raw"]])

    md_path = args.out_dir / "app_list.md"
    md_path.write_text(render_markdown(indexes, inconsistent), encoding="utf-8")

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(
        f"  unique_apps={indexes['n_unique_apps']} "
        f"packers={len(indexes['packers'])} "
        f"api_levels={len(indexes['api_levels'])} "
        f"n_rows={indexes['n_rows']} "
        f"inconsistent={inconsistent}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
