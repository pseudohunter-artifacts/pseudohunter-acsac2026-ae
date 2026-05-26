"""CLI: run the APKiD reference baseline over a list of APKs.

Input JSONL can be either:

1. An explicit entries file with ``apk_id``, ``apk_path``, ``true_label_id``
   per row.
2. A synthetic manifest produced by :mod:`android_packer.cli.generate_packed_apk`,
   in which case ``generated_apk_id`` and ``generated_apk_path`` are read
   and ``true_label_id`` defaults to 1 (every row is a packed APK).

The two shapes are detected heuristically: if the row carries
``generated_apk_path`` it is treated as a manifest row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from android_packer.baselines import (
    ApkidBaselineConfig,
    ApkidNotInstalledError,
    run_apkid_baseline,
)
from android_packer.utils.jsonl import read_jsonl, write_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score APKs with the APKiD reference baseline. APKiD is a "
            "YARA-backed Android packer fingerprint tool; this baseline "
            "reports APK-level predictions only."
        )
    )
    parser.add_argument(
        "--apk-entries",
        type=Path,
        required=True,
        help=(
            "JSONL file of APK entries or a synthetic manifest. Rows with "
            "'generated_apk_path' are auto-detected as manifest rows."
        ),
    )
    parser.add_argument(
        "--apk-predictions-out",
        type=Path,
        default=Path("outputs/predictions/apkid.apk_predictions.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/apkid_baseline_report.json"),
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=1,
        help="Predict positive when at least this many (category, family) matches are found.",
    )
    parser.add_argument(
        "--include-aux-categories",
        action="store_true",
        help="Count anti_vm / anti_disassembly / anti_debug categories as positive evidence.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-APK upper bound for the APKiD scan.",
    )
    return parser.parse_args(argv)


def _coerce_entry(row: Mapping) -> dict:
    """Map either shape into the ``{apk_id, apk_path, true_label_id}`` dict."""

    if "generated_apk_path" in row:
        return {
            "apk_id": str(row.get("generated_apk_id") or row.get("apk_id", "")),
            "apk_path": str(row["generated_apk_path"]),
            # Synthetic manifests only record packed APKs; label them positive.
            "true_label_id": 1,
        }
    return {
        "apk_id": str(row["apk_id"]),
        "apk_path": str(row["apk_path"]),
        "true_label_id": int(row.get("true_label_id", 0)),
    }


def _load_entries(path: Path) -> Iterable[dict]:
    for row in read_jsonl(path):
        yield _coerce_entry(row)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    config = ApkidBaselineConfig(
        include_aux_categories=args.include_aux_categories,
        min_hits=args.min_hits,
        timeout_seconds=args.timeout_seconds,
    )

    try:
        result = run_apkid_baseline(_load_entries(args.apk_entries), config=config)
    except ApkidNotInstalledError as exc:
        # Surface the friendly install hint to the user rather than a
        # bare Python traceback. Exit code 2 keeps this distinguishable
        # from a genuine runtime failure (exit 1).
        print(f"error: {exc}", file=sys.stderr)
        return 2

    apk_count = write_jsonl(
        args.apk_predictions_out,
        (row.to_dict() for row in result.apk_predictions),
    )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = result.report["metrics"]["apk"]
    print(
        " ".join(
            [
                f"apk_predictions={apk_count}",
                f"apk_f1={metrics['f1']}",
                f"scan_failures={result.report['counts']['scan_failures']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
