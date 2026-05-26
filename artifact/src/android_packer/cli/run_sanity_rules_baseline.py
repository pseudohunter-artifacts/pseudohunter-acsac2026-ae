"""CLI: run the sanity-check heuristic baseline over region training labels.

This baseline is an internal sanity check, not the reported rule baseline
for the paper (APKiD is). See :mod:`android_packer.baselines.sanity_rules`
for rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.baselines import SanityRulesConfig, run_sanity_rules_baseline
from android_packer.utils.jsonl import read_jsonl, write_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score region labels with a static heuristic baseline for "
            "internal sanity checking. NOT the reported rule baseline; "
            "see run_apkid_baseline for that."
        )
    )
    parser.add_argument(
        "--region-labels",
        type=Path,
        required=True,
        help="Region training labels JSONL produced by build-training-labels.",
    )
    parser.add_argument(
        "--region-predictions-out",
        type=Path,
        default=Path("outputs/predictions/sanity_rules.region_predictions.jsonl"),
    )
    parser.add_argument(
        "--object-predictions-out",
        type=Path,
        default=Path("outputs/predictions/sanity_rules.object_predictions.jsonl"),
    )
    parser.add_argument(
        "--apk-predictions-out",
        type=Path,
        default=Path("outputs/predictions/sanity_rules.apk_predictions.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/sanity_rules_baseline_report.json"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Predict positive when the sum of triggered rule weights is at least this value.",
    )
    parser.add_argument(
        "--suspicious-path-weight",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--unknown-extension-weight",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--large-object-weight",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--low-printable-weight",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--min-large-bytes",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--low-printable-threshold",
        type=float,
        default=0.1,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    result = run_sanity_rules_baseline(
        read_jsonl(args.region_labels),
        config=SanityRulesConfig(
            threshold=args.threshold,
            suspicious_path_weight=args.suspicious_path_weight,
            unknown_extension_weight=args.unknown_extension_weight,
            large_object_weight=args.large_object_weight,
            low_printable_weight=args.low_printable_weight,
            min_large_bytes=args.min_large_bytes,
            low_printable_threshold=args.low_printable_threshold,
        ),
    )

    region_count = write_jsonl(
        args.region_predictions_out,
        (row.to_dict() for row in result.region_predictions),
    )
    object_count = write_jsonl(
        args.object_predictions_out,
        (row.to_dict() for row in result.object_predictions),
    )
    apk_count = write_jsonl(
        args.apk_predictions_out,
        (row.to_dict() for row in result.apk_predictions),
    )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = result.report["metrics"]
    print(
        " ".join(
            [
                f"region_predictions={region_count}",
                f"object_predictions={object_count}",
                f"apk_predictions={apk_count}",
                f"region_f1={metrics['region']['f1']}",
                f"object_f1={metrics['object']['f1']}",
                f"apk_f1={metrics['apk']['f1']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
