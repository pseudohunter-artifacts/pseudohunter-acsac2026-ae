"""CLI: run an entropy-threshold baseline over region training labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.baselines import EntropyBaselineConfig, run_entropy_baseline
from android_packer.utils.jsonl import read_jsonl, write_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score region labels with an entropy-threshold baseline."
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
        default=Path("outputs/predictions/entropy.region_predictions.jsonl"),
    )
    parser.add_argument(
        "--object-predictions-out",
        type=Path,
        default=Path("outputs/predictions/entropy.object_predictions.jsonl"),
    )
    parser.add_argument(
        "--apk-predictions-out",
        type=Path,
        default=Path("outputs/predictions/entropy.apk_predictions.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/entropy_baseline_report.json"),
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=7.0,
        help="Predict positive when the configured region score is at least this value.",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=1.0,
        help="Weight applied to the raw entropy feature.",
    )
    parser.add_argument(
        "--nonprintable-weight",
        type=float,
        default=0.0,
        help="Optional weight for 1 - printable_ratio.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    result = run_entropy_baseline(
        read_jsonl(args.region_labels),
        config=EntropyBaselineConfig(
            entropy_threshold=args.entropy_threshold,
            entropy_weight=args.entropy_weight,
            nonprintable_weight=args.nonprintable_weight,
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
