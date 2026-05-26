"""CLI: score region training labels with a trained Ours model — F-MIL-e."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.baselines import OursBaselineModel
from android_packer.cli._apk_index import build_apk_index
from android_packer.utils.jsonl import read_jsonl, write_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score region training labels with a trained Ours "
            "(Typed-Instance MIL) baseline model."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--region-labels", type=Path, required=True)
    parser.add_argument("--apk-index", type=Path, required=True)
    parser.add_argument(
        "--region-predictions-out",
        type=Path,
        default=Path("outputs/predictions/ours.region_predictions.jsonl"),
    )
    parser.add_argument(
        "--object-predictions-out",
        type=Path,
        default=Path("outputs/predictions/ours.object_predictions.jsonl"),
    )
    parser.add_argument(
        "--apk-predictions-out",
        type=Path,
        default=Path("outputs/predictions/ours.apk_predictions.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/ours_baseline_report.json"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    model = OursBaselineModel.load(args.model)
    apk_index = build_apk_index(args.apk_index)
    if not apk_index:
        print(
            f"error: apk index at {args.apk_index} is empty; cannot score.",
            file=sys.stderr,
        )
        return 2

    result = model.predict(read_jsonl(args.region_labels), apk_index)

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

    print(
        " ".join(
            [
                f"region_predictions={region_count}",
                f"object_predictions={object_count}",
                f"apk_predictions={apk_count}",
                f"counts={result.report['counts']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
