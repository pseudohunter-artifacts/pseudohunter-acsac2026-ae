"""CLI: train an Ours (Typed-Instance MIL) baseline — F-MIL-e stage 2.

Mirrors ``train_ngram_baseline.py`` / ``train_payload_hunter_lite.py``.
Reads region training labels + an APK index, fits
:func:`android_packer.baselines.ours.train_ours_baseline`, and saves the
resulting :class:`OursBaselineModel` to disk.

The model artefact is a single ``.pt`` file loadable by
``android-packer-run-ours``.

See ``docs/method/ours_method_spec.md`` §12.6 batch F-MIL-e.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.baselines import OursBaselineConfig, train_ours_baseline
from android_packer.cli._apk_index import build_apk_index
from android_packer.utils.jsonl import read_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an Ours (Typed-Instance MIL) baseline on region "
            "training labels and save the checkpoint."
        )
    )
    parser.add_argument("--region-labels", type=Path, required=True)
    parser.add_argument("--apk-index", type=Path, required=True)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs/models/ours_baseline.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-diff-pseudo", type=float, default=0.3)
    parser.add_argument("--lambda-sparsity", type=float, default=0.01)
    parser.add_argument("--bag-pos-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--mil-pooling",
        choices=("topk", "noisy_or", "attention"),
        default="attention",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    apk_index = build_apk_index(args.apk_index)
    if not apk_index:
        print(
            f"error: apk index at {args.apk_index} is empty; cannot train.",
            file=sys.stderr,
        )
        return 2

    from android_packer.models.ours import OursConfig
    from android_packer.models.typed_encoder import TypedEncoderConfig

    cfg = OursBaselineConfig(
        ours_config=OursConfig(
            typed=TypedEncoderConfig(),
            mil_pooling=args.mil_pooling,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_diff_pseudo=args.lambda_diff_pseudo,
        lambda_sparsity=args.lambda_sparsity,
        bag_pos_weight=args.bag_pos_weight,
        threshold=args.threshold,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    model = train_ours_baseline(list(read_jsonl(args.region_labels)), apk_index, cfg)
    model.save(args.model_out)

    print(f"saved Ours model to {args.model_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
