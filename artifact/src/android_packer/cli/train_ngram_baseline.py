"""CLI: train the n-gram + logistic regression byte-level baseline.

Consumes a region training labels JSONL plus an APK index (explicit or
synthetic manifest). Emits a pickled model artefact and a small JSON
training report summarising the fitted coefficients at a glance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.baselines import NgramLogRegConfig, train_ngram_logreg
from android_packer.cli._apk_index import build_apk_index
from android_packer.features import ByteFeatureConfig
from android_packer.utils.jsonl import read_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a byte-level n-gram + logistic regression region classifier "
            "on the training-label JSONL produced by build-training-labels."
        )
    )
    parser.add_argument(
        "--region-labels",
        type=Path,
        required=True,
        help="Training region labels JSONL (apk_id / object_path / offsets / label_id).",
    )
    parser.add_argument(
        "--apk-index",
        type=Path,
        required=True,
        help=(
            "JSONL mapping apk_id to apk_path. Rows with 'generated_apk_path' "
            "are auto-detected as synthetic manifest rows."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs/models/ngram_logreg.pkl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/ngram_logreg_train_report.json"),
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="'none' disables class weighting; 'balanced' is the default.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-bigram", action="store_true")
    parser.add_argument("--no-scalars", action="store_true")
    parser.add_argument("--bigram-hash-dim", type=int, default=1024)
    parser.add_argument("--loader-cache-size", type=int, default=64)
    parser.add_argument(
        "--no-hashing-vectorizer",
        action="store_true",
        help=(
            "Use DictVectorizer instead of the default FeatureHasher. "
            "Only recommended for small-sample equivalence tests; "
            "large datasets (>50k regions) may OOM."
        ),
    )
    parser.add_argument(
        "--hashing-n-features",
        type=int,
        default=262144,
        help="Number of hashing buckets when FeatureHasher is used.",
    )
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

    feature_config = ByteFeatureConfig(
        include_bigram=not args.no_bigram,
        include_scalars=not args.no_scalars,
        bigram_hash_dim=args.bigram_hash_dim,
    )
    config = NgramLogRegConfig(
        feature_config=feature_config,
        C=args.C,
        max_iter=args.max_iter,
        # sklearn expects None rather than the string 'none' to disable.
        class_weight="balanced" if args.class_weight == "balanced" else None,  # type: ignore[arg-type]
        random_state=args.random_state,
        threshold=args.threshold,
        loader_cache_size=args.loader_cache_size,
        use_hashing_vectorizer=not args.no_hashing_vectorizer,
        hashing_n_features=args.hashing_n_features,
    )

    try:
        model = train_ngram_logreg(
            read_jsonl(args.region_labels),
            apk_index,
            config=config,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model.save(args.model_out)

    # Small, human-readable training report. Deliberately excludes the
    # full coefficient vector (thousands of dims) but keeps enough to
    # audit the fit at a glance.
    report = {
        "baseline": "ngram_logreg",
        "model_path": str(args.model_out),
        "apk_index_size": len(apk_index),
        "config": {
            "C": config.C,
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "random_state": config.random_state,
            "threshold": config.threshold,
            "feature_config": {
                "include_unigram": feature_config.include_unigram,
                "include_bigram": feature_config.include_bigram,
                "include_scalars": feature_config.include_scalars,
                "bigram_hash_dim": feature_config.bigram_hash_dim,
                "entropy_chunk_size": feature_config.entropy_chunk_size,
                "duplicate_block_size": feature_config.duplicate_block_size,
            },
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        " ".join(
            [
                f"model_path={args.model_out}",
                f"apk_index_size={len(apk_index)}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
