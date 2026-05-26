"""CLI for byte-MLM + grammar-aware pre-training (F-MIL-c).

This thin wrapper exposes :mod:`android_packer.training.pretrain_mlm` via a
stable ``android-packer-pretrain-mlm`` entry point. Two use-modes:

1. ``--dry-run`` (default in this MVP) builds the benign-only corpus,
   runs the MLM collator to produce a representative batch, and writes
   a JSON report covering:

   * corpus statistics (seen / kept / dropped / exclusion ratio);
   * benign-only invariant PASS/FAIL;
   * a small sample of masked positions for human audit.

   This is the CI-safe mode and does NOT require torch.

2. Omitting ``--dry-run`` additionally invokes the torch trainer.  This
   path requires the ``[dl]`` extra and is intentionally separated so
   the dry-run can be validated by reviewers without GPU access.

Default input: a directory tree of classes*.dex files extracted from
benign APKs.  The loader skips files that don't start with the
``dex\\n`` magic so a mixed tree of APK + APK-entry dumps can be pointed
at directly without pre-sorting.

See ``docs/method/ours_method_spec.md`` §5.1, §8, §12.4, and
``docs/research_framing.md`` §4.2 sellpoint 2 for the methodology.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from android_packer.models.tokenizer import ByteTokenizer
from android_packer.training.pretrain_mlm import (
    BenignCorpusError,
    MLMCollator,
    MLMCorpusBuilder,
    PretrainMLMConfig,
)


_DEX_MAGIC = b"dex\n"


def _iter_dex_buffers(root: Path) -> Iterator[bytes]:
    """Yield raw bytes of every file under ``root`` whose first 4 bytes
    look like a DEX magic marker.

    Non-matching files are silently skipped so the caller can point at a
    messy extraction tree.  Files failing to open (permissions, etc.)
    are also skipped after writing a single-line warning to stderr.
    """

    if not root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {root}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                head = fh.read(4)
                if head != _DEX_MAGIC:
                    continue
                rest = fh.read()
        except OSError as exc:
            print(f"[warn] skip unreadable {path}: {exc}", file=sys.stderr)
            continue
        yield head + rest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="android-packer-pretrain-mlm",
        description=(
            "Pre-train the byte-level MLM encoder with a DEX grammar-"
            "aware auxiliary loss on a benign-only DEX corpus "
            "(F-MIL-c)."
        ),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="Directory containing benign classes*.dex dumps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report destination (parents created if missing).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4098,
        help="Token budget per sequence (incl. BOS/EOS). Default 4098.",
    )
    parser.add_argument(
        "--mlm-mask-prob",
        type=float,
        default=0.15,
        help="Per-token masking probability. Default 0.15.",
    )
    parser.add_argument(
        "--item-type-aux-weight",
        type=float,
        default=0.2,
        help=(
            "Weight on L_item_type (sellpoint 2 auxiliary loss). "
            "Set to 0 for the ablation run. Default 0.2."
        ),
    )
    parser.add_argument(
        "--benign-exclusion-max-ratio",
        type=float,
        default=0.05,
        help=(
            "Hard cap on (non-benign / total) buffers. Default 5%. "
            "Exceeding this aborts with BenignCorpusError."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for masking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help=(
            "Only build corpus + collate one batch; do NOT invoke torch "
            "training loop. (Default: enabled.)"
        ),
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Opt in to the full torch training loop (requires [dl]).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4,
        help=(
            "Number of examples included in the sample batch of the "
            "dry-run report. Default 4."
        ),
    )
    return parser


def _run_dry(args: argparse.Namespace) -> int:
    cfg = PretrainMLMConfig(
        max_seq_length=args.max_seq_length,
        mlm_mask_prob=args.mlm_mask_prob,
        item_type_aux_weight=args.item_type_aux_weight,
        benign_exclusion_max_ratio=args.benign_exclusion_max_ratio,
        seed=args.seed,
    )
    tokenizer = ByteTokenizer(max_length=cfg.max_seq_length)
    builder = MLMCorpusBuilder(cfg, tokenizer=tokenizer)

    for buf in _iter_dex_buffers(args.corpus_root):
        builder.add(buf)

    try:
        examples, stats = builder.finalise()
    except BenignCorpusError as exc:
        report = {
            "status": "BENIGN_ONLY_INVARIANT_VIOLATED",
            "error": str(exc),
            "config": asdict(cfg),
        }
        _write_json(args.output, report)
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    collator = MLMCollator(cfg, tokenizer=tokenizer)
    sample = examples[: max(1, args.sample_size)]
    batch = collator.collate(sample)

    # Mask statistics on the sampled batch so reviewers can sanity-check
    # the 80/10/10 split.
    n_supervised = sum(
        1
        for row in batch.labels
        for v in row
        if v != -100  # DEX_ITEM_TYPE_PAD_ID
    )
    n_masked = sum(
        1
        for row in batch.input_ids
        for v in row
        if v == tokenizer.MASK_ID
    )
    total_tokens = sum(sum(row) for row in batch.attention_mask)

    report = {
        "status": "OK",
        "config": asdict(cfg),
        "corpus_stats": asdict(stats),
        "exclusion_ratio": stats.exclusion_ratio,
        "sample_batch": {
            "num_examples": len(sample),
            "total_real_tokens": total_tokens,
            "num_supervised_positions": n_supervised,
            "num_mask_tokens": n_masked,
            "observed_mask_ratio": (n_masked / total_tokens) if total_tokens else 0.0,
        },
    }
    _write_json(args.output, report)
    print(
        f"[ok] kept {stats.kept}/{stats.total_seen} buffers "
        f"(exclusion={stats.exclusion_ratio:.2%}); "
        f"report written to {args.output}"
    )
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run:
        return _run_dry(args)

    # Full torch training loop is out-of-scope for the MVP commit — it
    # would need data shuffling, distributed setup, optimizer config,
    # checkpointing, etc., which we schedule for F-MIL-c stage 2.  Fail
    # loudly instead of silently doing the wrong thing.
    print(
        "[error] full torch training loop is not wired up yet; "
        "re-run with --dry-run (default) for corpus + collator checks. "
        "See docs/workstreams/track_b/tasks.md::F-MIL-c.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
