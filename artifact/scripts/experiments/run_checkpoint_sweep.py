"""Sweep all pretrain checkpoints through BERT-only LOPO evaluation.

Run after pretraining completes to find the optimal epoch for 8L/512d BERT.
Tests each checkpoint in outputs/experiments/pseudo_bert_v3/checkpoints/*.pt.

Usage:
    python scripts/experiments/run_checkpoint_sweep.py [--device cuda] [--epochs 50]

Output:
    outputs/experiments/pseudo_bert_v3/checkpoint_sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "experiments" / "pseudo_bert_v3" / "checkpoints"
OUT_DIR = ROOT / "outputs" / "experiments" / "pseudo_bert_v3"


def find_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Find all epoch_NNN.pt files, sorted by epoch number."""
    pts = sorted(ckpt_dir.glob("epoch_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    return pts


def run_lopo_for_checkpoint(ckpt_path: Path, device: str, epochs: int,
                            bert_layers: int, bert_dim: int,
                            androzoo_benign: int) -> dict:
    """Run LOPO eval with a specific checkpoint, return results."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "experiments" / "run_lopo_eval.py"),
        "--pretrain-ckpt", str(ckpt_path),
        "--ablation", "bert_only",
        "--bert-layers", str(bert_layers),
        "--bert-dim", str(bert_dim),
        "--androzoo-benign", str(androzoo_benign),
        "--epochs", str(epochs),
        "--device", device,
    ]

    print(f"\n{'='*60}")
    print(f"  Testing: {ckpt_path.name}")
    print(f"  Command: {' '.join(cmd[-10:])}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.time() - t0

    # Parse LOPO results from the output JSON
    # The script saves results to outputs/experiments/lopo_eval/
    lopo_dir = OUT_DIR.parent / "lopo_eval"
    results_file = lopo_dir / "lopo_results.json"

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        return {
            "checkpoint": ckpt_path.name,
            "epoch": int(ckpt_path.stem.split("_")[1]),
            "status": "FAILED",
            "error": result.stderr[-500:] if result.stderr else "unknown",
            "elapsed_sec": elapsed,
        }

    # Try to load results
    try:
        with open(results_file) as f:
            lopo_results = json.load(f)
        mean_auroc = lopo_results.get("mean_auroc", 0)
        per_packer = lopo_results.get("per_packer", {})
    except Exception as e:
        mean_auroc = 0
        per_packer = {}
        print(f"  WARNING: Could not parse results: {e}")

    # Also check for localization metrics
    entry_mrr = lopo_results.get("mean_entry_mrr_normality", 0) if 'lopo_results' in dir() else 0

    return {
        "checkpoint": ckpt_path.name,
        "epoch": int(ckpt_path.stem.split("_")[1]),
        "status": "OK",
        "mean_auroc": mean_auroc,
        "entry_mrr": entry_mrr,
        "per_packer": per_packer,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Fine-tune epochs per fold")
    parser.add_argument("--bert-layers", type=int, default=8)
    parser.add_argument("--bert-dim", type=int, default=512)
    parser.add_argument("--androzoo-benign", type=int, default=50)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Specific checkpoint filenames (default: all)")
    args = parser.parse_args()

    checkpoints = find_checkpoints(CKPT_DIR)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {CKPT_DIR}")
        return

    if args.checkpoints:
        # Filter to specific checkpoints
        checkpoints = [c for c in checkpoints if c.name in args.checkpoints]

    print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        print(f"  {c.name} ({c.stat().st_size / 1e6:.1f} MB)")

    # Run LOPO for each checkpoint
    all_results = []
    for ckpt in checkpoints:
        result = run_lopo_for_checkpoint(
            ckpt, args.device, args.epochs,
            args.bert_layers, args.bert_dim, args.androzoo_benign
        )
        all_results.append(result)
        print(f"\n  → {ckpt.name}: AUROC={result.get('mean_auroc', 'N/A')}, "
              f"MRR={result.get('entry_mrr', 'N/A')}, "
              f"time={result.get('elapsed_sec', 0):.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("  CHECKPOINT SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Epoch':<8}{'AUROC':<10}{'MRR':<10}{'Status':<10}{'Time':<10}")
    print("-" * 48)

    best_auroc = 0
    best_ckpt = None
    for r in all_results:
        auroc = r.get("mean_auroc", 0)
        mrr = r.get("entry_mrr", 0)
        print(f"{r['epoch']:<8}{auroc:<10.4f}{mrr:<10.4f}{r['status']:<10}"
              f"{r.get('elapsed_sec',0):<10.0f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_ckpt = r["checkpoint"]

    print(f"\n  BEST: {best_ckpt} (AUROC={best_auroc:.4f})")

    # Save results
    out_path = OUT_DIR / "checkpoint_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "bert_layers": args.bert_layers,
                "bert_dim": args.bert_dim,
                "finetune_epochs": args.epochs,
                "androzoo_benign": args.androzoo_benign,
                "ablation": "bert_only",
            },
            "results": all_results,
            "best_checkpoint": best_ckpt,
            "best_auroc": best_auroc,
        }, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
