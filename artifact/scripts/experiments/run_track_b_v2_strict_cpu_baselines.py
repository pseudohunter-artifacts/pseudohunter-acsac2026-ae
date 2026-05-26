"""CPU baselines for Track B v2 strict app-disjoint DPT.

This script evaluates non-neural APK-level scores on the same 20 DPT-packed
APKs and 20 benign counterparts used by ``run_lopo_eval.py
--track-b-v2-strict``. It is intentionally CPU-only and training-free, so it
can run while GPU experiments are active.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android_packer.apkio.objects import iter_apk_objects  # noqa: E402


TRACK_B_V2 = ROOT / "data" / "real_world" / "track_b_v2"
DEFAULT_OUT = ROOT / "outputs" / "experiments" / "track_b_v2_strict_cpu_baselines"


@dataclass(frozen=True)
class ApkScore:
    apk_id: str
    apk_path: str
    label: int
    scores: Dict[str, float]
    n_entries: int


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def entry_scores(apk_path: Path, *, max_depth: int = 1) -> Dict[str, float]:
    entropies: List[float] = []
    weighted: List[float] = []
    sizes: List[int] = []
    dex_like = 0
    so_like = 0
    asset_like = 0

    for meta, blob in iter_apk_objects(apk_path, max_depth=max_depth):
        if len(blob) < 64:
            continue
        path = meta.object_path.lower()
        ent = byte_entropy(blob)
        size = len(blob)
        entropies.append(ent)
        weighted.append(ent * math.log2(max(size, 2)))
        sizes.append(size)
        if path.endswith(".dex") or path.startswith("classes"):
            dex_like += 1
        if path.endswith(".so"):
            so_like += 1
        if path.startswith("assets/"):
            asset_like += 1

    if not entropies:
        return {
            "max_entropy": 0.0,
            "mean_top3_entropy": 0.0,
            "max_weighted_entropy": 0.0,
            "max_entry_size_log2": 0.0,
            "apk_size_log2": math.log2(max(apk_path.stat().st_size, 1)),
            "n_entries_log2": 0.0,
            "dex_entry_count": 0.0,
            "so_entry_count": 0.0,
            "asset_entry_count": 0.0,
        }

    top3 = sorted(entropies, reverse=True)[:3]
    return {
        "max_entropy": max(entropies) / 8.0,
        "mean_top3_entropy": (sum(top3) / len(top3)) / 8.0,
        "max_weighted_entropy": max(weighted),
        "max_entry_size_log2": math.log2(max(sizes)),
        "apk_size_log2": math.log2(max(apk_path.stat().st_size, 1)),
        "n_entries_log2": math.log2(len(entropies) + 1),
        "dex_entry_count": float(dex_like),
        "so_entry_count": float(so_like),
        "asset_entry_count": float(asset_like),
    }


def roc_auc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    positives = [s for y, s in zip(y_true, y_score) if y == 1]
    negatives = [s for y, s in zip(y_true, y_score) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for ps in positives:
        for ns in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def collect_apks(limit: int = 0) -> List[tuple[Path, int]]:
    benign = sorted((TRACK_B_V2 / "benign").glob("*.apk"))
    packed = sorted((TRACK_B_V2 / "packed" / "dpt_shell").glob("*.apk"))
    if limit:
        benign = benign[:limit]
        packed = packed[:limit]
    return [(p, 0) for p in benign] + [(p, 1) for p in packed]


def evaluate(rows: Iterable[ApkScore]) -> Dict[str, object]:
    rows = list(rows)
    y_true = [r.label for r in rows]
    score_names = sorted({name for r in rows for name in r.scores})
    metrics = {}
    for name in score_names:
        values = [r.scores[name] for r in rows]
        auc = roc_auc(y_true, values)
        packed = [v for y, v in zip(y_true, values) if y == 1]
        benign = [v for y, v in zip(y_true, values) if y == 0]
        metrics[name] = {
            "auroc": auc,
            "packed_mean": sum(packed) / len(packed),
            "benign_mean": sum(benign) / len(benign),
        }
    return {
        "n_apks": len(rows),
        "n_packed": sum(y_true),
        "n_benign": len(y_true) - sum(y_true),
        "metrics": metrics,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows: List[ApkScore] = []
    apks = collect_apks(args.limit)
    print(f"Track B v2 strict CPU baselines: {len(apks)} APKs", flush=True)
    for i, (apk, label) in enumerate(apks, 1):
        scores = entry_scores(apk)
        rows.append(
            ApkScore(
                apk_id=apk.stem,
                apk_path=str(apk),
                label=label,
                scores=scores,
                n_entries=int(2 ** scores["n_entries_log2"] - 1),
            )
        )
        if i % 5 == 0 or i == len(apks):
            print(f"  processed {i}/{len(apks)}", flush=True)

    summary = evaluate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "apk_scores.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== APK AUROC ===")
    for name, metric in sorted(summary["metrics"].items()):
        print(
            f"{name:22s} AUROC={metric['auroc']:.4f} "
            f"packed_mean={metric['packed_mean']:.4f} "
            f"benign_mean={metric['benign_mean']:.4f}"
        )
    print(f"\nSaved: {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
