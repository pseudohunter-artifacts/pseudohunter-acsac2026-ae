"""Summarize APKiD results for Track B v2 strict DPT pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


DEFAULT_PAIRS = Path("outputs/experiments/paired_packed_apks/track_b_v2_dpt_pairs.jsonl")
DEFAULT_OUT = Path("outputs/experiments/track_b_v2_strict_apkid/summary.json")


def roc_auc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    positives = [s for y, s in zip(y_true, y_score) if y == 1]
    negatives = [s for y, s in zip(y_true, y_score) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for ps in positives:
        for ns in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    y_true: List[int] = []
    y_score: List[float] = []
    false_positive = []
    false_negative = []

    for line in args.pairs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        packed_hit = bool(row["apkid_packed_has_packer"])
        unpacked_hit = not bool(row["apkid_unpacked_clean"])

        y_true.extend([0, 1])
        y_score.extend([1.0 if unpacked_hit else 0.0, 1.0 if packed_hit else 0.0])

        if unpacked_hit:
            false_positive.append(row["unpacked_apk"])
        if not packed_hit:
            false_negative.append(row["packed_apk"])

    summary = {
        "baseline": "apkid_packer_or_protector_hit",
        "n_apks": len(y_true),
        "n_packed": sum(y_true),
        "n_benign": len(y_true) - sum(y_true),
        "auroc": roc_auc(y_true, y_score),
        "packed_detection": sum(y_score[i] for i, y in enumerate(y_true) if y == 1),
        "benign_false_positive": sum(y_score[i] for i, y in enumerate(y_true) if y == 0),
        "false_positive_apks": false_positive,
        "false_negative_apks": false_negative,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
