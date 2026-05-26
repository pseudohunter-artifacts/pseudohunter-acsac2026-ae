"""B2 calibration and bootstrap diagnostics for strict DPT-v2 runs.

This script reloads existing strict-DPT checkpoints, re-evaluates the strict
Track B v2 test set, and reports calibration / bootstrap statistics. It is a
measurement tool only: it does not tune thresholds or feed test labels back
into model design.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from android_packer.decoders.pseudo_tokenizer import BYTE_REPRESENTATION_LEGACY_RAW
from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES
from scripts.experiments.run_lopo_eval import (
    BAG_CACHE_DIR,
    PseudoCodeTokenizer,
    _torch_load_local,
    build_model,
    fixed_fpr_tpr_metrics,
    load_track_b_v2_strict_data,
)


STRICT_DIR = ROOT / "outputs" / "experiments" / "track_b_v2_strict_dpt"
DEFAULT_OUT = STRICT_DIR / "b2_calibration_bootstrap_summary.json"

DEFAULT_RUNS: Tuple[Tuple[str, Path], ...] = (
    ("B0_A_no_hard_benign", STRICT_DIR / "results_strict_dpt_clean_lowbyte025.json"),
    (
        "hard_benign_9",
        STRICT_DIR / "results_strict_dpt_clean_hardbenign_lowbyte025.json",
    ),
    (
        "hard_benign_fdroid24",
        STRICT_DIR / "results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json",
    ),
    (
        "hard_benign_androzoo50",
        STRICT_DIR / "results_strict_dpt_clean_hardbenign_androzoo50_lowbyte025.json",
    ),
    ("B1_1_A_non_dpt", STRICT_DIR / "results_strict_dpt_b11_A_non_dpt_b32_c128.json"),
    ("B1_1_B_add_old_dpt", STRICT_DIR / "results_strict_dpt_b11_B_add_old_dpt_b32_c128.json"),
    (
        "B1_1_C_add_old_dpt_benign",
        STRICT_DIR / "results_strict_dpt_b11_C_add_old_dpt_benign_b32_c128.json",
    ),
    (
        "B1_1_D_other_positive_replay",
        STRICT_DIR / "results_strict_dpt_b11_D_other_positive_replay_b32_c128.json",
    ),
)

DEFAULT_PAIR_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("B1_1_A_non_dpt", "B1_1_B_add_old_dpt"),
    ("B1_1_B_add_old_dpt", "B1_1_C_add_old_dpt_benign"),
    ("B1_1_A_non_dpt", "B1_1_D_other_positive_replay"),
    ("B1_1_D_other_positive_replay", "B1_1_C_add_old_dpt_benign"),
    ("hard_benign_9", "hard_benign_fdroid24"),
    ("hard_benign_fdroid24", "hard_benign_androzoo50"),
)


def result_checkpoint_path(result_path: Path) -> Path:
    return result_path.parent / "checkpoints" / result_path.stem / "strict_dpt" / "latest.pt"


def load_result(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def args_from_result_config(config: Mapping) -> SimpleNamespace:
    active_paths = config.get("active_paths", ("dalvik", "arm64", "byte"))
    return SimpleNamespace(
        bert_dim=int(config.get("bert_dim", 512)),
        bert_layers=int(config.get("bert_layers", 8)),
        ablation=str(config.get("ablation", "bert_only")),
        paths=tuple(active_paths),
        path_dropout=float(config.get("path_dropout", 0.0)),
        region_type_routing=bool(config.get("region_type_routing", False)),
        no_batch_bert_streams=not bool(config.get("batch_bert_streams", True)),
        routing_dex_byte_weight=float(config.get("routing_dex_byte_weight", 0.25)),
        routing_elf_byte_weight=float(config.get("routing_elf_byte_weight", 0.25)),
        routing_byte_entry_weight=float(config.get("routing_byte_entry_weight", 1.0)),
        routing_unknown_weight=float(config.get("routing_unknown_weight", 0.25)),
        byte_representation=str(
            config.get("byte_representation", BYTE_REPRESENTATION_LEGACY_RAW)
        ),
    )


def normalize_score(score: float, norm_info: Mapping) -> Optional[float]:
    mode = norm_info.get("mode")
    if mode == "train_benign_center":
        return float(score - float(norm_info["train_benign_mean"]))
    if mode == "train_benign_z":
        denom = float(norm_info.get("train_benign_std", 1.0))
        if denom <= 1e-6:
            denom = 1.0
        return float((score - float(norm_info["train_benign_mean"])) / denom)
    return None


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    return float(np.mean([(float(s) - int(y)) ** 2 for y, s in zip(labels, scores)]))


def expected_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels_arr = np.asarray(labels, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    total = len(labels_arr)
    rows: List[Dict[str, float]] = []
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (scores_arr >= lo) & (scores_arr <= hi)
        else:
            mask = (scores_arr >= lo) & (scores_arr < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        confidence = float(scores_arr[mask].mean())
        accuracy = float(labels_arr[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (count / total) * gap
        rows.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return float(ece), rows


def best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in sorted(set(scores_arr.tolist()), reverse=True):
        pred = scores_arr >= threshold
        tp = int(((labels_arr == 1) & pred).sum())
        fp = int(((labels_arr == 0) & pred).sum())
        fn = int(((labels_arr == 1) & (~pred)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def threshold_for_target_tpr(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    target_tpr: float = 0.95,
) -> Optional[Dict[str, float]]:
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = sum(1 for label in labels if int(label) == 0)
    if positives == 0 or negatives == 0:
        return None
    candidates = []
    for threshold in sorted(set(float(s) for s in scores), reverse=True):
        tp = fp = 0
        for label, score in zip(labels, scores):
            if float(score) >= threshold:
                if int(label) == 1:
                    tp += 1
                else:
                    fp += 1
        tpr = tp / positives
        fpr = fp / negatives
        if tpr >= target_tpr:
            candidates.append((fpr, threshold, tpr))
    if not candidates:
        return None
    fpr, threshold, tpr = min(candidates, key=lambda item: (item[0], -item[1]))
    return {"threshold": float(threshold), "tpr": float(tpr), "fpr": float(fpr)}


def binary_summary(labels: Sequence[int], scores: Sequence[float], *, n_bins: int) -> Dict:
    if len(set(int(v) for v in labels)) < 2:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    ece, bins = expected_calibration_error(labels, scores, n_bins=n_bins)
    return {
        "auroc": auroc,
        "auprc": auprc,
        **fixed_fpr_tpr_metrics(labels, scores),
        "brier": brier_score(labels, scores),
        "ece": ece,
        "ece_bins": bins,
        "best_f1": best_f1_threshold(labels, scores),
        "threshold_at_95_tpr": threshold_for_target_tpr(labels, scores, target_tpr=0.95),
    }


def _metric_value(labels: Sequence[int], scores: Sequence[float], metric: str) -> Optional[float]:
    if len(set(int(v) for v in labels)) < 2 and metric in {"auroc", "auprc"}:
        return None
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    if metric in {"fpr_at_95_tpr", "tpr_at_1_fpr", "tpr_at_5_fpr"}:
        value = fixed_fpr_tpr_metrics(labels, scores)[metric]
        return None if value is None else float(value)
    if metric == "brier":
        return brier_score(labels, scores)
    if metric == "ece":
        return expected_calibration_error(labels, scores, n_bins=10)[0]
    if metric == "best_f1_threshold":
        return best_f1_threshold(labels, scores)["threshold"]
    raise ValueError(f"unknown metric: {metric}")


def percentile_ci(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p50": float(np.percentile(arr, 50.0)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
    }


def bootstrap_cis(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    metrics: Sequence[str],
) -> Dict[str, Optional[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(labels_arr)
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels_arr[idx].tolist()
        sample_scores = scores_arr[idx].tolist()
        if len(set(sample_labels)) < 2:
            continue
        for metric in metrics:
            value = _metric_value(sample_labels, sample_scores, metric)
            if value is not None and math.isfinite(value):
                values[metric].append(float(value))
    return {metric: percentile_ci(metric_values) for metric, metric_values in values.items()}


def paired_bootstrap_delta(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Dict:
    if len(labels) != len(scores_a) or len(labels) != len(scores_b):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    a_arr = np.asarray(scores_a, dtype=np.float64)
    b_arr = np.asarray(scores_b, dtype=np.float64)
    n = len(labels_arr)
    deltas: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels_arr[idx].tolist()
        if len(set(sample_labels)) < 2:
            continue
        va = _metric_value(sample_labels, a_arr[idx].tolist(), metric)
        vb = _metric_value(sample_labels, b_arr[idx].tolist(), metric)
        if va is None or vb is None:
            continue
        deltas.append(float(vb - va))
    ci = percentile_ci(deltas)
    if ci is None:
        return {"metric": metric, "delta_b_minus_a": None, "p_two_sided": None}
    arr = np.asarray(deltas, dtype=np.float64)
    p_le_zero = float(np.mean(arr <= 0.0))
    p_ge_zero = float(np.mean(arr >= 0.0))
    return {
        "metric": metric,
        "delta_b_minus_a": ci,
        "p_two_sided": min(1.0, 2.0 * min(p_le_zero, p_ge_zero)),
    }


def summarize_scores(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def entry_type_name(bag: Mapping, entry_idx: int) -> str:
    boundaries = bag.get("entry_boundaries", [])
    entry_type_ids = bag.get("entry_type_ids")
    if entry_idx >= len(boundaries) or entry_type_ids is None:
        return "unknown"
    start, end = boundaries[entry_idx]
    ids = np.asarray(entry_type_ids[start:end], dtype=np.int64)
    if ids.size == 0:
        return "unknown"
    counts = np.bincount(ids)
    type_id = int(np.argmax(counts))
    if 0 <= type_id < len(ENTRY_COARSE_TYPES):
        return ENTRY_COARSE_TYPES[type_id]
    return "unknown"


def evaluate_result(
    name: str,
    result_path: Path,
    test_bags: Sequence[Mapping],
    device: torch.device,
    *,
    n_bins: int,
) -> Dict:
    result = load_result(result_path)
    config = dict(result.get("config", {}))
    args = args_from_result_config(config)
    checkpoint_path = result_checkpoint_path(result_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found for {name}: {checkpoint_path}")

    model = build_model(args)
    ckpt = _torch_load_local(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    chunk_size = int(config.get("bag_chunk_size", 64))
    norm_info = config.get("score_normalization_info", {})

    predictions: List[Dict] = []
    entry_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for index, bag in enumerate(test_bags):
            out = model.forward_bag(bag, device, chunk_size=chunk_size)
            score = float(torch.sigmoid(out["bag_logit"]).item())
            norm_score = normalize_score(score, norm_info) if isinstance(norm_info, Mapping) else None
            label = int(bag["apk_label"])
            predictions.append(
                {
                    "index": index,
                    "apk_id": str(bag.get("apk_id", "")),
                    "label": label,
                    "raw_score": score,
                    "normalized_score": norm_score,
                }
            )

            entry_indices = list(out.get("entry_indices", range(len(out["entry_attention"]))))
            attention = out["entry_attention"].detach().cpu().numpy()
            anomaly = (1.0 - out["entry_normality"]).detach().cpu().numpy()
            suspicion = out["entry_suspicion"].detach().cpu().numpy()
            entry_prob = torch.sigmoid(out["entry_logits"]).detach().cpu().numpy()
            contribution = attention * entry_prob
            for out_idx, original_idx in enumerate(entry_indices):
                entry_type = entry_type_name(bag, int(original_idx))
                prefix = f"{'packed' if label else 'benign'}::{entry_type}"
                entry_scores[prefix]["attention"].append(float(attention[out_idx]))
                entry_scores[prefix]["anomaly"].append(float(anomaly[out_idx]))
                entry_scores[prefix]["suspicion"].append(float(suspicion[out_idx]))
                entry_scores[prefix]["apk_contribution"].append(float(contribution[out_idx]))

    labels = [int(row["label"]) for row in predictions]
    raw_scores = [float(row["raw_score"]) for row in predictions]
    norm_scores = [
        float(row["normalized_score"])
        for row in predictions
        if row["normalized_score"] is not None
    ]
    score_distributions = {
        "raw_by_label": {
            "benign": summarize_scores([s for y, s in zip(labels, raw_scores) if y == 0]),
            "packed": summarize_scores([s for y, s in zip(labels, raw_scores) if y == 1]),
        }
    }
    if len(norm_scores) == len(raw_scores):
        score_distributions["normalized_by_label"] = {
            "benign": summarize_scores(
                [s for y, s in zip(labels, norm_scores) if y == 0]
            ),
            "packed": summarize_scores(
                [s for y, s in zip(labels, norm_scores) if y == 1]
            ),
        }

    entry_type_distributions = {
        key: {metric: summarize_scores(values) for metric, values in metric_values.items()}
        for key, metric_values in sorted(entry_scores.items())
    }

    return {
        "name": name,
        "result_path": str(result_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "n_predictions": len(predictions),
        "labels": labels,
        "raw_scores": raw_scores,
        "predictions": predictions,
        "raw_metrics": binary_summary(labels, raw_scores, n_bins=n_bins),
        "score_distributions": score_distributions,
        "entry_type_score_distributions": entry_type_distributions,
        "config": config,
    }


def write_markdown(summary: Mapping, path: Path) -> None:
    lines = [
        "# Strict DPT-v2 B2 Calibration / Bootstrap",
        "",
        "Diagnostic only: no threshold tuning or method design uses these test labels.",
        "",
        "## Runs",
        "",
        "| run | AUROC | AUROC 95% CI | AUPRC | ECE | Brier | FPR@95TPR | TPR@1%/5%FPR |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for name, run in summary["runs"].items():
        metrics = run["raw_metrics"]
        cis = run["bootstrap_ci"]
        auroc_ci = cis.get("auroc") or {}
        ci_text = (
            f"[{auroc_ci.get('p2_5', 0.0):.4f}, {auroc_ci.get('p97_5', 0.0):.4f}]"
            if auroc_ci
            else "n/a"
        )
        lines.append(
            "| {name} | {auroc:.4f} | {ci} | {auprc:.4f} | {ece:.4f} | "
            "{brier:.4f} | {fpr:.4f} | {tpr1:.4f} / {tpr5:.4f} |".format(
                name=name,
                auroc=metrics["auroc"],
                ci=ci_text,
                auprc=metrics["auprc"],
                ece=metrics["ece"],
                brier=metrics["brier"],
                fpr=metrics["fpr_at_95_tpr"],
                tpr1=metrics["tpr_at_1_fpr"],
                tpr5=metrics["tpr_at_5_fpr"],
            )
        )

    lines.extend(["", "## Paired Bootstrap", ""])
    lines.append("| comparison | metric | delta B-A mean | 95% CI | p(two-sided) |")
    lines.append("|---|---|---:|---|---:|")
    for row in summary["paired_comparisons"]:
        for metric, payload in row["metrics"].items():
            delta = payload["delta_b_minus_a"]
            if delta is None:
                lines.append(f"| {row['a']} -> {row['b']} | {metric} | n/a | n/a | n/a |")
                continue
            lines.append(
                "| {a} -> {b} | {metric} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | {p:.4f} |".format(
                    a=row["a"],
                    b=row["b"],
                    metric=metric,
                    mean=delta["mean"],
                    lo=delta["p2_5"],
                    hi=delta["p97_5"],
                    p=payload["p_two_sided"],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_run_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must be name=path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    return name, path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run_spec, default=[],
                        help="Run spec name=path. Defaults to key strict-DPT rows.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-bag-cache", action="store_true")
    parser.add_argument("--exclude-apkid-dirty-strict-benign", action="store_true",
                        default=True)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    runs = args.run or list(DEFAULT_RUNS)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    bag_cache_dir = None if args.no_bag_cache else BAG_CACHE_DIR
    run_configs = [load_result(path).get("config", {}) for _, path in runs]
    byte_representations = {
        str(config.get("byte_representation", BYTE_REPRESENTATION_LEGACY_RAW))
        for config in run_configs
    }
    if len(byte_representations) != 1:
        raise ValueError(
            "B2 paired evaluation currently requires one byte representation; "
            f"got {sorted(byte_representations)}"
        )
    byte_representation = next(iter(byte_representations))
    tokenizer = PseudoCodeTokenizer(
        max_length=128,
        byte_representation=byte_representation,
    )
    test_benign, test_packed = load_track_b_v2_strict_data(
        tokenizer,
        bag_cache_dir,
        exclude_dirty_benign=args.exclude_apkid_dirty_strict_benign,
    )
    test_bags = list(test_benign) + list(test_packed)
    print(f"Loaded strict DPT test bags: {len(test_bags)}")

    metric_names = (
        "auroc",
        "auprc",
        "fpr_at_95_tpr",
        "tpr_at_1_fpr",
        "tpr_at_5_fpr",
        "brier",
        "ece",
        "best_f1_threshold",
    )
    run_results: Dict[str, Dict] = {}
    for idx, (name, path) in enumerate(runs, 1):
        print(f"[{idx}/{len(runs)}] Evaluating {name}: {path}")
        result = evaluate_result(name, path, test_bags, device, n_bins=args.bins)
        result["bootstrap_ci"] = bootstrap_cis(
            result["labels"],
            result["raw_scores"],
            n_bootstrap=args.bootstrap,
            seed=args.seed + idx,
            metrics=metric_names,
        )
        run_results[name] = result

    paired_rows = []
    for pair_idx, (a, b) in enumerate(DEFAULT_PAIR_COMPARISONS, 1):
        if a not in run_results or b not in run_results:
            continue
        labels_a = run_results[a]["labels"]
        labels_b = run_results[b]["labels"]
        if labels_a != labels_b:
            raise ValueError(f"paired labels differ for {a} vs {b}")
        metrics = {
            metric: paired_bootstrap_delta(
                labels_a,
                run_results[a]["raw_scores"],
                run_results[b]["raw_scores"],
                metric=metric,
                n_bootstrap=args.bootstrap,
                seed=args.seed + 1000 + pair_idx,
            )
            for metric in ("auroc", "auprc", "fpr_at_95_tpr", "brier", "ece")
        }
        paired_rows.append({"a": a, "b": b, "metrics": metrics})

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "Stage B B2 calibration/bootstrap",
        "diagnostic_only": True,
        "n_bootstrap": args.bootstrap,
        "seed": args.seed,
        "n_test_benign": len(test_benign),
        "n_test_packed": len(test_packed),
        "runs": run_results,
        "paired_comparisons": paired_rows,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = output.with_suffix(".md")
    write_markdown(summary, md_path)
    print(f"Wrote {output}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
