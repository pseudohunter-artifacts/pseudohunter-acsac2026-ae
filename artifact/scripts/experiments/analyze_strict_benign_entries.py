"""Analyze top-scored entry types for Track B v2 strict benign APKs.

This diagnostic is intentionally separate from the training runner. It reloads a
strict-DPT checkpoint, rebuilds strict Track B v2 bags by default, and
aggregates which benign entry types receive high localization scores. Use
``--use-bag-cache`` when the current cache version already contains entry names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES
from scripts.experiments.run_lopo_eval import (
    APKID_DIRTY_STRICT_BENIGN,
    BAG_CACHE_DIR,
    PseudoCodeTokenizer,
    _torch_load_local,
    build_model,
    load_track_b_v2_strict_data,
)


DEFAULT_RESULT = (
    ROOT
    / "outputs"
    / "experiments"
    / "track_b_v2_strict_dpt"
    / "results_strict_dpt_clean_z.json"
)


def _result_checkpoint_path(result_path: Path) -> Path:
    return result_path.parent / "checkpoints" / result_path.stem / "strict_dpt" / "latest.pt"


def _load_result_config(result_path: Path) -> Dict:
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return dict(result.get("config", {}))


def _args_from_config(config: Mapping) -> SimpleNamespace:
    active_paths = config.get("active_paths", ("dalvik", "arm64", "byte"))
    return SimpleNamespace(
        bert_dim=int(config.get("bert_dim", 512)),
        bert_layers=int(config.get("bert_layers", 8)),
        ablation=str(config.get("ablation", "bert_only")),
        paths=tuple(active_paths),
        path_dropout=float(config.get("path_dropout", 0.0)),
        region_type_routing=bool(config.get("region_type_routing", False)),
        routing_dex_byte_weight=float(config.get("routing_dex_byte_weight", 0.25)),
        routing_elf_byte_weight=float(config.get("routing_elf_byte_weight", 0.25)),
        routing_byte_entry_weight=float(config.get("routing_byte_entry_weight", 1.0)),
        routing_unknown_weight=float(config.get("routing_unknown_weight", 0.25)),
    )


def _load_model(result_path: Path, checkpoint_path: Path, device: torch.device):
    config = _load_result_config(result_path)
    args = _args_from_config(config)
    model = build_model(args)
    ckpt = _torch_load_local(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model, config


def _entry_type_name(bag: Mapping, entry_idx: int) -> str:
    boundaries = bag.get("entry_boundaries", [])
    entry_type_ids = bag.get("entry_type_ids")
    if entry_idx >= len(boundaries) or entry_type_ids is None:
        return "unknown"
    start, end = boundaries[entry_idx]
    if start >= end:
        return "unknown"
    ids = np.asarray(entry_type_ids[start:end], dtype=np.int64)
    if ids.size == 0:
        return "unknown"
    counts = Counter(int(i) for i in ids.tolist())
    type_id = counts.most_common(1)[0][0]
    if 0 <= type_id < len(ENTRY_COARSE_TYPES):
        return ENTRY_COARSE_TYPES[type_id]
    return "unknown"


def _path_bucket(path: str, coarse_type: str) -> str:
    p = path.lower().replace("\\", "/")
    base = p.rsplit("/", 1)[-1] if "/" in p else p
    suffix = "." + base.rsplit(".", 1)[-1] if "." in base else ""

    if base == "classes.dex":
        return "classes.dex"
    if base.startswith("classes") and base.endswith(".dex"):
        return "secondary_dex"
    if p.startswith("lib/") or base.endswith(".so"):
        return "lib_so"
    if base == "resources.arsc":
        return "resources_arsc"
    if p == "androidmanifest.xml":
        return "manifest"
    if p.startswith("meta-inf/"):
        return "meta_inf"
    if coarse_type == "archive" or suffix in {".apk", ".jar", ".zip", ".gz"}:
        return "nested_archive"
    if p.startswith("res/raw/"):
        return "res_raw"
    if p.startswith("res/"):
        return "res_other"
    if p.startswith("assets/"):
        if suffix in {
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".tflite",
            ".onnx",
            ".pb",
            ".pt",
            ".mp4",
            ".mp3",
            ".ogg",
            ".wav",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            return "model_data_file"
        return "assets"
    if coarse_type == "asset":
        return "asset_entry"
    if coarse_type == "unknown":
        return "unknown_blob"
    return coarse_type


def _summarize_records(records: Sequence[Mapping], example_limit: int) -> List[Dict]:
    grouped: MutableMapping[str, List[Mapping]] = defaultdict(list)
    for record in records:
        grouped[str(record["bucket"])].append(record)

    rows = []
    for bucket, items in grouped.items():
        scores = [float(item["score"]) for item in items]
        examples = sorted(items, key=lambda item: float(item["score"]), reverse=True)[
            :example_limit
        ]
        rows.append(
            {
                "entry_type": bucket,
                "high_score_count": len(items),
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "max_score": float(np.max(scores)) if scores else 0.0,
                "examples": [
                    {
                        "apk_id": str(item["apk_id"]),
                        "entry": str(item["entry"]),
                        "coarse_type": str(item["coarse_type"]),
                        "score": float(item["score"]),
                        "apk_score": float(item["apk_score"]),
                    }
                    for item in examples
                ],
            }
        )
    return sorted(rows, key=lambda row: (-int(row["high_score_count"]), -float(row["max_score"])))


def _iter_top_records(
    model,
    bags: Sequence[Mapping],
    device: torch.device,
    top_k: int,
) -> Iterable[Dict]:
    with torch.no_grad():
        for bag in bags:
            out = model.forward_bag(bag, device)
            apk_score = float(torch.sigmoid(out["bag_logit"]).item())
            entry_indices = list(out.get("entry_indices", range(len(out["entry_attention"]))))
            entry_names = list(bag.get("entry_names", []))

            attention = out["entry_attention"].detach().cpu().numpy()
            anomaly = (1.0 - out["entry_normality"]).detach().cpu().numpy()
            suspicion = out["entry_suspicion"].detach().cpu().numpy()
            entry_prob = torch.sigmoid(out["entry_logits"]).detach().cpu().numpy()
            contribution = attention * entry_prob
            metrics = {
                "attention": attention,
                "anomaly": anomaly,
                "suspicion": suspicion,
                "apk_contribution": contribution,
            }

            for metric_name, scores in metrics.items():
                if len(scores) == 0:
                    continue
                k = min(top_k, len(scores))
                ranked = np.argsort(-scores)[:k]
                for rank, out_idx in enumerate(ranked.tolist(), 1):
                    original_idx = int(entry_indices[out_idx])
                    entry = (
                        entry_names[original_idx]
                        if original_idx < len(entry_names)
                        else f"<entry:{original_idx}>"
                    )
                    coarse_type = _entry_type_name(bag, original_idx)
                    yield {
                        "metric": metric_name,
                        "rank": rank,
                        "apk_id": str(bag.get("apk_id", "")),
                        "apk_score": apk_score,
                        "entry_index": original_idx,
                        "entry": entry,
                        "coarse_type": coarse_type,
                        "bucket": _path_bucket(entry, coarse_type),
                        "score": float(scores[out_idx]),
                    }


def _write_markdown(summary: Mapping, path: Path) -> None:
    lines = [
        "# Strict Benign Top-Entry Type Diagnostics",
        "",
        f"Result: `{summary['result_file']}`",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Strict benign APKs: {summary['n_benign']}",
        f"Top-k per APK per metric: {summary['top_k']}",
        "",
    ]
    for metric, rows in summary["by_metric"].items():
        lines.extend(
            [
                f"## {metric}",
                "",
                "| Entry type | High-score count | Mean score | Max score | Example |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            example = row["examples"][0] if row["examples"] else {}
            example_text = example.get("entry", "")
            if len(example_text) > 72:
                example_text = "..." + example_text[-69:]
            lines.append(
                "| {entry_type} | {count} | {mean:.4f} | {max_score:.4f} | `{example}` |".format(
                    entry_type=row["entry_type"],
                    count=row["high_score_count"],
                    mean=row["mean_score"],
                    max_score=row["max_score"],
                    example=example_text,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--example-limit", type=int, default=3)
    parser.add_argument("--include-apkid-dirty", action="store_true")
    parser.add_argument("--use-bag-cache", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    result_path = args.result.resolve()
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else _result_checkpoint_path(result_path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    model, config = _load_model(result_path, checkpoint_path, device)

    tokenizer = PseudoCodeTokenizer(max_length=128)
    test_benign, _ = load_track_b_v2_strict_data(
        tokenizer,
        bag_cache_dir=BAG_CACHE_DIR if args.use_bag_cache else None,
        exclude_dirty_benign=not args.include_apkid_dirty,
    )
    if not args.include_apkid_dirty:
        dirty_ids = tuple(APKID_DIRTY_STRICT_BENIGN)
        test_benign = [
            bag for bag in test_benign
            if not any(str(bag.get("apk_id", "")).upper().endswith(d[:24]) for d in dirty_ids)
        ]

    records = list(_iter_top_records(model, test_benign, device, args.top_k))
    by_metric = {}
    for metric in ("apk_contribution", "attention", "suspicion", "anomaly"):
        metric_records = [record for record in records if record["metric"] == metric]
        by_metric[metric] = _summarize_records(metric_records, args.example_limit)

    out_json = args.out_json or result_path.with_name(f"{result_path.stem}_benign_entry_types.json")
    out_md = args.out_md or result_path.with_name(f"{result_path.stem}_benign_entry_types.md")
    summary = {
        "result_file": str(result_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "top_k": args.top_k,
        "n_benign": len(test_benign),
        "config": config,
        "by_metric": by_metric,
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary, out_md)

    print(f"Analyzed {len(test_benign)} strict benign APKs")
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    for metric, rows in by_metric.items():
        top = rows[:3]
        print(f"\n{metric}:")
        for row in top:
            print(
                f"  {row['entry_type']}: n={row['high_score_count']} "
                f"mean={row['mean_score']:.4f} max={row['max_score']:.4f}"
            )


if __name__ == "__main__":
    main()
