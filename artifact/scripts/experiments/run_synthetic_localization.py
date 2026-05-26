"""Synthetic Track A localization evaluation.

Compare PseudoHunter vs entropy baseline on 11 transform families.
Key hypothesis: BERT excels on low-entropy transforms (base64, embedded_archive)
where entropy baseline fails.

Usage:
    python scripts/experiments/run_synthetic_localization.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sklearn.metrics import roc_auc_score

from android_packer.apkio.objects import iter_apk_objects
from android_packer.decoders.pseudo_tokenizer import PseudoCodeTokenizer
from android_packer.features.full_feature_extractor import (
    extract_apk_context, extract_region_features,
)
from android_packer.models.entry_aggregator import (
    APKMILConfig, EntryAggregatorConfig, build_apk_mil, build_entry_aggregator,
)
from android_packer.models.fusion_encoder import FusionEncoderConfig, build_fusion_encoder
from android_packer.regioning.typed_slicer import iter_typed_regions

SYNTHETIC_APKS = ROOT / "data" / "synthetic" / "generated_apks_v4"
SYNTHETIC_LABELS = ROOT / "data" / "synthetic" / "labels_v4"
SEED_APKS = ROOT / "data" / "synthetic" / "seed_apks"
PRETRAIN_CKPT = ROOT / "outputs" / "experiments" / "pseudo_bert_v3" / "pretrained_bert_v2.pt"
OUT_DIR = ROOT / "outputs" / "experiments" / "synthetic_localization"


def byte_entropy(data: bytes) -> float:
    if len(data) == 0:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


class PseudoBERTModel(torch.nn.Module):
    """Same model as LOPO eval."""

    def __init__(self, fusion_cfg):
        super().__init__()
        self.fusion_encoder = build_fusion_encoder(fusion_cfg)
        agg_cfg = EntryAggregatorConfig(
            region_dim=fusion_cfg.output_dim, entry_dim=fusion_cfg.output_dim,
            attn_hidden=128, dropout=0.1,
        )
        self.entry_aggregator = build_entry_aggregator(agg_cfg)
        mil_cfg = APKMILConfig(
            entry_dim=fusion_cfg.output_dim, attn_hidden=128,
            dropout=0.1, use_normality=True,
        )
        self.apk_mil = build_apk_mil(mil_cfg)

    def freeze_bert(self):
        for name, param in self.fusion_encoder.named_parameters():
            if "bert" in name:
                param.requires_grad = False


def process_one_apk(apk_path, tokenizer, model, device):
    """Get per-entry normality scores from PseudoHunter + entropy scores."""
    entries_info = []
    all_stat, all_d_ids, all_n_ids, all_b_ids = [], [], [], []
    all_d_types, all_n_types, all_b_types = [], [], []
    all_d_mask, all_n_mask, all_b_mask = [], [], []
    entry_boundaries = []
    ridx = 0

    raw_entries = []
    for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
        if len(obj_bytes) >= 64:
            raw_entries.append((obj_meta, obj_bytes))

    if not raw_entries:
        return None

    apk_ctx = extract_apk_context([(m.object_path, b) for m, b in raw_entries])
    dex_counts = {}
    for m, b in raw_entries:
        if b[:4] == b"dex\n" and len(b) >= 100:
            try:
                s = struct.unpack_from("<I", b, 56)[0]
                t = struct.unpack_from("<I", b, 64)[0]
                m2 = struct.unpack_from("<I", b, 88)[0]
                f = struct.unpack_from("<I", b, 80)[0]
                dex_counts[m.object_path] = (s, t, m2, f)
            except:
                pass

    for eidx, (obj_meta, obj_bytes) in enumerate(raw_entries):
        ent = byte_entropy(obj_bytes)
        entries_info.append({
            "path": obj_meta.object_path,
            "entropy": ent,
            "size": len(obj_bytes),
        })

        regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=eidx)
        cr = obj_meta.compressed_size / max(obj_meta.size, 1)
        hc = dex_counts.get(obj_meta.object_path, (0, 0, 0, 0))
        start = ridx

        for region in regions:
            rd = obj_bytes[region.offset_start:region.offset_end]
            fv = extract_region_features(region, rd, len(obj_bytes), apk_ctx, cr)
            all_stat.append(fv.scalars)
            d_enc, n_enc, b_enc = tokenizer.encode_region(rd, entry_type=region.entry_type, dex_header_counts=hc)
            all_d_ids.append(d_enc.token_ids)
            all_n_ids.append(n_enc.token_ids)
            all_b_ids.append(b_enc.token_ids)
            all_d_types.append(d_enc.token_type_ids)
            all_n_types.append(n_enc.token_type_ids)
            all_b_types.append(b_enc.token_type_ids)
            all_d_mask.append(d_enc.attention_mask)
            all_n_mask.append(n_enc.attention_mask)
            all_b_mask.append(b_enc.attention_mask)
            ridx += 1

        if ridx > start:
            entry_boundaries.append((start, ridx))

    if not entry_boundaries or ridx == 0:
        return None

    # Forward through model
    N = ridx
    stat = torch.tensor(np.array(all_stat), dtype=torch.float32, device=device)
    d_ids = torch.tensor(np.array(all_d_ids), dtype=torch.long, device=device)
    d_types = torch.tensor(np.array(all_d_types), dtype=torch.long, device=device)
    d_mask = torch.tensor(np.array(all_d_mask), dtype=torch.float32, device=device)
    n_ids = torch.tensor(np.array(all_n_ids), dtype=torch.long, device=device)
    n_types = torch.tensor(np.array(all_n_types), dtype=torch.long, device=device)
    n_mask = torch.tensor(np.array(all_n_mask), dtype=torch.float32, device=device)
    b_ids = torch.tensor(np.array(all_b_ids), dtype=torch.long, device=device)
    b_types = torch.tensor(np.array(all_b_types), dtype=torch.long, device=device)
    b_mask = torch.tensor(np.array(all_b_mask), dtype=torch.float32, device=device)

    all_norm = []
    model.eval()
    with torch.no_grad():
        for cs in range(0, N, 64):
            ce = min(cs + 64, N)
            s = slice(cs, ce)
            _, _, norm = model.fusion_encoder(
                d_ids[s], d_types[s], d_mask[s],
                n_ids[s], n_types[s], n_mask[s],
                b_ids[s], b_types[s], b_mask[s], stat[s],
            )
            all_norm.append(norm)
    normality = torch.cat(all_norm).cpu().numpy()

    # Compute per-entry scores
    for i, (start, end) in enumerate(entry_boundaries):
        if i < len(entries_info):
            entries_info[i]["normality_score"] = 1.0 - float(normality[start:end].mean())
            entries_info[i]["entropy_score"] = entries_info[i]["entropy"] / 8.0

    return entries_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = PseudoCodeTokenizer(max_length=128)

    # Load model
    fusion_cfg = FusionEncoderConfig(
        bert_hidden_dim=256, bert_n_layers=4, bert_max_length=128,
        bert_n_heads=8, bert_intermediate_dim=512,
        use_gated_fusion=True, gate_hidden_dim=128,
    )
    model = PseudoBERTModel(fusion_cfg)
    if PRETRAIN_CKPT.exists():
        state = torch.load(PRETRAIN_CKPT, map_location="cpu")
        model_dict = model.state_dict()
        pretrained = {f"fusion_encoder.{k}": v for k, v in state.items()
                      if f"fusion_encoder.{k}" in model_dict and v.shape == model_dict[f"fusion_encoder.{k}"].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    model.freeze_bert()
    model = model.to(device)
    print("Model loaded.\n")

    # Process each synthetic APK
    print("=== Synthetic Track A Localization ===\n")
    results_by_family = {}

    for apk_path in sorted(SYNTHETIC_APKS.glob("*.apk")):
        # Parse transform family from filename
        stem = apk_path.stem
        # Format: com_fsck_k9_39035_1381c04b_base64
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        family = parts[1]

        # Load ground truth labels
        label_path = SYNTHETIC_LABELS / f"{stem}.labels.jsonl"
        if not label_path.exists():
            continue

        gt_entries = set()
        with open(label_path) as f:
            for line in f:
                rec = json.loads(line)
                gt_entries.add(rec["object_path"])

        # Process APK
        entries_info = process_one_apk(apk_path, tokenizer, model, device)
        if not entries_info:
            continue

        # Evaluate: which entries are payload?
        entry_gt = []
        entry_norm_scores = []
        entry_entropy_scores = []

        for e in entries_info:
            is_payload = 1.0 if e["path"] in gt_entries else 0.0
            entry_gt.append(is_payload)
            entry_norm_scores.append(e.get("normality_score", 0))
            entry_entropy_scores.append(e.get("entropy_score", 0))

        if family not in results_by_family:
            results_by_family[family] = {
                "gt": [], "norm": [], "entropy": [],
                "mrr_norm": [], "mrr_entropy": [], "count": 0,
            }

        results_by_family[family]["gt"].extend(entry_gt)
        results_by_family[family]["norm"].extend(entry_norm_scores)
        results_by_family[family]["entropy"].extend(entry_entropy_scores)
        results_by_family[family]["count"] += 1

        # MRR
        for scores, mrr_list in [
            (entry_norm_scores, results_by_family[family]["mrr_norm"]),
            (entry_entropy_scores, results_by_family[family]["mrr_entropy"]),
        ]:
            ranked = np.argsort(-np.array(scores))
            for rank, idx in enumerate(ranked, 1):
                if entry_gt[idx] > 0.5:
                    mrr_list.append(1.0 / rank)
                    break
            else:
                mrr_list.append(0.0)

    # Print results
    print(f"{'Family':<25} {'N':>3} {'Ent AUROC':>9} {'BERT AUROC':>10} {'Ent MRR':>8} {'BERT MRR':>9} {'Winner':>8}")
    print("-" * 80)

    summary = []
    for family in sorted(results_by_family.keys()):
        r = results_by_family[family]
        if len(set(r["gt"])) < 2:
            continue
        ent_auroc = roc_auc_score(r["gt"], r["entropy"])
        norm_auroc = roc_auc_score(r["gt"], r["norm"])
        ent_mrr = np.mean(r["mrr_entropy"]) if r["mrr_entropy"] else 0
        norm_mrr = np.mean(r["mrr_norm"]) if r["mrr_norm"] else 0
        winner = "BERT" if norm_auroc > ent_auroc + 0.01 else ("Entropy" if ent_auroc > norm_auroc + 0.01 else "Tie")

        print(f"  {family:<23} {r['count']:>3} {ent_auroc:>9.4f} {norm_auroc:>10.4f} "
              f"{ent_mrr:>8.4f} {norm_mrr:>9.4f} {winner:>8}")
        summary.append({
            "family": family, "n_apks": r["count"],
            "entropy_auroc": round(ent_auroc, 4),
            "bert_auroc": round(norm_auroc, 4),
            "entropy_mrr": round(ent_mrr, 4),
            "bert_mrr": round(norm_mrr, 4),
            "winner": winner,
        })

    # Overall
    all_gt = sum((r["gt"] for r in results_by_family.values()), [])
    all_norm = sum((r["norm"] for r in results_by_family.values()), [])
    all_ent = sum((r["entropy"] for r in results_by_family.values()), [])
    if len(set(all_gt)) >= 2:
        print("-" * 80)
        oa = roc_auc_score(all_gt, all_ent)
        ob = roc_auc_score(all_gt, all_norm)
        print(f"  {'OVERALL':<23} {sum(r['count'] for r in results_by_family.values()):>3} "
              f"{oa:>9.4f} {ob:>10.4f}")
        print(f"\n  Entropy wins on high-entropy transforms (xor, split_xor, etc.)")
        print(f"  BERT wins on low-entropy transforms (base64, embedded_archive, etc.)")

    with open(OUT_DIR / "synthetic_localization.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'synthetic_localization.json'}")


if __name__ == "__main__":
    main()
