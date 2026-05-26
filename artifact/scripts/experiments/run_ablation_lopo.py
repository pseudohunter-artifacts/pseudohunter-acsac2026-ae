"""Ablation LOPO evaluation: 4 configurations to isolate component contributions.

Configurations:
  1. BERT-only:  No stat features (stat_proj zeroed), only BERT signal
  2. Stat-only:  No BERT (bert outputs zeroed), only stat features
  3. Concat:     Legacy concat fusion (no gating)
  4. Gated:      Full gated fusion (current best)

Each config runs 7-fold LOPO. Results saved incrementally.

Usage:
    python scripts/experiments/run_ablation_lopo.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from android_packer.apkio.objects import iter_apk_objects
from android_packer.decoders.pseudo_tokenizer import PseudoCodeTokenizer
from android_packer.features.full_feature_extractor import (
    SCALAR_FEATURE_DIM,
    extract_apk_context,
    extract_region_features,
)
from android_packer.labeling.happer_diff import compute_paired_diff, parse_inject_labels
from android_packer.models.entry_aggregator import (
    APKMILConfig,
    EntryAggregatorConfig,
    build_apk_mil,
    build_entry_aggregator,
)
from android_packer.models.fusion_encoder import FusionEncoderConfig, build_fusion_encoder
from android_packer.regioning.typed_slicer import iter_typed_regions

HAPPER = ROOT / "data" / "happer_dataset" / "FSet"
TRACK_B = ROOT / "data" / "real_world" / "track_b"
PRETRAIN_CKPT = ROOT / "outputs" / "experiments" / "pseudo_bert_v3" / "pretrained_bert_v2.pt"
OUT_DIR = ROOT / "outputs" / "experiments" / "ablation_lopo"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AblationModel(torch.nn.Module):
    def __init__(self, fusion_cfg: FusionEncoderConfig, ablation_mode: str = "gated"):
        super().__init__()
        self.ablation_mode = ablation_mode
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

    def forward_bag(self, bag: Dict, device: torch.device, chunk_size: int = 64):
        n_regions = bag["stat_features"].shape[0]
        max_regions = 128
        if n_regions > max_regions:
            rng_sub = np.random.RandomState(n_regions)
            indices = rng_sub.choice(n_regions, max_regions, replace=False)
            indices.sort()
            idx_set = set(indices.tolist())
            new_idx_map = {old: new for new, old in enumerate(indices)}
            entry_boundaries = []
            for start, end in bag["entry_boundaries"]:
                entry_idx = [new_idx_map[i] for i in range(start, end) if i in idx_set]
                if entry_idx:
                    entry_boundaries.append((entry_idx[0], entry_idx[-1] + 1))
        else:
            indices = np.arange(n_regions)
            entry_boundaries = bag["entry_boundaries"]

        N = len(indices)
        if N == 0 or not entry_boundaries:
            return torch.zeros((), device=device), torch.zeros((0,), device=device)

        stat = torch.tensor(bag["stat_features"][indices], dtype=torch.float32, device=device)
        d_ids = torch.tensor(bag["dalvik_ids"][indices], dtype=torch.long, device=device)
        d_types = torch.tensor(bag["dalvik_types"][indices], dtype=torch.long, device=device)
        d_mask = torch.tensor(bag["dalvik_mask"][indices], dtype=torch.float32, device=device)
        n_ids = torch.tensor(bag["native_ids"][indices], dtype=torch.long, device=device)
        n_types = torch.tensor(bag["native_types"][indices], dtype=torch.long, device=device)
        n_mask = torch.tensor(bag["native_mask"][indices], dtype=torch.float32, device=device)
        b_ids = torch.tensor(bag["byte_ids"][indices], dtype=torch.long, device=device)
        b_types = torch.tensor(bag["byte_types"][indices], dtype=torch.long, device=device)
        b_mask = torch.tensor(bag["byte_mask"][indices], dtype=torch.float32, device=device)

        # Ablation: zero out inputs based on mode
        if self.ablation_mode == "stat_only":
            # Zero out token inputs → BERT sees nothing useful
            d_ids = torch.zeros_like(d_ids)
            n_ids = torch.zeros_like(n_ids)
            b_ids = torch.zeros_like(b_ids)
        elif self.ablation_mode == "bert_only":
            # Zero out stat features → only BERT signal
            stat = torch.zeros_like(stat)

        all_emb, all_susp, all_norm = [], [], []
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            s = slice(cs, ce)
            emb, susp, norm = self.fusion_encoder(
                d_ids[s], d_types[s], d_mask[s],
                n_ids[s], n_types[s], n_mask[s],
                b_ids[s], b_types[s], b_mask[s], stat[s],
            )
            all_emb.append(emb)
            all_susp.append(susp)
            all_norm.append(norm)

        embeddings = torch.cat(all_emb)
        suspicion = torch.cat(all_susp)
        normality = torch.cat(all_norm)

        entry_emb, entry_susp, _ = self.entry_aggregator(embeddings, suspicion, entry_boundaries)
        entry_norms = []
        for start, end in entry_boundaries:
            if start < end and end <= len(normality):
                entry_norms.append(normality[start:end].mean())
            else:
                entry_norms.append(torch.tensor(0.5, device=device))
        entry_normality = torch.stack(entry_norms)
        bag_logit, entry_attn, entry_logits = self.apk_mil(entry_emb, entry_normality)
        return bag_logit, entry_attn


# ---------------------------------------------------------------------------
# Bag building (same as LOPO)
# ---------------------------------------------------------------------------


def build_bag(apk_path, apk_label, tokenizer, diff_result=None, apk_id=""):
    try:
        entries = []
        dex_counts = {}
        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) >= 64:
                entries.append((obj_meta, obj_bytes))
                if obj_bytes[:4] == b"dex\n" and len(obj_bytes) >= 100:
                    try:
                        s = struct.unpack_from("<I", obj_bytes, 56)[0]
                        t = struct.unpack_from("<I", obj_bytes, 64)[0]
                        m = struct.unpack_from("<I", obj_bytes, 88)[0]
                        f = struct.unpack_from("<I", obj_bytes, 80)[0]
                        dex_counts[obj_meta.object_path] = (s, t, m, f)
                    except (struct.error, IndexError):
                        pass
        if not entries:
            return None
        apk_ctx = extract_apk_context([(m.object_path, b) for m, b in entries])
        all_stat, all_d_ids, all_n_ids, all_b_ids = [], [], [], []
        all_d_types, all_n_types, all_b_types = [], [], []
        all_d_mask, all_n_mask, all_b_mask = [], [], []
        entry_boundaries, entry_names = [], []
        ridx = 0
        for eidx, (obj_meta, obj_bytes) in enumerate(entries):
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
                entry_names.append(obj_meta.object_path)
        if not entry_boundaries:
            return None
        diff_targets = None
        if diff_result:
            diff_targets = np.zeros(len(entry_boundaries), dtype=np.float32)
            for i, name in enumerate(entry_names):
                if name in diff_result.entry_diffs:
                    diff_targets[i] = diff_result.entry_diffs[name].diff_score
        return {
            "apk_id": apk_id, "apk_label": apk_label,
            "stat_features": np.array(all_stat, dtype=np.float32),
            "dalvik_ids": np.array(all_d_ids, dtype=np.int64),
            "native_ids": np.array(all_n_ids, dtype=np.int64),
            "byte_ids": np.array(all_b_ids, dtype=np.int64),
            "dalvik_types": np.array(all_d_types, dtype=np.int64),
            "native_types": np.array(all_n_types, dtype=np.int64),
            "byte_types": np.array(all_b_types, dtype=np.int64),
            "dalvik_mask": np.array(all_d_mask, dtype=np.float32),
            "native_mask": np.array(all_n_mask, dtype=np.float32),
            "byte_mask": np.array(all_b_mask, dtype=np.float32),
            "entry_boundaries": entry_boundaries,
            "diff_targets": diff_targets,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class PackerFamily:
    name: str
    bags: List[Dict]


def load_all_data(tokenizer, max_per_family=15, androzoo_benign=0):
    print("  Loading all data...", flush=True)
    benign_bags = []
    origins = {}
    for p in sorted((HAPPER / "Origin-16").glob("*.apk"))[:30]:
        origins[p.stem] = p
        bag = build_bag(p, 0, tokenizer, apk_id=f"benign_{p.stem}")
        if bag:
            benign_bags.append(bag)
    # NOTE: Track B benign intentionally EXCLUDED from training to avoid
    # app-identity leakage. These 9 apps are the same ones Track B packers pack.
    # They are still used OFFLINE for diff_targets computation.

    # AndroZoo benign (modern 2020+ apps, no overlap with test)
    ANDROZOO_DIR = ROOT / "data" / "androzoo" / "benign_corpus"
    if androzoo_benign > 0 and ANDROZOO_DIR.exists():
        az_apks = sorted(ANDROZOO_DIR.rglob("*.apk"))
        rng_az = np.random.RandomState(123)
        if len(az_apks) > androzoo_benign:
            az_indices = rng_az.choice(len(az_apks), androzoo_benign, replace=False)
            az_apks = [az_apks[i] for i in sorted(az_indices)]
        print(f"    Loading {len(az_apks)} AndroZoo benign...", flush=True)
        az_count = 0
        for p in az_apks:
            bag = build_bag(p, 0, tokenizer, apk_id=f"az_{p.stem[:20]}")
            if bag:
                benign_bags.append(bag)
                az_count += 1
        print(f"    AndroZoo: {az_count} bags added")

    benign_test_bags = []
    for p in sorted((HAPPER / "Oirgin-18").glob("*.apk"))[:15]:
        bag = build_bag(p, 0, tokenizer, apk_id=f"test_benign_{p.stem}")
        if bag:
            benign_test_bags.append(bag)

    print(f"    Benign: {len(benign_bags)} train + {len(benign_test_bags)} test")

    families = []
    for family_name, dirname in [("Ali", "Ali-16"), ("Qihoo", "Qihoo-16"), ("Tencent", "Tencent-16")]:
        family_bags = []
        family_dir = HAPPER / dirname
        if not family_dir.exists():
            continue
        for p in sorted(family_dir.glob("*.apk"))[:max_per_family]:
            origin_match = None
            for ostem, opath in origins.items():
                if p.stem.startswith(ostem):
                    origin_match = opath
                    break
            diff = compute_paired_diff(origin_match, p) if origin_match else None
            bag = build_bag(p, 1, tokenizer, diff, f"{family_name}_{p.stem}")
            if bag:
                family_bags.append(bag)
        if family_bags:
            families.append(PackerFamily(name=family_name, bags=family_bags))
            print(f"    {family_name}: {len(family_bags)} bags")

    tb_packed = TRACK_B / "packed"
    tb_benign = TRACK_B / "benign"
    for packer_name, short_name in [
        ("cs1_360_jiagu", "360"),
        ("cs3_bangcle", "Bangcle"),
        ("s5_timscriptov_apkprotector_multiplatform", "APKProtector"),
        ("s6_dpt_shell", "DPT"),
    ]:
        family_bags = []
        packer_dir = tb_packed / packer_name
        if packer_dir.exists() and packer_dir.is_dir():
            for seed_dir in sorted(packer_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                packed_apk = seed_dir / "packed.apk"
                if not packed_apk.exists():
                    continue
                benign_apk = tb_benign / f"{seed_dir.name}.apk"
                inject_path = seed_dir / "inject_labels.jsonl"
                if inject_path.exists():
                    diff = parse_inject_labels(inject_path)
                elif benign_apk.exists():
                    diff = compute_paired_diff(benign_apk, packed_apk)
                else:
                    diff = None
                bag = build_bag(packed_apk, 1, tokenizer, diff, f"{short_name}_{seed_dir.name}")
                if bag:
                    family_bags.append(bag)
        for apk_file in sorted(tb_packed.glob(f"{packer_name}__*.apk")):
            jsonl = apk_file.with_name(apk_file.stem + ".inject_labels.jsonl")
            diff = parse_inject_labels(jsonl) if jsonl.exists() else None
            bag = build_bag(apk_file, 1, tokenizer, diff, f"{short_name}_flat_{apk_file.stem[:30]}")
            if bag:
                family_bags.append(bag)
        if family_bags:
            families.append(PackerFamily(name=short_name, bags=family_bags))
            print(f"    {short_name}: {len(family_bags)} bags")

    return benign_bags, benign_test_bags, families


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_fold(model, train_bags, device, epochs=50, lr=5e-4):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.RandomState(42)
    packed_logits, benign_logits = [], []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0
        order = rng.permutation(len(train_bags))
        packed_logits.clear()
        benign_logits.clear()

        for i in range(0, len(train_bags), 4):
            batch_indices = order[i:i + 4]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for idx in batch_indices:
                bag = train_bags[idx]
                bag_logit, _ = model.forward_bag(bag, device)
                label = bag["apk_label"]
                target = torch.tensor(float(label), device=device)
                l_bag = F.binary_cross_entropy_with_logits(bag_logit, target)
                l_rank = torch.tensor(0.0, device=device)
                margin = 2.0
                if label == 1 and benign_logits:
                    for bl in benign_logits[-5:]:
                        l_rank = l_rank + F.relu(margin + bl - bag_logit)
                    l_rank = l_rank / len(benign_logits[-5:])
                elif label == 0 and packed_logits:
                    for pl in packed_logits[-5:]:
                        l_rank = l_rank + F.relu(margin + bag_logit - pl)
                    l_rank = l_rank / len(packed_logits[-5:])
                loss = l_bag + 0.5 * l_rank
                batch_loss = batch_loss + loss / len(batch_indices)
                logit_val = bag_logit.detach().item()
                if label == 1:
                    packed_logits.append(logit_val)
                else:
                    benign_logits.append(logit_val)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_steps += 1
        scheduler.step()
    return model


def evaluate_fold(model, test_bags, device):
    """Evaluate one fold, return (y_true, y_score, inference_times_ms)."""
    model.eval()
    y_true, y_score, times_ms = [], [], []
    with torch.no_grad():
        for bag in test_bags:
            t0 = time.time()
            bag_logit, _ = model.forward_bag(bag, device)
            score = torch.sigmoid(bag_logit).item()
            times_ms.append((time.time() - t0) * 1000)
            y_true.append(bag["apk_label"])
            y_score.append(score)
    return y_true, y_score, times_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ABLATION_CONFIGS = {
    "bert_only": {"use_gated_fusion": True, "ablation_mode": "bert_only"},
    "stat_only": {"use_gated_fusion": True, "ablation_mode": "stat_only"},
    "concat":    {"use_gated_fusion": False, "ablation_mode": "full"},
    "gated":     {"use_gated_fusion": True, "ablation_mode": "full"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        help="Which ablation configs to run")
    parser.add_argument("--androzoo-benign", type=int, default=0,
                        help="Add N AndroZoo benign APKs to training")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")
    print(f"Configs to run: {args.configs}")

    tokenizer = PseudoCodeTokenizer(max_length=128)

    # Load data once (expensive)
    print("\n=== Loading Data ===", flush=True)
    t0 = time.time()
    benign_bags, benign_test_bags, families = load_all_data(
        tokenizer, androzoo_benign=args.androzoo_benign
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.0f}s")
    print(f"  Families: {[f.name for f in families]}")

    # Progress file (for monitoring)
    progress_path = OUT_DIR / "progress.json"
    all_results = {}

    for config_name in args.configs:
        cfg_params = ABLATION_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"=== Ablation: {config_name} ===")
        print(f"{'='*60}", flush=True)

        fold_results = []

        for fold_idx, held_out in enumerate(families):
            print(f"  Fold {fold_idx+1}/{len(families)}: held-out={held_out.name}", flush=True)

            # Build train/test
            train_bags = list(benign_bags)
            for fam in families:
                if fam.name != held_out.name:
                    train_bags.extend(fam.bags)
            test_bags = list(benign_test_bags) + held_out.bags

            # Build model
            fusion_cfg = FusionEncoderConfig(
                bert_hidden_dim=256, bert_n_layers=4, bert_max_length=128,
                bert_n_heads=8, bert_intermediate_dim=512,
                use_gated_fusion=cfg_params["use_gated_fusion"],
                gate_hidden_dim=128,
            )
            model = AblationModel(fusion_cfg, ablation_mode=cfg_params["ablation_mode"])

            # Load pretrained
            if PRETRAIN_CKPT.exists():
                state = torch.load(PRETRAIN_CKPT, map_location="cpu")
                model_dict = model.state_dict()
                first_key = next(iter(state.keys()))
                if first_key.startswith("bert.") or first_key.startswith("fusion."):
                    pretrained = {f"fusion_encoder.{k}": v for k, v in state.items()
                                  if f"fusion_encoder.{k}" in model_dict
                                  and v.shape == model_dict[f"fusion_encoder.{k}"].shape}
                else:
                    pretrained = {k: v for k, v in state.items()
                                  if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained)
                model.load_state_dict(model_dict)

            model.freeze_bert()
            model = model.to(device)

            # Train + eval
            model = train_fold(model, train_bags, device, epochs=args.epochs, lr=args.lr)
            y_true, y_score, times_ms = evaluate_fold(model, test_bags, device)

            packed_scores = [s for s, l in zip(y_score, y_true) if l == 1]
            benign_scores = [s for s, l in zip(y_score, y_true) if l == 0]
            auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else 0.0
            det_rate = sum(1 for s in packed_scores if s > 0.5) / max(len(packed_scores), 1)
            mean_inference_ms = float(np.mean(times_ms))

            fold_results.append({
                "fold": fold_idx + 1, "held_out": held_out.name,
                "auroc": auroc, "det_rate": det_rate,
                "packed_mean": float(np.mean(packed_scores)),
                "benign_mean": float(np.mean(benign_scores)),
                "mean_inference_ms": round(mean_inference_ms, 1),
            })
            print(f"    AUROC={auroc:.4f} det={det_rate:.1%} infer={mean_inference_ms:.0f}ms/APK")

            # Free GPU memory
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # Summarize this config
        mean_auroc = np.mean([r["auroc"] for r in fold_results])
        mean_det = np.mean([r["det_rate"] for r in fold_results])
        all_results[config_name] = {
            "mean_auroc": float(mean_auroc),
            "mean_det_rate": float(mean_det),
            "folds": fold_results,
        }
        print(f"\n  {config_name} MEAN AUROC: {mean_auroc:.4f}, Det: {mean_det:.1%}")

        # Save progress incrementally
        with open(progress_path, "w") as f:
            json.dump({
                "status": "running",
                "completed_configs": list(all_results.keys()),
                "remaining_configs": [c for c in args.configs if c not in all_results],
                "results": all_results,
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"=== Ablation Results Summary ===")
    print(f"{'='*60}")
    print(f"{'Config':<12} {'AUROC':>7} {'Det.Rate':>9}")
    print(f"{'-'*35}")
    for name, res in all_results.items():
        print(f"{name:<12} {res['mean_auroc']:>7.4f} {res['mean_det_rate']:>8.1%}")

    # Save final
    with open(progress_path, "w") as f:
        json.dump({
            "status": "completed",
            "results": all_results,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
