"""Quick stat-only baseline using the same training/eval pipeline as v3.

This gives us the pure stat_features (318-dim) performance in the same
MIL framework, serving as a control for BERT value-add.

Expected AUROC: close to 0.7543 (stat-only cross-dataset from v2 full framework).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from android_packer.apkio.objects import iter_apk_objects
from android_packer.features.full_feature_extractor import (
    SCALAR_FEATURE_DIM,
    extract_apk_context,
    extract_region_features,
)
from android_packer.labeling.happer_diff import compute_paired_diff
from android_packer.models.entry_aggregator import (
    APKMILConfig,
    EntryAggregatorConfig,
    build_apk_mil,
    build_entry_aggregator,
)
from android_packer.regioning.typed_slicer import iter_typed_regions

HAPPER = ROOT / "data" / "happer_dataset" / "FSet"
TRACK_B = ROOT / "data" / "real_world" / "track_b"
OUT_DIR = ROOT / "outputs" / "experiments" / "stat_only_v3_baseline"


class StatOnlyEncoder(nn.Module):
    """Simple MLP encoder for 318-dim stat features."""

    def __init__(self, input_dim=318, hidden_dim=256, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
        self.suspicion_head = nn.Linear(output_dim, 1)
        self.normality_head = nn.Linear(output_dim, 1)

    def forward(self, stat_features):
        h = self.mlp(stat_features)
        suspicion = self.suspicion_head(h).squeeze(-1)
        normality = torch.sigmoid(self.normality_head(h).squeeze(-1))
        return h, suspicion, normality


class StatOnlyModel(nn.Module):
    """Stat-only model: MLP → EntryAggregator → APK MIL."""

    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = StatOnlyEncoder(output_dim=output_dim)
        agg_cfg = EntryAggregatorConfig(
            region_dim=output_dim, entry_dim=output_dim,
            attn_hidden=128, dropout=0.1,
        )
        self.entry_aggregator = build_entry_aggregator(agg_cfg)
        mil_cfg = APKMILConfig(
            entry_dim=output_dim, attn_hidden=128,
            dropout=0.1, use_normality=True,
        )
        self.apk_mil = build_apk_mil(mil_cfg)

    def forward_bag(self, bag: Dict, device: torch.device):
        stat = torch.tensor(bag["stat_features"], dtype=torch.float32, device=device)
        entry_boundaries = bag["entry_boundaries"]

        embeddings, suspicion, normality = self.encoder(stat)

        entry_emb, entry_susp, _ = self.entry_aggregator(
            embeddings, suspicion, entry_boundaries
        )

        entry_norms = []
        for start, end in entry_boundaries:
            if start < end and end <= len(normality):
                entry_norms.append(normality[start:end].mean())
            else:
                entry_norms.append(torch.tensor(0.5, device=device))
        entry_normality = torch.stack(entry_norms)

        bag_logit, entry_attn, entry_logits = self.apk_mil(entry_emb, entry_normality)
        return bag_logit, entry_attn, entry_logits, normality


def build_stat_bag(apk_path, apk_label, diff_result=None, apk_id=""):
    """Build bag with only stat features (no token sequences)."""
    try:
        entries = []
        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) >= 64:
                entries.append((obj_meta, obj_bytes))
        if not entries:
            return None

        apk_ctx = extract_apk_context([(m.object_path, b) for m, b in entries])
        all_stat = []
        entry_boundaries, entry_names = [], []
        ridx = 0

        for eidx, (obj_meta, obj_bytes) in enumerate(entries):
            regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=eidx)
            cr = obj_meta.compressed_size / max(obj_meta.size, 1)
            start = ridx
            for region in regions:
                rd = obj_bytes[region.offset_start:region.offset_end]
                fv = extract_region_features(region, rd, len(obj_bytes), apk_ctx, cr)
                all_stat.append(fv.scalars)
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
            "apk_id": apk_id,
            "apk_label": apk_label,
            "stat_features": np.array(all_stat, dtype=np.float32),
            "entry_boundaries": entry_boundaries,
            "entry_names": entry_names,
            "diff_targets": diff_targets,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    model = StatOnlyModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    model = model.to(device)

    # Build training bags
    print("\nBuilding training bags...")
    t0 = time.time()
    train_bags = []
    origins = {}

    for p in sorted((HAPPER / "Origin-16").glob("*.apk"))[:30]:
        origins[p.stem] = p
        bag = build_stat_bag(p, 0, apk_id=f"benign_{p.stem}")
        if bag:
            train_bags.append(bag)

    if (TRACK_B / "benign").exists():
        for p in sorted((TRACK_B / "benign").glob("*.apk")):
            bag = build_stat_bag(p, 0, apk_id=f"tb_benign_{p.stem}")
            if bag:
                train_bags.append(bag)

    for family, dirname in [("Ali", "Ali-16"), ("Qihoo", "Qihoo-16"), ("Tencent", "Tencent-16")]:
        family_dir = HAPPER / dirname
        if not family_dir.exists():
            continue
        for p in sorted(family_dir.glob("*.apk"))[:15]:
            origin_match = None
            for ostem, opath in origins.items():
                if p.stem.startswith(ostem):
                    origin_match = opath
                    break
            diff = compute_paired_diff(origin_match, p) if origin_match else None
            bag = build_stat_bag(p, 1, diff, f"{family}_{p.stem}")
            if bag:
                train_bags.append(bag)

    n_packed = sum(1 for b in train_bags if b["apk_label"] == 1)
    print(f"Train: {len(train_bags)} bags ({n_packed} packed, "
          f"{len(train_bags)-n_packed} benign) [{time.time()-t0:.0f}s]")

    # Training
    print(f"\nTraining (epochs={args.epochs}, lr={args.lr})...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = np.random.RandomState(42)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        order = rng.permutation(len(train_bags))
        n_steps = 0
        packed_logits, benign_logits = [], []

        for i in range(0, len(train_bags), 4):
            batch_indices = order[i:i + 4]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_indices:
                bag = train_bags[idx]
                bag_logit, entry_attn, _, normality = model.forward_bag(bag, device)
                label = bag["apk_label"]
                target = torch.tensor(float(label), device=device)

                # L_bag
                l_bag = F.binary_cross_entropy_with_logits(bag_logit, target)

                # L_rank
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

                # L_align
                l_align = torch.tensor(0.0, device=device)
                if bag["diff_targets"] is not None and len(entry_attn) > 0:
                    dt = bag["diff_targets"][:len(entry_attn)]
                    if len(dt) < len(entry_attn):
                        dt = np.pad(dt, (0, len(entry_attn) - len(dt)))
                    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
                    if dt_t.sum() > 1e-6:
                        target_dist = torch.softmax(dt_t / 0.5, dim=0)
                        log_attn = torch.log(entry_attn.clamp(min=1e-8))
                        l_align = F.kl_div(log_attn, target_dist, reduction="batchmean")

                # L_normality
                l_norm = torch.tensor(0.0, device=device)
                if label == 0 and len(normality) > 0:
                    l_norm = F.mse_loss(normality, torch.ones_like(normality))

                loss = l_bag + 0.5 * l_rank + 0.3 * l_align + 0.2 * l_norm
                batch_loss = batch_loss + loss / len(batch_indices)

                logit_val = bag_logit.detach().item()
                if label == 1:
                    packed_logits.append(logit_val)
                else:
                    benign_logits.append(logit_val)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        sep = (np.mean(packed_logits) - np.mean(benign_logits)) if packed_logits and benign_logits else 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f} sep={sep:.3f}")

    # Evaluation
    print("\nEvaluation (cross-dataset)...")
    model.eval()
    test_bags = []

    if (HAPPER / "Oirgin-18").exists():
        for p in sorted((HAPPER / "Oirgin-18").glob("*.apk"))[:10]:
            bag = build_stat_bag(p, 0, apk_id=f"test_b_{p.stem}")
            if bag:
                test_bags.append(bag)

    packed_dir = TRACK_B / "packed"
    if packed_dir.exists():
        for item in sorted(packed_dir.iterdir()):
            if item.is_dir():
                for seed_dir in sorted(item.iterdir()):
                    if seed_dir.is_dir():
                        apk = seed_dir / "packed.apk"
                        if apk.exists():
                            bag = build_stat_bag(apk, 1, apk_id=f"tb_{item.name}_{seed_dir.name}")
                            if bag:
                                test_bags.append(bag)
            elif item.suffix == ".apk":
                bag = build_stat_bag(item, 1, apk_id=f"tb_{item.stem}")
                if bag:
                    test_bags.append(bag)

    y_true, y_score = [], []
    with torch.no_grad():
        for bag in test_bags:
            bag_logit, _, _, _ = model.forward_bag(bag, device)
            score = torch.sigmoid(bag_logit).item()
            y_true.append(bag["apk_label"])
            y_score.append(score)

    benign_s = [s for s, l in zip(y_score, y_true) if l == 0]
    packed_s = [s for s, l in zip(y_score, y_true) if l == 1]
    print(f"  Benign: n={len(benign_s)}, mean={np.mean(benign_s):.4f}")
    print(f"  Packed: n={len(packed_s)}, mean={np.mean(packed_s):.4f}")

    auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else None
    if auroc:
        print(f"  AUROC: {auroc:.4f}")
        print(f"\n  Comparison:")
        print(f"    Previous stat-only (v2): 0.7543")
        print(f"    This run (v3 pipeline): {auroc:.4f}")

    results = {"auroc": auroc, "benign_mean": float(np.mean(benign_s)),
               "packed_mean": float(np.mean(packed_s)), "n_test": len(test_bags)}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
