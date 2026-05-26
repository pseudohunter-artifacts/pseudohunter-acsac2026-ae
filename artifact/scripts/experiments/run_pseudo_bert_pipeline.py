"""End-to-end Pseudo-code BERT training: spMLM pretrain + Stage 3 diff fine-tune.

Orchestrates the full pipeline:
1. Load pretrained corpus cache
2. spMLM pretrain the FusionEncoder
3. Integrate with entry_aggregator + APK MIL
4. Stage 3 differential fine-tuning on Happer pairs
5. Evaluation on Track B

Usage:
    python scripts/experiments/run_pseudo_bert_pipeline.py [--skip-pretrain] [--epochs-pretrain N]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sklearn.metrics import roc_auc_score

from android_packer.apkio.objects import iter_apk_objects
from android_packer.decoders.pseudo_tokenizer import PseudoCodeTokenizer
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
from android_packer.models.fusion_encoder import FusionEncoderConfig, build_fusion_encoder
from android_packer.regioning.typed_slicer import iter_typed_regions
from android_packer.training.pretrain_spmlm import SpMLMConfig, train_spmlm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HAPPER = ROOT / "data" / "happer_dataset" / "FSet"
TRACK_B = ROOT / "data" / "real_world" / "track_b"
PRETRAIN_CACHE = ROOT / "data" / "pretrain_cache"
OUT_DIR = ROOT / "outputs" / "experiments" / "pseudo_bert"


# ---------------------------------------------------------------------------
# Data building: APK → tokenized bags for BERT
# ---------------------------------------------------------------------------


def build_bert_bag(
    apk_path: Path,
    apk_label: int,
    tokenizer: PseudoCodeTokenizer,
    diff_result=None,
    apk_id: str = "",
) -> Optional[Dict]:
    """Build a bag with both stat features AND token sequences for each region."""
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

        # Collect per-region data
        all_stat_features = []
        all_dalvik_ids = []
        all_native_ids = []
        all_byte_ids = []
        all_dalvik_types = []
        all_native_types = []
        all_byte_types = []
        all_dalvik_mask = []
        all_native_mask = []
        all_byte_mask = []
        entry_boundaries = []
        entry_names = []
        region_idx = 0

        for entry_idx, (obj_meta, obj_bytes) in enumerate(entries):
            regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=entry_idx)
            comp_ratio = obj_meta.compressed_size / max(obj_meta.size, 1)
            header_counts = dex_counts.get(obj_meta.object_path, (0, 0, 0, 0))

            start = region_idx
            for region in regions:
                region_data = obj_bytes[region.offset_start:region.offset_end]

                # Stat features
                fv = extract_region_features(
                    region, region_data, len(obj_bytes), apk_ctx, comp_ratio
                )
                all_stat_features.append(fv.scalars)

                # Token sequences
                dalvik_enc, native_enc, byte_enc = tokenizer.encode_region(
                    region_data,
                    entry_type=region.entry_type,
                    dex_header_counts=header_counts,
                )
                all_dalvik_ids.append(dalvik_enc.token_ids)
                all_native_ids.append(native_enc.token_ids)
                all_byte_ids.append(byte_enc.token_ids)
                all_dalvik_types.append(dalvik_enc.token_type_ids)
                all_native_types.append(native_enc.token_type_ids)
                all_byte_types.append(byte_enc.token_type_ids)
                all_dalvik_mask.append(dalvik_enc.attention_mask)
                all_native_mask.append(native_enc.attention_mask)
                all_byte_mask.append(byte_enc.attention_mask)

                region_idx += 1

            if region_idx > start:
                entry_boundaries.append((start, region_idx))
                entry_names.append(obj_meta.object_path)

        if not entry_boundaries:
            return None

        # Diff targets
        diff_targets = None
        if diff_result:
            diff_targets = np.zeros(len(entry_boundaries), dtype=np.float32)
            for i, name in enumerate(entry_names):
                if name in diff_result.entry_diffs:
                    diff_targets[i] = diff_result.entry_diffs[name].diff_score

        return {
            "apk_id": apk_id,
            "apk_label": apk_label,
            "stat_features": np.array(all_stat_features, dtype=np.float32),
            "dalvik_ids": np.array(all_dalvik_ids, dtype=np.int64),
            "native_ids": np.array(all_native_ids, dtype=np.int64),
            "byte_ids": np.array(all_byte_ids, dtype=np.int64),
            "dalvik_types": np.array(all_dalvik_types, dtype=np.int64),
            "native_types": np.array(all_native_types, dtype=np.int64),
            "byte_types": np.array(all_byte_types, dtype=np.int64),
            "dalvik_mask": np.array(all_dalvik_mask, dtype=np.float32),
            "native_mask": np.array(all_native_mask, dtype=np.float32),
            "byte_mask": np.array(all_byte_mask, dtype=np.float32),
            "entry_boundaries": entry_boundaries,
            "diff_targets": diff_targets,
        }
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Complete model (FusionEncoder + EntryAggregator + APK MIL)
# ---------------------------------------------------------------------------


class PseudoBERTFullModel(torch.nn.Module):
    """Complete model: FusionEncoder → EntryAggregator → APK MIL."""

    def __init__(self, fusion_cfg: FusionEncoderConfig):
        super().__init__()
        self.fusion_encoder = build_fusion_encoder(fusion_cfg)

        agg_cfg = EntryAggregatorConfig(
            region_dim=fusion_cfg.output_dim,
            entry_dim=fusion_cfg.output_dim,
            attn_hidden=128,
            dropout=0.1,
        )
        self.entry_aggregator = build_entry_aggregator(agg_cfg)

        mil_cfg = APKMILConfig(
            entry_dim=fusion_cfg.output_dim,
            attn_hidden=128,
            dropout=0.1,
            use_normality=True,
        )
        self.apk_mil = build_apk_mil(mil_cfg)

    def forward(self, bag: Dict, device: torch.device, max_regions: int = 128):
        """Forward pass for one APK bag.

        Returns: (bag_logit, entry_attention, entry_logits, region_suspicion)
        """
        n_regions = bag["stat_features"].shape[0]

        # Subsample if too large
        if n_regions > max_regions:
            indices = np.random.choice(n_regions, max_regions, replace=False)
            indices.sort()
            # Recompute entry boundaries
            idx_set = set(indices)
            new_idx_map = {old: new for new, old in enumerate(indices)}
            new_boundaries = []
            for start, end in bag["entry_boundaries"]:
                entry_idx = [new_idx_map[i] for i in range(start, end) if i in idx_set]
                if entry_idx:
                    new_boundaries.append((entry_idx[0], entry_idx[-1] + 1))
            entry_boundaries = new_boundaries
        else:
            indices = np.arange(n_regions)
            entry_boundaries = bag["entry_boundaries"]

        N = len(indices)

        # To tensors
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

        # Process BERT in chunks to avoid OOM (8GB GPU limitation)
        chunk_size = 32  # 32 regions per BERT forward pass
        all_embeddings = []
        all_suspicion = []
        all_normality = []

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            cs = slice(chunk_start, chunk_end)

            emb_chunk, susp_chunk, norm_chunk = self.fusion_encoder(
                d_ids[cs], d_types[cs], d_mask[cs],
                n_ids[cs], n_types[cs], n_mask[cs],
                b_ids[cs], b_types[cs], b_mask[cs],
                stat[cs],
            )
            all_embeddings.append(emb_chunk)
            all_suspicion.append(susp_chunk)
            all_normality.append(norm_chunk)

        embeddings = torch.cat(all_embeddings, dim=0)
        suspicion = torch.cat(all_suspicion, dim=0)
        normality = torch.cat(all_normality, dim=0)

        # Entry aggregation
        entry_embeddings, entry_suspicions, region_attn = self.entry_aggregator(
            embeddings, suspicion, entry_boundaries
        )

        # Entry normality
        entry_norms = []
        for start, end in entry_boundaries:
            if start < end:
                entry_norms.append(normality[start:end].mean())
            else:
                entry_norms.append(torch.tensor(0.5, device=device))
        entry_normality = torch.stack(entry_norms)

        # APK MIL
        bag_logit, entry_attention, entry_logits = self.apk_mil(
            entry_embeddings, entry_normality
        )

        return bag_logit, entry_attention, entry_logits, suspicion


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--epochs-pretrain", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=25)
    parser.add_argument("--bert-layers", type=int, default=4)
    parser.add_argument("--bert-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    # Build model
    fusion_cfg = FusionEncoderConfig(
        bert_hidden_dim=args.bert_dim,
        bert_n_layers=args.bert_layers,
        bert_max_length=args.max_length,
        bert_n_heads=8,
        bert_intermediate_dim=args.bert_dim * 2,
    )
    model = PseudoBERTFullModel(fusion_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # --- Stage 1: spMLM Pretraining ---
    if not args.skip_pretrain:
        print("\n=== Stage 1: spMLM Pretraining ===")
        # Load corpus
        if PRETRAIN_CACHE.exists() and (PRETRAIN_CACHE / "corpus_meta.json").exists():
            with open(PRETRAIN_CACHE / "corpus_meta.json") as f:
                meta = json.load(f)
            print(f"  Corpus: {meta['n_sequences']} sequences from {meta['n_apks']} APKs")

            corpus = []
            max_len = args.max_length
            for chunk_idx in range(meta["n_chunks"]):
                chunk_path = PRETRAIN_CACHE / f"corpus_chunk_{chunk_idx:04d}.npz"
                if chunk_path.exists():
                    data = np.load(chunk_path)
                    for i in range(len(data["token_ids"])):
                        # Truncate to model's max_length
                        ids = data["token_ids"][i][:max_len].tolist()
                        types = data["token_type_ids"][i][:max_len].tolist()
                        mask = data["attention_mask"][i][:max_len].tolist()
                        abn = data["abnormal_mask"][i][:max_len].tolist()
                        # Pad if shorter
                        pad_len = max_len - len(ids)
                        if pad_len > 0:
                            ids += [0] * pad_len
                            types += [0] * pad_len
                            mask += [0] * pad_len
                            abn += [0] * pad_len
                        corpus.append({
                            "token_ids": ids,
                            "token_type_ids": types,
                            "attention_mask": mask,
                            "abnormal_mask": abn,
                        })
            print(f"  Loaded {len(corpus)} sequences from cache")
        else:
            print("  WARNING: No corpus cache found. Using Happer benign directly.")
            # Quick corpus from Happer
            tokenizer = PseudoCodeTokenizer(max_length=args.max_length)
            corpus = []
            for apk_path in sorted((HAPPER / "Origin-16").glob("*.apk"))[:20]:
                for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
                    if len(obj_bytes) < 256:
                        continue
                    for region in iter_typed_regions(obj_meta, obj_bytes, entry_index=0)[:5]:
                        rd = obj_bytes[region.offset_start:region.offset_end]
                        for enc in tokenizer.encode_region(rd, entry_type=region.entry_type):
                            corpus.append({
                                "token_ids": enc.token_ids,
                                "token_type_ids": enc.token_type_ids,
                                "attention_mask": enc.attention_mask,
                                "abnormal_mask": [0] * len(enc.token_ids),
                            })
            print(f"  Built {len(corpus)} sequences on-the-fly")

        # Train
        spmlm_cfg = SpMLMConfig(
            epochs=args.epochs_pretrain,
            batch_size=32,
            learning_rate=1e-4,
            device=str(device),
            use_fp16=(device.type == "cuda"),
        )
        model.fusion_encoder = train_spmlm(model.fusion_encoder, corpus, spmlm_cfg)
        # Save checkpoint
        ckpt_path = OUT_DIR / "pretrained_bert.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved pretrained checkpoint: {ckpt_path}")

        # Reload model to fresh CUDA state (avoids CUBLAS corruption from prior run)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            model_state = torch.load(ckpt_path, map_location="cpu")
            model = PseudoBERTFullModel(fusion_cfg)
            model.load_state_dict(model_state)
            model = model.to(device)
            print("  Reloaded model to fresh CUDA state")

    # --- Stage 3: Differential Fine-tuning ---
    print("\n=== Stage 3: Differential Fine-tuning ===")
    tokenizer = PseudoCodeTokenizer(max_length=args.max_length)

    # Build bags from Happer pairs + Track B benign
    print("  Building training bags...")
    t0 = time.time()
    train_bags = []

    # Happer benign
    origins = {}
    for p in sorted((HAPPER / "Origin-16").glob("*.apk"))[:30]:
        origins[p.stem] = p
        bag = build_bert_bag(p, 0, tokenizer, apk_id=f"benign_{p.stem}")
        if bag:
            train_bags.append(bag)

    # Track B benign
    for p in sorted((TRACK_B / "benign").glob("*.apk")):
        bag = build_bert_bag(p, 0, tokenizer, apk_id=f"tb_benign_{p.stem}")
        if bag:
            train_bags.append(bag)

    # Happer packed (Ali + Qihoo + Tencent with diff)
    for family, dirname in [("Ali", "Ali-16"), ("Qihoo", "Qihoo-16"), ("Tencent", "Tencent-16")]:
        for p in sorted((HAPPER / dirname).glob("*.apk"))[:15]:
            origin_match = None
            for ostem, opath in origins.items():
                if p.stem.startswith(ostem):
                    origin_match = opath
                    break
            diff = compute_paired_diff(origin_match, p) if origin_match else None
            bag = build_bert_bag(p, 1, tokenizer, diff, f"{family}_{p.stem}")
            if bag:
                train_bags.append(bag)

    n_packed = sum(1 for b in train_bags if b["apk_label"] == 1)
    n_benign = len(train_bags) - n_packed
    print(f"  Train: {len(train_bags)} bags ({n_packed} packed, {n_benign} benign) [{time.time()-t0:.0f}s]")

    # Fine-tune with differential loss
    print("  Fine-tuning...")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.RandomState(42)

    for epoch in range(args.epochs_finetune):
        epoch_loss = 0.0
        order = rng.permutation(len(train_bags))

        for i in range(0, len(train_bags), 4):
            batch_indices = order[i:i+4]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_indices:
                bag = train_bags[idx]
                bag_logit, entry_attn, entry_logits, _ = model.forward(bag, device, max_regions=256)

                # L_bag
                target = torch.tensor(float(bag["apk_label"]), device=device)
                l_bag = torch.nn.functional.binary_cross_entropy_with_logits(bag_logit, target)

                # L_align (if diff targets available)
                l_align = torch.tensor(0.0, device=device)
                if bag["diff_targets"] is not None:
                    dt = bag["diff_targets"][:len(entry_attn)]
                    if len(dt) < len(entry_attn):
                        dt = np.pad(dt, (0, len(entry_attn) - len(dt)))
                    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
                    if dt_tensor.sum() > 0:
                        target_dist = torch.softmax(dt_tensor / 0.5, dim=0)
                        log_attn = torch.log(entry_attn.clamp(min=1e-8))
                        l_align = torch.nn.functional.kl_div(log_attn, target_dist, reduction="batchmean")

                loss = l_bag + 0.3 * l_align
                batch_loss = batch_loss + loss / len(batch_indices)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()

        if (epoch + 1) % 5 == 0:
            n_batches = max(len(train_bags) // 4, 1)
            print(f"    Epoch {epoch+1}/{args.epochs_finetune}: loss={epoch_loss/n_batches:.4f}", flush=True)

    # --- Evaluation ---
    print("\n=== Evaluation: Cross-dataset ===")
    model.eval()

    test_bags = []
    # Test benign: Oirgin-18
    for p in sorted((HAPPER / "Oirgin-18").glob("*.apk"))[:10]:
        bag = build_bert_bag(p, 0, tokenizer, apk_id=f"test_benign_{p.stem}")
        if bag:
            test_bags.append(bag)

    # Test packed: Track B
    packed_dir = TRACK_B / "packed"
    for item in sorted(packed_dir.iterdir()):
        if item.is_dir():
            for seed_dir in sorted(item.iterdir()):
                if seed_dir.is_dir():
                    apk = seed_dir / "packed.apk"
                    if apk.exists():
                        bag = build_bert_bag(apk, 1, tokenizer, apk_id=f"tb_{item.name}_{seed_dir.name}")
                        if bag:
                            test_bags.append(bag)
        elif item.suffix == ".apk":
            bag = build_bert_bag(item, 1, tokenizer, apk_id=f"tb_{item.stem}")
            if bag:
                test_bags.append(bag)

    print(f"  Test: {len(test_bags)} bags")

    y_true, y_score = [], []
    with torch.no_grad():
        for bag in test_bags:
            bag_logit, _, _, _ = model.forward(bag, device, max_regions=512)
            score = torch.sigmoid(bag_logit).item()
            y_true.append(bag["apk_label"])
            y_score.append(score)

    benign_s = [s for s, l in zip(y_score, y_true) if l == 0]
    packed_s = [s for s, l in zip(y_score, y_true) if l == 1]
    print(f"  Benign: mean={np.mean(benign_s):.4f}")
    print(f"  Packed: mean={np.mean(packed_s):.4f}")

    if len(set(y_true)) >= 2:
        auroc = roc_auc_score(y_true, y_score)
        print(f"  APK AUROC: {auroc:.4f}")
        print(f"\n  Comparison:")
        print(f"    Entropy baseline:       0.7246")
        print(f"    Stat-only framework:    0.7543")
        print(f"    Pseudo-code BERT:       {auroc:.4f}")

    # Save results
    results = {
        "auroc": auroc if len(set(y_true)) >= 2 else None,
        "benign_mean": float(np.mean(benign_s)),
        "packed_mean": float(np.mean(packed_s)),
        "n_test": len(test_bags),
        "config": {
            "bert_layers": args.bert_layers,
            "bert_dim": args.bert_dim,
            "max_length": args.max_length,
            "epochs_pretrain": args.epochs_pretrain,
            "epochs_finetune": args.epochs_finetune,
        },
    }
    with open(OUT_DIR / "pseudo_bert_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR / 'pseudo_bert_results.json'}")


if __name__ == "__main__":
    main()
