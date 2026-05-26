"""Pseudo-code BERT pipeline v2 — with checkpoint resume + live progress.

Improvements over v1:
1. Checkpoint saving every N epochs (resume if interrupted)
2. Live progress file (queryable without interrupting training)
3. Chunked BERT forward (GPU memory efficient)
4. Separate stages with independent checkpoints

Progress file: outputs/experiments/pseudo_bert/progress.json
  Updated every epoch, readable anytime to check status.

Checkpoint files:
  pretrained_bert.pt — Stage 1 output (skip pretrain if exists)
  finetune_epoch_N.pt — Stage 3 intermediate (resume from latest)
  finetune_final.pt — Stage 3 final

Usage:
    python scripts/experiments/run_pseudo_bert_v2.py [options]

    # Resume from last checkpoint:
    python scripts/experiments/run_pseudo_bert_v2.py --resume

    # Check progress (non-blocking):
    python -c "import json; print(json.load(open('outputs/experiments/pseudo_bert/progress.json', encoding='utf-8')))"
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
# Progress tracking (live queryable)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Writes progress to a JSON file after each epoch. Non-blocking read."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "status": "initializing",
            "stage": "",
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "best_loss": float("inf"),
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "last_update": "",
            "history": [],
        }
        self._start_time = time.time()
        self._save()

    def update(self, stage: str, epoch: int, total_epochs: int, loss: float, **kwargs):
        elapsed = time.time() - self._start_time
        if epoch > 0:
            eta = elapsed / epoch * (total_epochs - epoch)
        else:
            eta = 0

        self.data.update({
            "status": "running",
            "stage": stage,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": round(loss, 6),
            "best_loss": min(self.data["best_loss"], loss),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        })
        self.data["history"].append({
            "stage": stage, "epoch": epoch, "loss": round(loss, 6),
            "time": round(elapsed, 1),
        })
        self._save()

    def set_status(self, status: str, **kwargs):
        self.data["status"] = status
        self.data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data.update(kwargs)
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)  # atomic on Windows


# ---------------------------------------------------------------------------
# Model definition (same as v1 but factored out)
# ---------------------------------------------------------------------------


class PseudoBERTFullModel(torch.nn.Module):
    """Complete model: FusionEncoder -> EntryAggregator -> APK MIL."""

    def __init__(self, fusion_cfg: FusionEncoderConfig):
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

    def forward_bag(self, bag: Dict, device: torch.device,
                    max_regions: int = 128, chunk_size: int = 32):
        """Forward pass with chunked BERT (GPU memory safe)."""
        n_regions = bag["stat_features"].shape[0]

        # Subsample
        if n_regions > max_regions:
            indices = np.random.choice(n_regions, max_regions, replace=False)
            indices.sort()
            idx_set = set(indices)
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
            return (torch.zeros((), device=device),
                    torch.zeros((0,), device=device),
                    torch.zeros((0,), device=device),
                    torch.zeros((0,), device=device))

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

        # Chunked BERT forward
        all_emb, all_susp, all_norm = [], [], []
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            s = slice(cs, ce)
            emb, susp, norm = self.fusion_encoder(
                d_ids[s], d_types[s], d_mask[s],
                n_ids[s], n_types[s], n_mask[s],
                b_ids[s], b_types[s], b_mask[s],
                stat[s],
            )
            all_emb.append(emb)
            all_susp.append(susp)
            all_norm.append(norm)

        embeddings = torch.cat(all_emb)
        suspicion = torch.cat(all_susp)
        normality = torch.cat(all_norm)

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
# Bag building (same as v1)
# ---------------------------------------------------------------------------


def build_bert_bag(apk_path, apk_label, tokenizer, diff_result=None, apk_id=""):
    """Build bag with stat features + token sequences."""
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
                d_enc, n_enc, b_enc = tokenizer.encode_region(
                    rd, entry_type=region.entry_type, dex_header_counts=hc,
                )
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
# Stage 3: Fine-tuning with checkpoint & progress
# ---------------------------------------------------------------------------


def finetune_with_checkpoints(
    model: PseudoBERTFullModel,
    train_bags: List[Dict],
    device: torch.device,
    progress: ProgressTracker,
    *,
    epochs: int = 25,
    lr: float = 5e-4,
    save_every: int = 5,
    resume_epoch: int = 0,
):
    """Fine-tune with periodic checkpoint saving and progress reporting."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.RandomState(42)

    for epoch in range(resume_epoch, epochs):
        epoch_loss = 0.0
        order = rng.permutation(len(train_bags))
        n_steps = 0

        for i in range(0, len(train_bags), 4):
            batch_indices = order[i:i + 4]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_indices:
                bag = train_bags[idx]
                bag_logit, entry_attn, entry_logits, _ = model.forward_bag(
                    bag, device, max_regions=128, chunk_size=32
                )

                # L_bag
                target = torch.tensor(float(bag["apk_label"]), device=device)
                l_bag = torch.nn.functional.binary_cross_entropy_with_logits(
                    bag_logit, target
                )

                # L_align
                l_align = torch.tensor(0.0, device=device)
                if bag["diff_targets"] is not None and len(entry_attn) > 0:
                    dt = bag["diff_targets"][:len(entry_attn)]
                    if len(dt) < len(entry_attn):
                        dt = np.pad(dt, (0, len(entry_attn) - len(dt)))
                    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
                    if dt_tensor.sum() > 0:
                        target_dist = torch.softmax(dt_tensor / 0.5, dim=0)
                        log_attn = torch.log(entry_attn.clamp(min=1e-8))
                        l_align = torch.nn.functional.kl_div(
                            log_attn, target_dist, reduction="batchmean"
                        )

                loss = l_bag + 0.3 * l_align
                batch_loss = batch_loss + loss / len(batch_indices)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)

        # Update progress (live queryable)
        progress.update(
            stage="finetune", epoch=epoch + 1, total_epochs=epochs, loss=avg_loss,
            n_train_bags=len(train_bags),
        )

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}", flush=True)

        # Save checkpoint every save_every epochs
        if (epoch + 1) % save_every == 0:
            ckpt = OUT_DIR / f"finetune_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt)

    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
    }, OUT_DIR / "finetune_final.pt")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest finetune checkpoint")
    parser.add_argument("--epochs-pretrain", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=25)
    parser.add_argument("--bert-layers", type=int, default=4)
    parser.add_argument("--bert-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    progress = ProgressTracker(OUT_DIR / "progress.json")
    progress.set_status("starting", device=str(device))

    # Build model
    fusion_cfg = FusionEncoderConfig(
        bert_hidden_dim=args.bert_dim, bert_n_layers=args.bert_layers,
        bert_max_length=args.max_length, bert_n_heads=8,
        bert_intermediate_dim=args.bert_dim * 2,
    )
    model = PseudoBERTFullModel(fusion_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Model: {n_params:,} params")

    # --- Stage 1: spMLM Pretraining ---
    pretrain_ckpt = OUT_DIR / "pretrained_bert.pt"

    if not args.skip_pretrain and not pretrain_ckpt.exists():
        print("\n=== Stage 1: spMLM Pretraining ===", flush=True)
        progress.set_status("pretrain_loading_corpus")

        # Load corpus
        max_len = args.max_length
        corpus = []
        if PRETRAIN_CACHE.exists() and (PRETRAIN_CACHE / "corpus_meta.json").exists():
            with open(PRETRAIN_CACHE / "corpus_meta.json") as f:
                meta = json.load(f)
            for chunk_idx in range(meta["n_chunks"]):
                chunk_path = PRETRAIN_CACHE / f"corpus_chunk_{chunk_idx:04d}.npz"
                if chunk_path.exists():
                    data = np.load(chunk_path)
                    for i in range(len(data["token_ids"])):
                        ids = data["token_ids"][i][:max_len].tolist()
                        types = data["token_type_ids"][i][:max_len].tolist()
                        mask = data["attention_mask"][i][:max_len].tolist()
                        abn = data["abnormal_mask"][i][:max_len].tolist()
                        pad_len = max_len - len(ids)
                        if pad_len > 0:
                            ids += [0] * pad_len
                            types += [0] * pad_len
                            mask += [0] * pad_len
                            abn += [0] * pad_len
                        corpus.append({
                            "token_ids": ids, "token_type_ids": types,
                            "attention_mask": mask, "abnormal_mask": abn,
                        })
            print(f"  Loaded {len(corpus)} sequences from cache")

        if not corpus:
            print("  WARNING: No corpus. Building from Happer benign...")
            tokenizer = PseudoCodeTokenizer(max_length=max_len)
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

        progress.set_status("pretrain_training")
        spmlm_cfg = SpMLMConfig(
            epochs=args.epochs_pretrain, batch_size=32, learning_rate=1e-4,
            device=str(device), use_fp16=(device.type == "cuda"),
        )
        model.fusion_encoder = train_spmlm(model.fusion_encoder, corpus, spmlm_cfg)
        torch.save(model.state_dict(), pretrain_ckpt)
        print(f"  Saved: {pretrain_ckpt}")

    elif pretrain_ckpt.exists():
        print(f"\n=== Loading pretrained checkpoint ===", flush=True)
        state = torch.load(pretrain_ckpt, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded from {pretrain_ckpt}")

    model = model.to(device)

    # --- Stage 3: Differential Fine-tuning ---
    print("\n=== Stage 3: Differential Fine-tuning ===", flush=True)
    progress.set_status("finetune_building_bags")

    tokenizer = PseudoCodeTokenizer(max_length=args.max_length)

    # Check for resume
    resume_epoch = 0
    if args.resume:
        # Find latest checkpoint
        ckpts = sorted(OUT_DIR.glob("finetune_epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            ckpt_data = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt_data["model_state_dict"])
            model = model.to(device)
            resume_epoch = ckpt_data["epoch"]
            print(f"  Resuming from epoch {resume_epoch} ({latest.name})")

    # Build bags (cache to disk for reuse)
    bags_cache = OUT_DIR / "train_bags.npz"
    if bags_cache.exists() and bags_cache.stat().st_size > 0:
        print("  Loading cached bags...", flush=True)
        # TODO: implement bag cache loading
        # For now, rebuild
        pass

    print("  Building training bags...", flush=True)
    t0 = time.time()
    train_bags = []

    # Happer benign (30)
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

    # Happer packed (Ali + Qihoo + Tencent, 15 each)
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
    print(f"  Train: {len(train_bags)} bags ({n_packed} packed, "
          f"{len(train_bags)-n_packed} benign) [{time.time()-t0:.0f}s]")

    # Fine-tune
    progress.set_status("finetune_training")
    print("  Fine-tuning...", flush=True)
    model = finetune_with_checkpoints(
        model, train_bags, device, progress,
        epochs=args.epochs_finetune, lr=5e-4,
        save_every=args.save_every, resume_epoch=resume_epoch,
    )

    # --- Evaluation ---
    print("\n=== Evaluation ===", flush=True)
    progress.set_status("evaluating")
    model.eval()

    test_bags = []
    for p in sorted((HAPPER / "Oirgin-18").glob("*.apk"))[:10]:
        bag = build_bert_bag(p, 0, tokenizer, apk_id=f"test_b_{p.stem}")
        if bag:
            test_bags.append(bag)

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
            bag_logit, _, _, _ = model.forward_bag(bag, device)
            score = torch.sigmoid(bag_logit).item()
            y_true.append(bag["apk_label"])
            y_score.append(score)

    benign_s = [s for s, l in zip(y_score, y_true) if l == 0]
    packed_s = [s for s, l in zip(y_score, y_true) if l == 1]
    print(f"  Benign: mean={np.mean(benign_s):.4f}")
    print(f"  Packed: mean={np.mean(packed_s):.4f}")

    auroc = None
    if len(set(y_true)) >= 2:
        auroc = roc_auc_score(y_true, y_score)
        print(f"  APK AUROC: {auroc:.4f}")
        print(f"\n  Comparison:")
        print(f"    Entropy baseline:       0.7246")
        print(f"    Stat-only framework:    0.7543")
        print(f"    Pseudo-code BERT:       {auroc:.4f}")

    # Final progress
    progress.set_status("completed", auroc=auroc,
                        benign_mean=float(np.mean(benign_s)),
                        packed_mean=float(np.mean(packed_s)))

    results = {
        "auroc": auroc,
        "benign_mean": float(np.mean(benign_s)),
        "packed_mean": float(np.mean(packed_s)),
        "n_test": len(test_bags),
    }
    with open(OUT_DIR / "results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR / 'results_v2.json'}")


if __name__ == "__main__":
    main()
