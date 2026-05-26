"""Full Framework Happer LOPO + Track B Cross-Dataset Evaluation.

Runs:
1. Happer 7-fold LOPO (hold out one packer family, test on it + unseen benign)
2. Track B cross-dataset evaluation (train on full Happer, test on Track B)

Reports APK AUROC + Entry MRR per fold.

Usage:
    python scripts/experiments/run_full_framework_happer.py [--epochs N] [--device DEVICE]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android_packer.apkio.objects import iter_apk_objects
from android_packer.features.full_feature_extractor import (
    extract_apk_context,
    extract_region_features,
)
from android_packer.labeling.happer_diff import compute_paired_diff
from android_packer.regioning.typed_slicer import iter_typed_regions
from android_packer.training.differential_trainer import (
    APKBag,
    DifferentialTrainerConfig,
    train_differential,
)

try:
    import torch
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"ERROR: {e}. Install: pip install torch scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HAPPER = ROOT / "data" / "happer_dataset" / "FSet"
TRACK_B = ROOT / "data" / "real_world" / "track_b"
OUT_DIR = ROOT / "outputs" / "experiments" / "full_framework_happer"

ORIGIN_DIRS = {"16": "Origin-16", "18": "Oirgin-18"}
PACKER_DIRS = {
    "Ali": [("Ali-16", "16")],
    "Baidu": [("Baidu-16", "16"), ("Baidu-18", "18")],
    "Bangcle": [("Bangcle-18", "18")],
    "Ijiami": [("Ijiami-16", "16"), ("Ijiami-18", "18")],
    "Kiwi": [("Kiwi-18", "18")],
    "Qihoo": [("Qihoo-16", "16"), ("Qihoo-18", "18")],
    "Tencent": [("Tencent-16", "16"), ("Tencent-18", "18")],
}


# ---------------------------------------------------------------------------
# Bag building
# ---------------------------------------------------------------------------


def build_bag_from_apk(
    apk_path: Path,
    apk_label: int,
    diff_result=None,
    apk_id: str = "",
) -> Optional[APKBag]:
    """Build an APKBag from a single APK file."""
    try:
        entries = []
        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) >= 64:
                entries.append((obj_meta, obj_bytes))

        if not entries:
            return None

        apk_ctx = extract_apk_context([(m.object_path, b) for m, b in entries])

        all_scalars, all_et, all_st = [], [], []
        entry_boundaries, entry_names = [], []
        region_idx = 0

        for entry_idx, (obj_meta, obj_bytes) in enumerate(entries):
            regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=entry_idx)
            comp_ratio = obj_meta.compressed_size / max(obj_meta.size, 1)

            start = region_idx
            for region in regions:
                region_data = obj_bytes[region.offset_start:region.offset_end]
                fv = extract_region_features(
                    region, region_data, len(obj_bytes), apk_ctx, comp_ratio
                )
                all_scalars.append(fv.scalars)
                all_et.append(fv.entry_type_id)
                all_st.append(fv.section_type_id)
                region_idx += 1
            end = region_idx
            if end > start:
                entry_boundaries.append((start, end))
                entry_names.append(obj_meta.object_path)

        if not entry_boundaries:
            return None

        # Diff targets
        diff_targets = None
        if diff_result is not None:
            diff_targets = np.zeros(len(entry_boundaries), dtype=np.float32)
            for i, name in enumerate(entry_names):
                if name in diff_result.entry_diffs:
                    diff_targets[i] = diff_result.entry_diffs[name].diff_score

        return APKBag(
            scalar_features=np.array(all_scalars, dtype=np.float32),
            entry_type_ids=np.array(all_et, dtype=np.int64),
            section_type_ids=np.array(all_st, dtype=np.int64),
            entry_boundaries=entry_boundaries,
            apk_label=apk_label,
            diff_targets=diff_targets,
            apk_id=apk_id,
        )
    except Exception as e:
        print(f"  ERROR building bag for {apk_path.name}: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Happer data loading
# ---------------------------------------------------------------------------


def load_happer_bags(limit_per_packer: int = 0) -> Tuple[
    Dict[str, List[APKBag]],  # packer_family → list of packed bags
    List[APKBag],             # benign bags (split by year)
]:
    """Load all Happer APKs as bags with diff labels."""
    print("[Happer] Loading and building bags...", flush=True)

    # Load origins (by year)
    origins: Dict[str, Dict[str, Path]] = {}  # year → {stem: path}
    for year, dir_name in ORIGIN_DIRS.items():
        origin_dir = HAPPER / dir_name
        if not origin_dir.exists():
            continue
        origins[year] = {}
        for apk_path in sorted(origin_dir.glob("*.apk")):
            origins[year][apk_path.stem] = apk_path

    # Build benign bags
    benign_bags: List[APKBag] = []
    for year, stem_paths in origins.items():
        for stem, apk_path in stem_paths.items():
            bag = build_bag_from_apk(apk_path, 0, apk_id=f"benign_{year}_{stem}")
            if bag is not None:
                benign_bags.append(bag)
    print(f"  Benign: {len(benign_bags)} bags", flush=True)

    # Build packed bags per family
    packed_bags: Dict[str, List[APKBag]] = {}
    for family, dirs in PACKER_DIRS.items():
        packed_bags[family] = []
        count = 0
        for dir_name, year in dirs:
            packer_dir = HAPPER / dir_name
            if not packer_dir.exists():
                continue

            for apk_path in sorted(packer_dir.glob("*.apk")):
                if limit_per_packer and count >= limit_per_packer:
                    break

                # Try to find matching origin for diff
                origin_path = _find_origin(apk_path, origins.get(year, {}))
                diff_result = None
                if origin_path is not None:
                    diff_result = compute_paired_diff(origin_path, apk_path)

                bag = build_bag_from_apk(
                    apk_path, 1, diff_result,
                    apk_id=f"{family}_{dir_name}_{apk_path.stem}"
                )
                if bag is not None:
                    packed_bags[family].append(bag)
                    count += 1

        print(f"  {family}: {len(packed_bags[family])} bags", flush=True)

    return packed_bags, benign_bags


def _find_origin(packed_path: Path, origins: Dict[str, Path]) -> Optional[Path]:
    """Try to find the origin APK for a packed APK via name matching."""
    stem = packed_path.stem

    # Try progressively shorter prefixes
    for origin_stem, origin_path in origins.items():
        if stem.startswith(origin_stem):
            return origin_path

    # Try known suffix stripping
    suffixes = ["_unsign_sign", "_legu_signed", "_legu_sign",
                "_protected_sign", "_unsigned_sign"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            candidate = stem[:-len(suffix)]
            if candidate in origins:
                return origins[candidate]

    # Qihoo/Ijiami pattern: {name}_{digits}_{packer}_sign
    for keyword in ["_jiagu_sign", "_ijiami_sign", "_kiwi_sign"]:
        if keyword in stem:
            prefix = stem[:stem.index(keyword)]
            parts = prefix.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                candidate = parts[0]
            else:
                candidate = prefix
            if candidate in origins:
                return origins[candidate]

    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    test_bags: List[APKBag],
    device: torch.device,
) -> Dict:
    """Evaluate model on test bags. Returns APK AUROC + Entry MRR."""
    model.eval()
    y_true, y_score = [], []
    entry_mrrs = []

    with torch.no_grad():
        for bag in test_bags:
            s = torch.tensor(bag.scalar_features, dtype=torch.float32, device=device)
            et = torch.tensor(bag.entry_type_ids, dtype=torch.long, device=device)
            st = torch.tensor(bag.section_type_ids, dtype=torch.long, device=device)

            out = model(s, et, st, bag.entry_boundaries)
            bag_logit, entry_attention = out[0], out[1]

            apk_score = torch.sigmoid(bag_logit).item()
            y_true.append(bag.apk_label)
            y_score.append(apk_score)

            # Entry MRR (only for packed bags with diff targets)
            if bag.apk_label == 1 and bag.diff_targets is not None:
                attn = entry_attention.cpu().numpy()
                dt = bag.diff_targets[:len(attn)]
                if (dt > 0.5).any():
                    # Rank entries by attention (descending)
                    ranked = np.argsort(-attn)
                    # Find first true-positive entry
                    for rank, idx in enumerate(ranked):
                        if idx < len(dt) and dt[idx] > 0.5:
                            entry_mrrs.append(1.0 / (rank + 1))
                            break

    results = {}

    if len(set(y_true)) >= 2:
        results["apk_auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        results["apk_auroc"] = None

    if entry_mrrs:
        results["entry_mrr"] = float(np.mean(entry_mrrs))
    else:
        results["entry_mrr"] = None

    results["n_test"] = len(test_bags)
    results["n_packed"] = sum(1 for b in test_bags if b.apk_label == 1)
    results["n_benign"] = sum(1 for b in test_bags if b.apk_label == 0)

    return results


# ---------------------------------------------------------------------------
# LOPO
# ---------------------------------------------------------------------------


def run_happer_lopo(
    packed_bags: Dict[str, List[APKBag]],
    benign_bags: List[APKBag],
    cfg: DifferentialTrainerConfig,
    device: torch.device,
) -> List[Dict]:
    """Run 7-fold LOPO on Happer (hold out one packer family per fold)."""
    families = sorted(packed_bags.keys())
    results = []

    # Split benign bags into train (70%) and test (30%) - fixed split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(benign_bags))
    n_test_benign = max(3, len(benign_bags) // 3)
    test_benign_idx = set(perm[:n_test_benign])
    train_benign = [b for i, b in enumerate(benign_bags) if i not in test_benign_idx]
    test_benign = [b for i, b in enumerate(benign_bags) if i in test_benign_idx]

    print(f"\n[LOPO] Benign split: {len(train_benign)} train, {len(test_benign)} test", flush=True)
    print(f"[LOPO] Families: {families}\n", flush=True)

    for held_out in families:
        print(f"[LOPO] Fold: held_out={held_out}", flush=True)

        # Train bags: other packers + train benign
        train_bags = list(train_benign)
        for fam, bags in packed_bags.items():
            if fam != held_out:
                train_bags.extend(bags)

        # Test bags: held-out packer + test benign
        test_bags = list(test_benign) + packed_bags[held_out]

        print(f"  Train: {len(train_bags)} bags, Test: {len(test_bags)} bags", flush=True)

        # Train
        t0 = time.time()
        model = train_differential(train_bags, cfg)
        train_time = time.time() - t0

        # Evaluate
        model = model.to(device)
        fold_result = evaluate_model(model, test_bags, device)
        fold_result["packer"] = held_out
        fold_result["train_time"] = round(train_time, 1)
        results.append(fold_result)

        auroc = fold_result.get("apk_auroc")
        mrr = fold_result.get("entry_mrr")
        print(f"  APK AUROC={auroc}  Entry MRR={mrr}  ({train_time:.0f}s)\n", flush=True)

    # Summary
    valid = [r for r in results if r["apk_auroc"] is not None]
    if valid:
        mean_auroc = np.mean([r["apk_auroc"] for r in valid])
        print(f"[LOPO] Mean APK AUROC ({len(valid)} folds): {mean_auroc:.4f}", flush=True)
    valid_mrr = [r for r in results if r["entry_mrr"] is not None]
    if valid_mrr:
        mean_mrr = np.mean([r["entry_mrr"] for r in valid_mrr])
        print(f"[LOPO] Mean Entry MRR ({len(valid_mrr)} folds): {mean_mrr:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-per-packer", type=int, default=0,
                        help="Limit APKs per packer (0=all, for debugging)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}", flush=True)

    cfg = DifferentialTrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=str(device),
        verbose=False,
    )

    # Load data
    packed_bags, benign_bags = load_happer_bags(limit_per_packer=args.limit_per_packer)

    # Run LOPO
    lopo_results = run_happer_lopo(packed_bags, benign_bags, cfg, device)

    # Save results
    out_file = OUT_DIR / "lopo_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "device": str(device),
            },
            "folds": lopo_results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
