"""Block 1 (Fixed): XGBoost Decision Gate with proper evaluation protocol.

Fixes the evaluation flaw: benign samples must NOT appear in both train and test.
Two evaluation modes:

1. LOPO-SeedSplit: Hold out one packer family + hold out 30% of seed apps for test benign.
   Train never sees test-benign apps in any form (neither packed nor unpacked).

2. Cross-Dataset: Train on full Happer, test on Track B (completely different apps + packers).
   This is the GOLD STANDARD for generalization.

Usage:
    python scripts/experiments/happer_xgb_decision_gate_v2.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "outputs" / "experiments" / "happer_ds_amil"
FEATURES_FILE = DATA_DIR / "entry_features.jsonl"
APK_INDEX_FILE = DATA_DIR / "apk_index.json"

TRACK_B = ROOT / "data" / "real_world" / "track_b"

HAPPER_DIR = ROOT / "data" / "happer_dataset" / "FSet"

PACKER_FAMILIES = ["Ali", "Baidu", "Bangcle", "Ijiami", "Kiwi", "Qihoo", "Tencent"]


# ---------------------------------------------------------------------------
# Feature extraction (lightweight, for Track B)
# ---------------------------------------------------------------------------


def compute_entropy(data: bytes) -> float:
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    n = len(data)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def compute_byte_histogram(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(256, dtype=np.float32)
    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1
    return (counts / len(data)).astype(np.float32)


def extract_apk_entry_histograms(apk_path: Path) -> Optional[List[np.ndarray]]:
    """Extract per-entry byte histograms from an APK."""
    try:
        with zipfile.ZipFile(apk_path) as zf:
            histograms = []
            for info in zf.infolist():
                if info.is_dir() or info.file_size == 0:
                    continue
                if info.file_size > 50 * 1024 * 1024:
                    continue
                try:
                    data = zf.read(info.filename)
                    if len(data) < 64:
                        continue
                    hist = compute_byte_histogram(data)
                    histograms.append(hist)
                except Exception:
                    continue
            return histograms if histograms else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Seed identification for proper splitting
# ---------------------------------------------------------------------------


def identify_seed_from_apk_id(apk_id: str) -> str:
    """Extract seed identity from apk_id.

    apk_id format: "{family}__{dir}__{filename_stem}" for packed
                   "benign__{dir}__{filename_stem}" for benign

    We need to match packed APKs to their benign counterpart.
    Since naming varies by packer, we use a simplified approach:
    group benign APKs by their stem, and track which stems appear
    in each packer directory.
    """
    parts = apk_id.split("__")
    if len(parts) >= 3:
        return parts[2]  # filename stem
    return apk_id


def build_seed_groups(apk_index: Dict) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Identify seed apps and which packers each seed was packed with.

    Returns:
      - benign_seeds: set of filename stems for benign APKs
      - seed_to_packers: {seed_stem: set of packer families that packed it}
    """
    benign_seeds = set()
    seed_to_packers: Dict[str, Set[str]] = {}

    for apk_id, info in apk_index.items():
        stem = identify_seed_from_apk_id(apk_id)
        family = info["family"]
        if family == "benign":
            benign_seeds.add(stem)
        else:
            if stem not in seed_to_packers:
                seed_to_packers[stem] = set()
            seed_to_packers[stem].add(family)

    return benign_seeds, seed_to_packers


# ---------------------------------------------------------------------------
# Proper LOPO with seed-stratified benign split
# ---------------------------------------------------------------------------


def run_lopo_seed_split(
    apk_entries: Dict[str, List[Dict]],
    apk_index: Dict,
    feature_slice: slice = slice(0, 256),
    test_benign_ratio: float = 0.3,
    n_repeats: int = 5,
) -> Dict:
    """LOPO with held-out benign seeds.

    For each fold (hold out packer P):
      - Randomly split benign seeds into train (70%) and test (30%)
      - Train: packed APKs from other packers (only using train seeds) + benign train seeds
      - Test: packed APKs from P (only test seeds available) + benign test seeds
      - CRITICAL: Test seeds never appear in training in ANY form

    Repeat n_repeats times with different random benign splits for stability.
    """
    benign_seeds, _ = build_seed_groups(apk_index)
    benign_seeds_list = sorted(benign_seeds)

    # Group APKs by family and seed
    apk_by_family_seed: Dict[str, Dict[str, List[str]]] = {}
    for apk_id, info in apk_index.items():
        family = info["family"]
        seed = identify_seed_from_apk_id(apk_id)
        if family not in apk_by_family_seed:
            apk_by_family_seed[family] = {}
        if seed not in apk_by_family_seed[family]:
            apk_by_family_seed[family][seed] = []
        apk_by_family_seed[family][seed].append(apk_id)

    results = {}
    rng = np.random.RandomState(42)

    for held_out in PACKER_FAMILIES:
        fold_aurocs = []

        for repeat in range(n_repeats):
            # Split benign seeds
            perm = rng.permutation(len(benign_seeds_list))
            n_test = max(1, int(len(benign_seeds_list) * test_benign_ratio))
            test_seed_indices = perm[:n_test]
            train_seed_indices = perm[n_test:]

            test_seeds = set(benign_seeds_list[i] for i in test_seed_indices)
            train_seeds = set(benign_seeds_list[i] for i in train_seed_indices)

            # Build train set: non-held-out packers (train seeds only) + benign (train seeds)
            train_ids = []
            for family in list(PACKER_FAMILIES) + ["benign"]:
                if family == held_out:
                    continue
                if family not in apk_by_family_seed:
                    continue
                for seed, ids in apk_by_family_seed[family].items():
                    if seed in train_seeds:
                        train_ids.extend(ids)

            # Build test set: held-out packer (test seeds) + benign (test seeds)
            test_ids = []
            # Held-out packer: use test seeds if available, otherwise all
            if held_out in apk_by_family_seed:
                for seed, ids in apk_by_family_seed[held_out].items():
                    if seed in test_seeds:
                        test_ids.extend(ids)
            # Benign test seeds
            if "benign" in apk_by_family_seed:
                for seed, ids in apk_by_family_seed["benign"].items():
                    if seed in test_seeds:
                        test_ids.extend(ids)

            # Build feature matrices
            X_train, y_train = [], []
            for apk_id in train_ids:
                if apk_id not in apk_entries:
                    continue
                entries = apk_entries[apk_id]
                feats = np.array([e["features"][feature_slice] for e in entries])
                X_train.append(feats.max(axis=0))
                y_train.append(1 if apk_index[apk_id]["is_packed"] else 0)

            X_test, y_test = [], []
            for apk_id in test_ids:
                if apk_id not in apk_entries:
                    continue
                entries = apk_entries[apk_id]
                feats = np.array([e["features"][feature_slice] for e in entries])
                X_test.append(feats.max(axis=0))
                y_test.append(1 if apk_index[apk_id]["is_packed"] else 0)

            if not X_train or not X_test:
                continue

            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test)

            if len(set(y_test)) < 2 or len(set(y_train)) < 2:
                continue

            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42 + repeat
            )
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_score)
            fold_aurocs.append(auroc)

        if fold_aurocs:
            results[held_out] = {
                "mean": float(np.mean(fold_aurocs)),
                "std": float(np.std(fold_aurocs)),
                "n_repeats": len(fold_aurocs),
            }

    return results


# ---------------------------------------------------------------------------
# Cross-dataset: Happer train -> Track B test
# ---------------------------------------------------------------------------


def run_cross_dataset(
    apk_entries: Dict[str, List[Dict]],
    apk_index: Dict,
    feature_slice: slice = slice(0, 256),
) -> Dict:
    """Train on full Happer, test on Track B (gold standard)."""

    # Build Happer training set
    X_train, y_train = [], []
    for apk_id, entries in apk_entries.items():
        feats = np.array([e["features"][feature_slice] for e in entries])
        X_train.append(feats.max(axis=0))
        y_train.append(1 if apk_index[apk_id]["is_packed"] else 0)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train, y_train)

    # Parse Track B APKs
    track_b_data = []

    # Benign
    benign_dir = TRACK_B / "benign"
    if benign_dir.exists():
        for apk_path in sorted(benign_dir.glob("*.apk")):
            hists = extract_apk_entry_histograms(apk_path)
            if hists:
                feat = np.array(hists).max(axis=0)
                track_b_data.append({"label": 0, "features": feat, "family": "benign", "name": apk_path.name})

    # Packed (subdirectory structure)
    packed_dir = TRACK_B / "packed"
    if packed_dir.exists():
        for item in sorted(packed_dir.iterdir()):
            if item.is_dir():
                packer_name = item.name
                for seed_dir in sorted(item.iterdir()):
                    if seed_dir.is_dir():
                        packed_apk = seed_dir / "packed.apk"
                        if packed_apk.exists():
                            hists = extract_apk_entry_histograms(packed_apk)
                            if hists:
                                feat = np.array(hists).max(axis=0)
                                track_b_data.append({"label": 1, "features": feat, "family": packer_name, "name": f"{packer_name}/{seed_dir.name}"})
            elif item.suffix == ".apk":
                packer_name = item.stem.split("__")[0]
                hists = extract_apk_entry_histograms(item)
                if hists:
                    feat = np.array(hists).max(axis=0)
                    track_b_data.append({"label": 1, "features": feat, "family": packer_name, "name": item.name})

    if not track_b_data:
        return {"error": "No Track B data found"}

    X_test = np.array([d["features"] for d in track_b_data], dtype=np.float32)
    y_test = np.array([d["label"] for d in track_b_data])

    if len(set(y_test)) < 2:
        return {"error": "Single class in Track B test"}

    # Slice features to match training
    if feature_slice != slice(None) and X_test.shape[1] != X_train.shape[1]:
        # Track B features are raw 256-dim histograms
        pass  # Already correct if feature_slice is 0:256

    y_score = clf.predict_proba(X_test)[:, 1]
    overall_auroc = roc_auc_score(y_test, y_score)

    # Per-family
    per_family = {}
    families = sorted(set(d["family"] for d in track_b_data if d["label"] == 1))
    for fam in families:
        fam_data = [d for d in track_b_data if d["family"] == fam or d["family"] == "benign"]
        if len(set(d["label"] for d in fam_data)) < 2:
            continue
        X_f = np.array([d["features"] for d in fam_data], dtype=np.float32)
        y_f = np.array([d["label"] for d in fam_data])
        y_s = clf.predict_proba(X_f)[:, 1]
        per_family[fam] = float(roc_auc_score(y_f, y_s))

    return {
        "overall_auroc": float(overall_auroc),
        "per_family": per_family,
        "n_test": len(track_b_data),
        "n_packed": int(y_test.sum()),
        "n_benign": int((1 - y_test).sum()),
    }


# ---------------------------------------------------------------------------
# Entropy baseline (no training)
# ---------------------------------------------------------------------------


def run_entropy_cross_dataset() -> Dict:
    """Max-entry-entropy as APK score, tested on Track B."""
    track_b_data = []

    benign_dir = TRACK_B / "benign"
    if benign_dir.exists():
        for apk_path in sorted(benign_dir.glob("*.apk")):
            try:
                with zipfile.ZipFile(apk_path) as zf:
                    max_ent = 0.0
                    for info in zf.infolist():
                        if info.is_dir() or info.file_size < 64:
                            continue
                        try:
                            data = zf.read(info.filename)
                            max_ent = max(max_ent, compute_entropy(data))
                        except:
                            pass
                    track_b_data.append({"label": 0, "score": max_ent, "family": "benign"})
            except:
                pass

    packed_dir = TRACK_B / "packed"
    if packed_dir.exists():
        for item in sorted(packed_dir.iterdir()):
            if item.is_dir():
                packer_name = item.name
                for seed_dir in sorted(item.iterdir()):
                    if seed_dir.is_dir():
                        packed_apk = seed_dir / "packed.apk"
                        if packed_apk.exists():
                            try:
                                with zipfile.ZipFile(packed_apk) as zf:
                                    max_ent = 0.0
                                    for info in zf.infolist():
                                        if info.is_dir() or info.file_size < 64:
                                            continue
                                        try:
                                            data = zf.read(info.filename)
                                            max_ent = max(max_ent, compute_entropy(data))
                                        except:
                                            pass
                                    track_b_data.append({"label": 1, "score": max_ent, "family": packer_name})
                            except:
                                pass
            elif item.suffix == ".apk":
                packer_name = item.stem.split("__")[0]
                try:
                    with zipfile.ZipFile(item) as zf:
                        max_ent = 0.0
                        for info in zf.infolist():
                            if info.is_dir() or info.file_size < 64:
                                continue
                            try:
                                data = zf.read(info.filename)
                                max_ent = max(max_ent, compute_entropy(data))
                            except:
                                pass
                        track_b_data.append({"label": 1, "score": max_ent, "family": packer_name})
                except:
                    pass

    if not track_b_data or len(set(d["label"] for d in track_b_data)) < 2:
        return {"error": "insufficient data"}

    y_true = np.array([d["label"] for d in track_b_data])
    y_score = np.array([d["score"] for d in track_b_data])
    return {"overall_auroc": float(roc_auc_score(y_true, y_score))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60, flush=True)
    print("Block 1 v2: Proper Evaluation (no benign leakage)", flush=True)
    print("=" * 60, flush=True)

    # Load Happer features
    print("\n[Loading] Happer entry features...", flush=True)
    t0 = time.time()
    apk_entries: Dict[str, List[Dict]] = {}
    with open(FEATURES_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            apk_id = row["apk_id"]
            if apk_id not in apk_entries:
                apk_entries[apk_id] = []
            apk_entries[apk_id].append(row)

    with open(APK_INDEX_FILE) as f:
        apk_index = json.load(f)
    print(f"  {len(apk_entries)} APKs in {time.time()-t0:.1f}s", flush=True)

    # --- Evaluation A: LOPO with seed-stratified benign split ---
    print("\n" + "=" * 60, flush=True)
    print("Eval A: LOPO + Seed-Stratified Benign Split (5 repeats)", flush=True)
    print("  (test benign seeds NEVER appear in training)", flush=True)
    print("=" * 60, flush=True)

    configs_a = [
        ("entropy (max)", None),
        ("byte_hist_256_xgb", slice(0, 256)),
        ("full_313_xgb", slice(None)),
    ]

    for name, feat_slice in configs_a:
        if feat_slice is None:
            # Entropy doesn't use LOPO-seed-split (it's unsupervised)
            continue
        print(f"\n  --- {name} ---", flush=True)
        results = run_lopo_seed_split(apk_entries, apk_index, feature_slice=feat_slice)
        aurocs = []
        for family, res in sorted(results.items()):
            print(f"    {family:10s}: AUROC = {res['mean']:.4f} +/- {res['std']:.4f}", flush=True)
            aurocs.append(res["mean"])
        if aurocs:
            print(f"    {'MEAN':10s}: AUROC = {np.mean(aurocs):.4f}", flush=True)

    # --- Evaluation B: Cross-dataset (Gold Standard) ---
    print("\n" + "=" * 60, flush=True)
    print("Eval B: Cross-Dataset (Happer train -> Track B test)", flush=True)
    print("  (completely different apps AND different packers)", flush=True)
    print("=" * 60, flush=True)

    # Entropy baseline on Track B
    print("\n  --- entropy (max entry entropy) ---", flush=True)
    ent_result = run_entropy_cross_dataset()
    if "error" not in ent_result:
        print(f"    Overall AUROC = {ent_result['overall_auroc']:.4f}", flush=True)
    else:
        print(f"    ERROR: {ent_result['error']}", flush=True)

    # XGBoost byte_hist_256 on Track B
    print("\n  --- byte_hist_256_xgb (Happer -> Track B) ---", flush=True)
    xgb_result = run_cross_dataset(apk_entries, apk_index, feature_slice=slice(0, 256))
    if "error" not in xgb_result:
        print(f"    Overall AUROC = {xgb_result['overall_auroc']:.4f} "
              f"(n={xgb_result['n_test']}: {xgb_result['n_packed']} packed, "
              f"{xgb_result['n_benign']} benign)", flush=True)
        print(f"    Per-family:", flush=True)
        for fam, auroc in sorted(xgb_result["per_family"].items()):
            print(f"      {fam:50s}: {auroc:.4f}", flush=True)
    else:
        print(f"    ERROR: {xgb_result['error']}", flush=True)

    # XGBoost full_313 on Track B
    print("\n  --- full_313_xgb (Happer -> Track B) ---", flush=True)
    full_result = run_cross_dataset(apk_entries, apk_index, feature_slice=slice(None))
    if "error" not in full_result:
        print(f"    Overall AUROC = {full_result['overall_auroc']:.4f}", flush=True)
        print(f"    Per-family:", flush=True)
        for fam, auroc in sorted(full_result["per_family"].items()):
            print(f"      {fam:50s}: {auroc:.4f}", flush=True)

    # --- Summary ---
    print("\n" + "=" * 60, flush=True)
    print("DECISION GATE v2 SUMMARY (honest evaluation)", flush=True)
    print("=" * 60, flush=True)

    if "error" not in xgb_result:
        gate = xgb_result["overall_auroc"]
        print(f"\n  Cross-dataset byte_hist_256_xgb AUROC = {gate:.4f}", flush=True)
        if gate > 0.90:
            print("  VERDICT: PIVOT", flush=True)
        elif gate >= 0.75:
            print("  VERDICT: PROCEED (features help, MIL can add localization value)", flush=True)
        elif gate >= 0.60:
            print("  VERDICT: PROCEED-CAUTIOUS (features have signal but not strong alone)", flush=True)
        else:
            print("  VERDICT: ENHANCE (features insufficient, need stronger representation)", flush=True)

    # Save
    out_file = DATA_DIR / "decision_gate_v2_results.json"
    save_data = {
        "entropy_cross": ent_result,
        "xgb_256_cross": xgb_result,
        "xgb_313_cross": full_result if "error" not in full_result else {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_file}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
