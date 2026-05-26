"""Block 1: XGBoost Decision Gate — Byte Histogram APK AUROC.

Tests whether 256-dim byte histogram features + simple classifier already
achieves high APK AUROC on Happer LOPO. This determines paper direction:
  - [0.75, 0.90] -> proceed with DS-AMIL (MIL has room to add value)
  - > 0.90 -> pivot (features alone solve detection; MIL only for localization)
  - < 0.75 -> enhance (add DEX structure features)

Usage:
    python scripts/experiments/happer_xgb_decision_gate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("ERROR: scikit-learn required. Install: pip install scikit-learn", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "outputs" / "experiments" / "happer_ds_amil"
FEATURES_FILE = DATA_DIR / "entry_features.jsonl"
SPLITS_FILE = DATA_DIR / "splits" / "lopo_splits.json"
APK_INDEX_FILE = DATA_DIR / "apk_index.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_entry_data() -> Dict[str, List[Dict]]:
    """Load entry features grouped by apk_id."""
    print("[Loading] Reading entry features...", flush=True)
    t0 = time.time()
    apk_entries: Dict[str, List[Dict]] = {}

    with open(FEATURES_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            apk_id = row["apk_id"]
            if apk_id not in apk_entries:
                apk_entries[apk_id] = []
            apk_entries[apk_id].append(row)

    print(f"  Loaded {sum(len(v) for v in apk_entries.values())} entries "
          f"from {len(apk_entries)} APKs in {time.time()-t0:.1f}s", flush=True)
    return apk_entries


def apk_feature_aggregation(
    entries: List[Dict],
    feature_slice: slice = slice(None),
    method: str = "max_mean"
) -> np.ndarray:
    """Aggregate entry features to APK-level.

    Methods:
      - "max": max over entries per dimension
      - "mean": mean over entries
      - "max_mean": concat(max, mean) -> 2D
      - "top3_mean": mean of top-3 entries by entropy
    """
    if not entries:
        return np.zeros(313, dtype=np.float32)

    feats = np.array([e["features"][feature_slice] for e in entries], dtype=np.float32)

    if method == "max":
        return feats.max(axis=0)
    elif method == "mean":
        return feats.mean(axis=0)
    elif method == "max_mean":
        return np.concatenate([feats.max(axis=0), feats.mean(axis=0)])
    elif method == "top3_mean":
        # Top-3 by entropy (index 256 in feature vector)
        entropies = feats[:, 256] if feats.shape[1] > 256 else np.array([e["entropy"] for e in entries])
        top_idx = np.argsort(entropies)[-3:]
        return feats[top_idx].mean(axis=0)
    else:
        return feats.max(axis=0)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_lopo_fold(
    train_ids: List[str],
    test_ids: List[str],
    apk_entries: Dict[str, List[Dict]],
    apk_index: Dict,
    classifier_type: str = "xgb",
    feature_range: str = "full",
) -> Dict:
    """Run one LOPO fold. Returns dict with metrics."""

    # Feature slicing
    if feature_range == "hist_256":
        feat_slice = slice(0, 256)
    elif feature_range == "hist_261":
        feat_slice = slice(0, 261)  # histogram + entropy + rolling stats
    elif feature_range == "15dim":
        # Simulate 15-dim: entropy(1) + rolling(4) + printable(1) + zero(1) + comp(1) + size(1) + is_dex(1) + is_elf(1) + dex_feats(4) = ~15
        feat_slice = slice(256, 280)  # rough approximation
    else:
        feat_slice = slice(None)

    # Build train/test sets
    X_train, y_train = [], []
    X_test, y_test = [], []

    for apk_id in train_ids:
        if apk_id not in apk_entries:
            continue
        entries = apk_entries[apk_id]
        feat = apk_feature_aggregation(entries, feat_slice, method="max")
        label = 1 if apk_index[apk_id]["is_packed"] else 0
        X_train.append(feat)
        y_train.append(label)

    for apk_id in test_ids:
        if apk_id not in apk_entries:
            continue
        entries = apk_entries[apk_id]
        feat = apk_feature_aggregation(entries, feat_slice, method="max")
        label = 1 if apk_index[apk_id]["is_packed"] else 0
        X_test.append(feat)
        y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    if len(set(y_test)) < 2:
        return {"auroc": None, "reason": "single_class_in_test"}

    # Train classifier
    if classifier_type == "xgb":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
    elif classifier_type == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    else:
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_score)

    return {
        "auroc": auroc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((1 - y_test).sum()),
    }


def run_entropy_baseline(
    test_ids: List[str],
    apk_entries: Dict[str, List[Dict]],
    apk_index: Dict,
) -> Dict:
    """Simple entropy threshold baseline: max entry entropy as APK score."""
    y_true, y_score = [], []
    for apk_id in test_ids:
        if apk_id not in apk_entries:
            continue
        entries = apk_entries[apk_id]
        max_entropy = max(e["entropy"] for e in entries) if entries else 0.0
        label = 1 if apk_index[apk_id]["is_packed"] else 0
        y_true.append(label)
        y_score.append(max_entropy)

    if len(set(y_true)) < 2:
        return {"auroc": None}
    return {"auroc": roc_auc_score(y_true, y_score)}


def main() -> None:
    print("=" * 60, flush=True)
    print("Block 1: XGBoost Decision Gate", flush=True)
    print("=" * 60, flush=True)

    # Load data
    apk_entries = load_entry_data()

    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    with open(APK_INDEX_FILE) as f:
        apk_index = json.load(f)

    families = sorted(splits.keys())

    # Run experiments for each configuration
    configs = [
        ("entropy_only", "entropy", "na"),
        ("byte_hist_256_lr", "lr", "hist_256"),
        ("byte_hist_256_xgb", "xgb", "hist_256"),
        ("byte_hist_261_xgb", "xgb", "hist_261"),
        ("full_313_xgb", "xgb", "full"),
    ]

    results = {}

    for config_name, clf_type, feat_range in configs:
        print(f"\n--- {config_name} ---", flush=True)
        fold_aurocs = []

        for family in families:
            train_ids = splits[family]["train"]
            test_ids = splits[family]["test"]

            if clf_type == "entropy":
                fold_result = run_entropy_baseline(test_ids, apk_entries, apk_index)
            else:
                fold_result = run_lopo_fold(
                    train_ids, test_ids, apk_entries, apk_index,
                    classifier_type=clf_type, feature_range=feat_range
                )

            auroc = fold_result.get("auroc")
            if auroc is not None:
                fold_aurocs.append(auroc)
                print(f"  {family:10s}: AUROC = {auroc:.4f}", flush=True)
            else:
                print(f"  {family:10s}: SKIPPED", flush=True)

        if fold_aurocs:
            mean_auroc = np.mean(fold_aurocs)
            std_auroc = np.std(fold_aurocs)
            print(f"  {'MEAN':10s}: AUROC = {mean_auroc:.4f} +/- {std_auroc:.4f}", flush=True)
            results[config_name] = {
                "mean_auroc": float(mean_auroc),
                "std_auroc": float(std_auroc),
                "per_fold": {f: a for f, a in zip(families, fold_aurocs)},
            }

    # Summary and Decision Gate
    print("\n" + "=" * 60, flush=True)
    print("DECISION GATE SUMMARY", flush=True)
    print("=" * 60, flush=True)

    for name, res in results.items():
        print(f"  {name:25s}: {res['mean_auroc']:.4f} +/- {res['std_auroc']:.4f}", flush=True)

    # Decision
    gate_metric = results.get("byte_hist_256_xgb", {}).get("mean_auroc", 0)
    print(f"\n  Decision Gate metric (byte_hist_256_xgb): {gate_metric:.4f}", flush=True)

    if gate_metric > 0.90:
        print("  VERDICT: PIVOT - XGBoost too strong, MIL paper needs repositioning", flush=True)
        print("  -> Emphasize localization (Entry MRR), not detection (APK AUROC)", flush=True)
    elif gate_metric >= 0.75:
        print("  VERDICT: PROCEED - MIL has room to add value", flush=True)
        print("  -> Continue with DS-AMIL (Block 3)", flush=True)
    else:
        print("  VERDICT: ENHANCE - Features not strong enough alone", flush=True)
        print("  -> Add DEX structure features before proceeding", flush=True)

    # Save results
    out_file = DATA_DIR / "decision_gate_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_file}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
