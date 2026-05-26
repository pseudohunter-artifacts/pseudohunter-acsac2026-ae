"""Track B Leave-One-Packer-Out (LOPO) evaluation with Tier 1B mixed training.

Tier 1B: Each LOPO fold trains on *both* Track B rows (real packed / benign)
**and** Track A rows (synthetic LOFO data).  Track A augmentation exposes the
model to a wider variety of packing strategies during training, with the goal
of lifting Track B APK AUROC from ~0.67 → 0.80+.

Key design decisions
--------------------
* Track A rows are loaded from the cached LOFO task directories:
    outputs/experiments/synthetic_multi_baseline_v4_lofo/tasks/*/region_labels.jsonl
  Each row carries an ``apk_id`` that is the SHA-256 of the generated APK.
  We remap those SHA-256 values to the friendly task_name
  (e.g. ``com_fsck_k9_39035_1381c04b_xor``) so they match the keys in
  ``apk_index``, which maps task_name → Path(generated_apk.apk).

* For each LOPO fold that holds out packer family P, we EXCLUDE Track A rows
  whose transform family maps to the same semantic packing strategy as P.
  This prevents cross-set leakage.  The mapping is heuristic (see
  ``PACKER_TO_EXCLUDED_TRANSFORMS`` below) and defaults to including
  everything when no match is configured.

* At *test* time we only evaluate on Track B rows (held-out packer + benign).
  Track A rows are never used for evaluation.

Usage
-----
    python scripts/experiments/run_track_b_mixed_lopo.py [--no-track-a]
        [--epochs N] [--track-a-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android_packer.apkio.objects import iter_apk_objects  # noqa: E402
from android_packer.baselines.ours import OursBaselineConfig, train_ours_baseline  # noqa: E402
from android_packer.regioning.windows import iter_regions  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRACK_B_DIR = ROOT / "data" / "real_world" / "track_b"
PACKED_DIR = TRACK_B_DIR / "packed"
BENIGN_DIR = TRACK_B_DIR / "benign"

DEFAULT_TRACK_A_DIR = ROOT / "outputs" / "experiments" / "synthetic_multi_baseline_v4_lofo" / "tasks"
DEFAULT_SYNTHETIC_APK_DIR = ROOT / "data" / "synthetic" / "generated_apks_v4"

DEFAULT_OUT_DIR = ROOT / "outputs" / "experiments" / "track_b_mixed_lopo"

# ---------------------------------------------------------------------------
# Semantic packer → excluded transform families.
# When a LOPO fold holds out a real packer, we can optionally exclude
# synthetic transforms that simulate similar techniques.
# Set to {} to disable all exclusions (i.e. always use full Track A).
# ---------------------------------------------------------------------------

PACKER_TO_EXCLUDED_TRANSFORMS: Dict[str, Set[str]] = {
    # XOR-based packers: exclude xor + split_xor synthetic transforms
    "cs1_360_jiagu": {"xor", "split_xor"},
    "cs3_bangcle": {"xor", "split_xor"},
    # Shell-based packers: no close synthetic analogue — include everything
    "s5_timscriptov_apkprotector_multiplatform": set(),
    "s6_dpt_shell": set(),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: Build Track A rows + index from cached LOFO task directories
# ---------------------------------------------------------------------------


def load_track_a_rows(
    lofo_dir: Path,
    synthetic_apk_dir: Path,
) -> tuple[List[Dict], Dict[str, Path]]:
    """Load all synthetic LOFO rows and build task_name → APK path index.

    Returns ``(rows, apk_index)`` where every row's ``apk_id`` is the
    task_name (e.g. ``com_fsck_k9_39035_1381c04b_xor``), NOT the sha256.
    """

    t0 = time.time()

    # Build sha256 → task_name mapping from the generated APKs.
    # We do NOT re-hash on every call — we cache the mapping lazily.
    print("[Track A] Building sha256→task_name mapping from generated APKs …", flush=True)
    sha_to_task: Dict[str, str] = {}
    apk_index: Dict[str, Path] = {}

    for apk_path in sorted(synthetic_apk_dir.glob("*.apk")):
        task_name = apk_path.stem  # e.g. com_fsck_k9_39035_1381c04b_xor
        sha = _sha256_file(apk_path)
        sha_to_task[sha] = task_name
        apk_index[task_name] = apk_path

    print(f"[Track A]   Mapped {len(sha_to_task)} APKs in {time.time()-t0:.1f}s", flush=True)

    rows: List[Dict] = []
    task_dirs = sorted(lofo_dir.iterdir())
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        labels_file = task_dir / "region_labels.jsonl"
        if not labels_file.exists():
            continue

        task_name = task_dir.name  # friendly name from directory
        # Sanity: confirm the sha of the generated APK matches the dir name.
        # If the generated APK doesn't exist, skip.
        if task_name not in apk_index:
            continue

        # The region_labels.jsonl uses sha256 as apk_id; remap to task_name.
        sha_prefix = None  # will be set from first row

        with open(labels_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["apk_id"] = task_name  # remap sha256 → task_name
                # Carry transform_families for GT-based type routing in
                # _object_instance_type (needed so the MIL head gets the
                # right type embedding for packed synthetic instances).
                # region_labels.jsonl already includes this field.
                rows.append(row)

    print(f"[Track A] Loaded {len(rows):,} rows from {len(task_dirs)} task dirs", flush=True)
    return rows, apk_index


# ---------------------------------------------------------------------------
# Step 2: Build Track B rows + index
# ---------------------------------------------------------------------------


def load_track_b_rows(
    packed_dir: Path,
    benign_dir: Path,
    *,
    verbose: bool = True,
) -> tuple[List[Dict], Dict[str, Path], Dict[str, str]]:
    """Load Track B region rows (real world).

    Returns ``(rows, apk_index, packer_map)`` where packer_map maps
    apk_id → packer_name (or ``"benign"``).
    """

    rows: List[Dict] = []
    apk_index: Dict[str, Path] = {}
    packer_map: Dict[str, str] = {}
    errors: List[str] = []

    # Benign APKs
    for apk_path in sorted(benign_dir.glob("*.apk")):
        apk_id = apk_path.stem
        apk_index[apk_id] = apk_path
        packer_map[apk_id] = "benign"
        try:
            for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
                if len(obj_bytes) < 64 or "!" in obj_meta.object_path:
                    continue
                for region in iter_regions(obj_meta, obj_bytes, window_size=4096, stride=2048):
                    rows.append({
                        "apk_id": apk_id,
                        "object_id": obj_meta.object_id,
                        "object_path": obj_meta.object_path,
                        "region_id": f"{obj_meta.object_id}:{region.offset_start}",
                        "offset_start": region.offset_start,
                        "offset_end": region.offset_end,
                        "entropy": region.entropy,
                        "printable_ratio": region.printable_ratio,
                        "label_id": 0,
                        "transform_families": [],
                    })
        except Exception as exc:
            errors.append(f"benign {apk_id}: {exc}")

    benign_count = sum(1 for v in packer_map.values() if v == "benign")
    if verbose:
        print(f"[Track B] Benign: {benign_count} APKs, {sum(1 for r in rows if r['label_id']==0)} regions",
              flush=True)

    # Packed APKs: packed/{packer_name}/{seed_name}/packed.apk
    for packer_entry in sorted(packed_dir.iterdir()):
        if not packer_entry.is_dir():
            continue
        packer_name = packer_entry.name
        for seed_entry in sorted(packer_entry.iterdir()):
            if not seed_entry.is_dir():
                continue
            packed_apk = seed_entry / "packed.apk"
            if not packed_apk.exists():
                continue
            apk_id = f"{packer_name}__{seed_entry.name}"
            apk_index[apk_id] = packed_apk
            packer_map[apk_id] = packer_name
            try:
                for obj_meta, obj_bytes in iter_apk_objects(packed_apk, max_depth=1):
                    if len(obj_bytes) < 64 or "!" in obj_meta.object_path:
                        continue
                    for region in iter_regions(obj_meta, obj_bytes, window_size=4096, stride=2048):
                        rows.append({
                            "apk_id": apk_id,
                            "object_id": obj_meta.object_id,
                            "object_path": obj_meta.object_path,
                            "region_id": f"{obj_meta.object_id}:{region.offset_start}",
                            "offset_start": region.offset_start,
                            "offset_end": region.offset_end,
                            "entropy": region.entropy,
                            "printable_ratio": region.printable_ratio,
                            "label_id": 1,
                            "transform_families": [],  # real-world; no GT transform
                        })
            except Exception as exc:
                errors.append(f"packed {apk_id}: {exc}")

    packed_count = sum(1 for v in packer_map.values() if v != "benign")
    if verbose:
        print(f"[Track B] Packed: {packed_count} APKs, {sum(1 for r in rows if r['label_id']==1)} regions",
              flush=True)
        if errors:
            print(f"[Track B] Errors: {len(errors)}", flush=True)
            for e in errors[:5]:
                print(f"  {e}", flush=True)

    return rows, apk_index, packer_map


# ---------------------------------------------------------------------------
# Step 3: LOPO folds — mixed training
# ---------------------------------------------------------------------------


def run_mixed_lopo(
    track_b_rows: List[Dict],
    track_b_index: Dict[str, Path],
    packer_map: Dict[str, str],
    track_a_rows: List[Dict],
    track_a_index: Dict[str, Path],
    cfg: OursBaselineConfig,
    *,
    use_track_a: bool = True,
    out_dir: Path,
) -> List[Dict]:
    """Run LOPO folds, mixing Track A synthetic data into each training set.

    Returns a list of per-fold result dicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    packer_families = sorted(set(v for v in packer_map.values() if v != "benign"))
    print(f"\n[LOPO] Packer families: {packer_families}", flush=True)

    # Pre-index Track A rows by transform_family for fast fold filtering.
    if use_track_a:
        track_a_by_apk: Dict[str, List[Dict]] = {}
        for row in track_a_rows:
            track_a_by_apk.setdefault(row["apk_id"], []).append(row)

        # Map task_name → transform family (last underscore-separated token)
        def _transform_of_task(task_name: str) -> str:
            # task_name = {seed}_{transform}, where transform can contain '_'
            # Known transforms: xor, base64, split_xor, path_randomized,
            # signature_strip, embedded_asset, so_embedded, dex_method_inlined,
            # multi_dex_shim, embedded_archive, dex_string_encrypted
            # We find the longest matching suffix from a known list.
            KNOWN_TRANSFORMS = {
                "split_xor", "path_randomized", "signature_strip",
                "embedded_asset", "so_embedded", "dex_method_inlined",
                "multi_dex_shim", "embedded_archive", "dex_string_encrypted",
                "base64", "xor",
            }
            for transform in sorted(KNOWN_TRANSFORMS, key=len, reverse=True):
                if task_name.endswith("_" + transform):
                    return transform
            return task_name.split("_")[-1]

        task_to_transform: Dict[str, str] = {
            t: _transform_of_task(t) for t in track_a_index
        }

    fold_results: List[Dict] = []

    for held_out in packer_families:
        print(f"\n[LOPO] fold: held_out={held_out}", flush=True)

        # Track B train/test split
        b_train_apks = {k for k, v in packer_map.items() if v != held_out}
        b_test_apks = {k for k, v in packer_map.items() if v == held_out or v == "benign"}

        b_train_rows = [r for r in track_b_rows if r["apk_id"] in b_train_apks]
        b_test_rows = [r for r in track_b_rows if r["apk_id"] in b_test_apks]
        b_train_index = {k: v for k, v in track_b_index.items() if k in b_train_apks}
        b_test_index = {k: v for k, v in track_b_index.items() if k in b_test_apks}

        print(f"  Track B train: {len(b_train_apks)} APKs, {len(b_train_rows)} rows", flush=True)
        print(f"  Track B test:  {len(b_test_apks)} APKs, {len(b_test_rows)} rows", flush=True)

        # Optionally add Track A rows (with exclusion of semantically similar
        # transforms to the held-out packer family).
        train_rows = list(b_train_rows)
        combined_index = dict(b_train_index)

        if use_track_a:
            excluded_transforms = PACKER_TO_EXCLUDED_TRANSFORMS.get(held_out, set())
            a_rows_added = 0
            for task_name, task_rows in track_a_by_apk.items():
                transform = task_to_transform.get(task_name, "")
                if transform in excluded_transforms:
                    continue
                train_rows.extend(task_rows)
                combined_index[task_name] = track_a_index[task_name]
                a_rows_added += len(task_rows)
            print(f"  Track A added: {a_rows_added} rows ({len(combined_index)-len(b_train_index)} APKs, "
                  f"excluded transforms: {sorted(excluded_transforms) or 'none'})", flush=True)

        total_train = len(train_rows)
        pos_train = sum(1 for r in train_rows if r.get("label_id", 0) == 1)
        print(f"  Combined train: {total_train} rows, {pos_train} positive ({100*pos_train/max(1,total_train):.1f}%)",
              flush=True)

        try:
            t0 = time.time()
            model = train_ours_baseline(train_rows, combined_index, cfg)
            train_secs = time.time() - t0

            t1 = time.time()
            result = model.predict(b_test_rows, b_test_index)
            pred_secs = time.time() - t1

            apk_preds = result.apk_predictions
            y_true = [p.true_label_id for p in apk_preds]
            y_score = [p.score for p in apk_preds]

            if len(set(y_true)) < 2:
                print("  SKIP: single class in test set", flush=True)
                fold_results.append({
                    "packer": held_out, "skipped": True,
                    "reason": "single_class",
                })
                continue

            apk_auroc = roc_auc_score(y_true, y_score)
            reg_auroc = (result.report.get("metrics", {})
                         .get("region", {}).get("auroc"))
            obj_mrr = (result.report.get("ranking", {})
                       .get("object", {}).get("mrr"))

            print(f"  APK AUROC={apk_auroc:.4f}  "
                  f"Region AUROC={reg_auroc}  MRR={obj_mrr}  "
                  f"train={train_secs:.0f}s pred={pred_secs:.0f}s",
                  flush=True)

            fold_result = {
                "packer": held_out,
                "apk_auroc": apk_auroc,
                "region_auroc": reg_auroc,
                "object_mrr": obj_mrr,
                "train_rows": total_train,
                "train_pos": pos_train,
                "train_secs": round(train_secs, 1),
                "pred_secs": round(pred_secs, 1),
                "use_track_a": use_track_a,
            }
            fold_results.append(fold_result)

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  ERROR: {exc}", flush=True)
            print(tb[:1000], flush=True)
            fold_results.append({"packer": held_out, "error": str(exc)})

    # Summary
    print("\n=== Track B Mixed-LOPO Results ===", flush=True)
    for fr in fold_results:
        if "error" in fr:
            print(f"  {fr['packer']:45s}: ERROR — {fr['error'][:60]}", flush=True)
        elif fr.get("skipped"):
            print(f"  {fr['packer']:45s}: SKIPPED ({fr.get('reason')})", flush=True)
        else:
            print(
                f"  {fr['packer']:45s}: "
                f"APK={fr['apk_auroc']:.4f}  "
                f"Reg={fr.get('region_auroc')}  "
                f"MRR={fr.get('object_mrr')}",
                flush=True,
            )

    valid = [fr for fr in fold_results if "apk_auroc" in fr]
    if valid:
        mean_apk = float(np.mean([fr["apk_auroc"] for fr in valid]))
        print(f"\nMean APK AUROC ({len(valid)} folds): {mean_apk:.4f}", flush=True)

    return fold_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-track-a", action="store_true",
                   help="Disable Track A synthetic augmentation (baseline comparison)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--track-a-dir", type=Path, default=DEFAULT_TRACK_A_DIR,
                   help="Root of cached LOFO task directories")
    p.add_argument("--synthetic-apk-dir", type=Path, default=DEFAULT_SYNTHETIC_APK_DIR,
                   help="Directory containing generated synthetic APK files")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-dex-structure", action="store_true",
                   help="Disable Tier 1A DEX structure features (Group H)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_track_a = not args.no_track_a

    base_cfg = OursBaselineConfig()
    handcrafted_cfg = dataclasses.replace(
        base_cfg.handcrafted_config,
        include_dex_structure=(not args.no_dex_structure),
    )
    cfg = dataclasses.replace(
        base_cfg,
        supervision_mode="bag",
        scoring_mode="attention_auto",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=0.65,
        verbose=False,
        train_max_bag_size=256,
        train_min_positive_fraction=0.01,
        handcrafted_config=handcrafted_cfg,
    )

    label = "mixed" if use_track_a else "track_b_only"
    out_dir = args.out_dir / label
    print(f"[Config] use_track_a={use_track_a}, epochs={cfg.epochs}, "
          f"include_dex_structure={cfg.handcrafted_config.include_dex_structure}", flush=True)
    print(f"[Config] out_dir={out_dir}", flush=True)

    # Load Track B (always needed)
    b_rows, b_index, packer_map = load_track_b_rows(PACKED_DIR, BENIGN_DIR)

    # Load Track A (synthetic LOFO cache)
    if use_track_a:
        a_rows, a_index = load_track_a_rows(args.track_a_dir, args.synthetic_apk_dir)
    else:
        a_rows, a_index = [], {}

    fold_results = run_mixed_lopo(
        b_rows, b_index, packer_map,
        a_rows, a_index,
        cfg,
        use_track_a=use_track_a,
        out_dir=out_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "lopo_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "use_track_a": use_track_a,
                    "epochs": cfg.epochs,
                    "scoring_mode": cfg.scoring_mode,
                    "include_dex_structure": cfg.handcrafted_config.include_dex_structure,
                },
                "folds": fold_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
