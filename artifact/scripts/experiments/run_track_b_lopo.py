"""Track B Leave-One-Packer-Out evaluation for TI-MIL."""
import json, sys, os, dataclasses
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from pathlib import Path
from android_packer.apkio.objects import iter_apk_objects
from android_packer.regioning.windows import iter_regions
from android_packer.baselines.ours import OursBaselineConfig, train_ours_baseline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
packed_dir = ROOT / "data" / "real_world" / "track_b" / "packed"
benign_dir = ROOT / "data" / "real_world" / "track_b" / "benign"

rows = []
apk_index = {}
packer_map = {}
errors = []

# Benign APKs
for apk_path in sorted(benign_dir.glob("*.apk")):
    apk_id = apk_path.stem
    apk_index[apk_id] = str(apk_path)
    packer_map[apk_id] = "benign"
    try:
        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) < 64 or "!" in obj_meta.object_path:
                continue
            for region in iter_regions(obj_meta, obj_bytes, window_size=4096, stride=2048):
                rows.append({
                    "apk_id": apk_id, "object_id": obj_meta.object_id,
                    "object_path": obj_meta.object_path,
                    "region_id": f"{obj_meta.object_id}:{region.offset_start}",
                    "offset_start": region.offset_start, "offset_end": region.offset_end,
                    "entropy": region.entropy, "printable_ratio": region.printable_ratio,
                    "label_id": 0, "transform_family": "benign",
                })
    except Exception as e:
        errors.append(f"benign {apk_id}: {e}")

print(f"Benign: {len([r for r in rows])} regions, {sum(1 for v in packer_map.values() if v=='benign')} APKs")

# Packed APKs
for packer_entry in sorted(packed_dir.iterdir()):
    if not packer_entry.is_dir():
        continue
    packer_name = packer_entry.name
    for benign_entry in sorted(packer_entry.iterdir()):
        if not benign_entry.is_dir():
            continue
        packed_apk = benign_entry / "packed.apk"
        if not packed_apk.exists():
            continue
        apk_id = f"{packer_name}__{benign_entry.name}"
        apk_index[apk_id] = str(packed_apk)
        packer_map[apk_id] = packer_name
        try:
            for obj_meta, obj_bytes in iter_apk_objects(packed_apk, max_depth=1):
                if len(obj_bytes) < 64 or "!" in obj_meta.object_path:
                    continue
                for region in iter_regions(obj_meta, obj_bytes, window_size=4096, stride=2048):
                    rows.append({
                        "apk_id": apk_id, "object_id": obj_meta.object_id,
                        "object_path": obj_meta.object_path,
                        "region_id": f"{obj_meta.object_id}:{region.offset_start}",
                        "offset_start": region.offset_start, "offset_end": region.offset_end,
                        "entropy": region.entropy, "printable_ratio": region.printable_ratio,
                        "label_id": 1, "transform_family": packer_name,
                    })
        except Exception as e:
            errors.append(f"packed {apk_id}: {e}")

packed_count = sum(1 for v in packer_map.values() if v != "benign")
print(f"Packed: {packed_count} APKs")
print(f"Total rows: {len(rows)}, Errors: {len(errors)}")

# LOPO
packer_families = sorted(set(v for v in packer_map.values() if v != "benign"))
print(f"\nPacker families: {packer_families}")

cfg = dataclasses.replace(
    OursBaselineConfig(),
    supervision_mode="bag", scoring_mode="attention_auto",
    epochs=10, threshold=0.65, verbose=False,
    train_max_bag_size=256, train_min_positive_fraction=0.01,
)

fold_results = []
for held_out in packer_families:
    train_apks = set(k for k, v in packer_map.items() if v != held_out)
    test_apks = set(k for k, v in packer_map.items() if v == held_out or v == "benign")

    train_rows = [r for r in rows if r["apk_id"] in train_apks]
    test_rows = [r for r in rows if r["apk_id"] in test_apks]
    train_index = {k: Path(v) for k, v in apk_index.items() if k in train_apks}
    test_index = {k: Path(v) for k, v in apk_index.items() if k in test_apks}

    train_packed = sum(1 for k in train_apks if packer_map[k] != "benign")
    test_packed = sum(1 for k in test_apks if packer_map[k] != "benign")
    print(f"\n  LOPO: held_out={held_out}, train_packed={train_packed}, test_packed={test_packed}")

    try:
        model = train_ours_baseline(train_rows, train_index, cfg)
        result = model.predict(test_rows, test_index)
        apk_preds = result.apk_predictions
        y_true = [p.true_label_id for p in apk_preds]
        y_score = [p.score for p in apk_preds]
        if len(set(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_score)
            reg_auroc = result.report.get("metrics", {}).get("region", {}).get("auroc")
            obj_mrr = result.report.get("ranking", {}).get("object", {}).get("mrr")
            print(f"    APK AUROC={auroc:.4f}, Region AUROC={reg_auroc}, MRR={obj_mrr}")
            fold_results.append({"packer": held_out, "apk_auroc": auroc,
                                 "region_auroc": reg_auroc, "object_mrr": obj_mrr})
        else:
            print("    SKIP: single class")
    except Exception as e:
        print(f"    ERROR: {e}")
        fold_results.append({"packer": held_out, "error": str(e)})

print("\n=== Track B LOPO Results ===")
for fr in fold_results:
    if "error" in fr:
        print(f"  {fr['packer']:40s}: ERROR")
    else:
        print(f"  {fr['packer']:40s}: APK={fr['apk_auroc']:.4f} Reg={fr.get('region_auroc')} MRR={fr.get('object_mrr')}")

valid = [fr for fr in fold_results if "apk_auroc" in fr]
if valid:
    print(f"\nMean APK AUROC: {np.mean([fr['apk_auroc'] for fr in valid]):.4f}")

out_dir = ROOT / "outputs" / "experiments" / "track_b_lopo"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "lopo_results.json", "w") as f:
    json.dump(fold_results, f, indent=2)
print(f"Saved to {out_dir / 'lopo_results.json'}")
