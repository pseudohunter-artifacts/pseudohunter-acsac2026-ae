"""Prepare Track C wild samples for TI-MIL APK-level detection.

Extracts objects + regions from each APK, assigns type labels via heuristic,
and outputs region_labels.jsonl + apk_index.json for the Ours MIL baseline runner.

Usage:
    python scripts/data/prepare_track_c_for_mil.py --execute
    python scripts/data/prepare_track_c_for_mil.py --dry-run  (default)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default="outputs/experiments/track_c/labels.jsonl",
        help="Path to Track C labels JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/real_world/track_c_mil_input",
        help="Output directory for MIL-ready data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--window-size", type=int, default=4096, help="Region window size"
    )
    parser.add_argument("--stride", type=int, default=2048, help="Region stride")
    parser.add_argument(
        "--execute", action="store_true", help="Actually run (default: dry-run)"
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)

    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}")
        return 1

    # Load labels
    with open(labels_path) as f:
        all_labels = [json.loads(line) for line in f]

    # Filter to samples with valid paths and probed status
    valid = []
    for lbl in all_labels:
        sample_path = Path(lbl["sample_path"])
        is_packed = lbl.get("labels", {}).get("is_packed_probed")
        if is_packed is None:
            continue  # not probed yet
        if not sample_path.exists():
            continue
        valid.append(
            {
                "apk_id": lbl["sample_id"],
                "apk_path": str(sample_path),
                "label_id": 1 if is_packed else 0,
                "family": lbl.get("family", "unknown"),
                "packer": lbl.get("labels", {}).get("suspected_packer"),
                "source": lbl.get("source", "unknown"),
            }
        )

    if args.max_samples:
        valid = valid[: args.max_samples]

    packed_count = sum(1 for v in valid if v["label_id"] == 1)
    benign_count = sum(1 for v in valid if v["label_id"] == 0)
    print(f"Track C samples: {len(valid)} total ({packed_count} packed, {benign_count} benign)")

    if not args.execute:
        print("[DRY RUN] Would extract objects + regions from each APK.")
        print(f"[DRY RUN] Output: {output_dir}/region_labels.jsonl + apk_index.json")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import extraction modules
    from android_packer.apkio.objects import iter_apk_objects
    from android_packer.regioning.windows import iter_regions

    apk_index = {}
    region_rows = []
    errors = []

    for i, sample in enumerate(valid):
        apk_path = Path(sample["apk_path"])
        apk_id = sample["apk_id"]

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(valid)}: {apk_id[:12]}...")

        apk_index[apk_id] = str(apk_path)

        try:
            # Extract objects: iter_apk_objects yields (ApkObject, bytes)
            obj_pairs = list(
                iter_apk_objects(apk_path, max_depth=1)
            )
            if not obj_pairs:
                errors.append(f"{apk_id}: no objects extracted")
                continue

            # Generate regions for each object
            for obj_meta, obj_bytes in obj_pairs:
                obj_path = obj_meta.object_path
                obj_id = obj_meta.object_id

                if len(obj_bytes) < 64:
                    continue  # skip tiny objects

                regions = list(
                    iter_regions(
                        obj_meta,
                        obj_bytes,
                        window_size=args.window_size,
                        stride=args.stride,
                    )
                )

                for region in regions:
                    region_rows.append(
                        {
                            "apk_id": apk_id,
                            "object_id": obj_id,
                            "object_path": obj_path,
                            "region_id": f"{obj_id}:{region.offset_start}",
                            "offset_start": region.offset_start,
                            "offset_end": region.offset_end,
                            "entropy": region.entropy,
                            "printable_ratio": region.printable_ratio,
                            "label_id": sample["label_id"],
                            "transform_family": "wild",
                        }
                    )
        except Exception as e:
            errors.append(f"{apk_id}: {e}")
            continue

    # Write outputs
    region_labels_path = output_dir / "region_labels.jsonl"
    apk_index_path = output_dir / "apk_index.json"

    with open(region_labels_path, "w") as f:
        for row in region_rows:
            f.write(json.dumps(row) + "\n")

    with open(apk_index_path, "w") as f:
        json.dump(apk_index, f, indent=2)

    print(f"\nDone!")
    print(f"  Regions: {len(region_rows)}")
    print(f"  APKs indexed: {len(apk_index)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output: {region_labels_path}")
    if errors:
        print(f"  First 5 errors:")
        for e in errors[:5]:
            print(f"    {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
