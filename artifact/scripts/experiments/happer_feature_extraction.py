"""Happer Dataset Feature Extraction + Diff Label Generation (Block 0).

Extracts 310-dim byte-distribution features from Happer paired APKs,
generates differential labels for attention alignment, and builds LOPO splits.

Usage:
    python scripts/experiments/happer_feature_extraction.py [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HAPPER_DIR = ROOT / "data" / "happer_dataset" / "FSet"
DEFAULT_OUT_DIR = ROOT / "outputs" / "experiments" / "happer_ds_amil"

# Packer families and their directory names
PACKER_FAMILIES = {
    "Ali": ["Ali-16"],
    "Baidu": ["Baidu-16", "Baidu-18"],
    "Bangcle": ["Bangcle-18"],
    "Ijiami": ["Ijiami-16", "Ijiami-18"],
    "Kiwi": ["Kiwi-18"],
    "Qihoo": ["Qihoo-16", "Qihoo-18"],
    "Tencent": ["Tencent-16", "Tencent-18"],
}

ORIGIN_DIRS = ["Origin-16", "Oirgin-18"]


# ---------------------------------------------------------------------------
# Feature extraction (per ZIP entry)
# ---------------------------------------------------------------------------


def compute_byte_histogram(data: bytes) -> np.ndarray:
    """256-dim byte frequency histogram (normalized)."""
    if len(data) == 0:
        return np.zeros(256, dtype=np.float32)
    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1
    return (counts / len(data)).astype(np.float32)


def compute_entropy(data: bytes) -> float:
    """Shannon entropy in bits."""
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


def compute_rolling_entropy_stats(data: bytes, window: int = 1024) -> np.ndarray:
    """Rolling entropy statistics: [mean, std, max, min]."""
    if len(data) < window:
        e = compute_entropy(data)
        return np.array([e, 0.0, e, e], dtype=np.float32)

    entropies = []
    for i in range(0, len(data) - window + 1, window // 2):
        chunk = data[i:i + window]
        entropies.append(compute_entropy(chunk))

    if not entropies:
        e = compute_entropy(data)
        return np.array([e, 0.0, e, e], dtype=np.float32)

    arr = np.array(entropies)
    return np.array([arr.mean(), arr.std(), arr.max(), arr.min()], dtype=np.float32)


def detect_magic(data: bytes) -> str:
    """Detect file format from magic bytes."""
    if len(data) < 4:
        return "unknown"
    if data[:4] == b"dex\n":
        return "dex"
    if data[:4] == b"\x7fELF":
        return "elf"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:3] == b"\x03\x00\x08" or (len(data) > 4 and data[:4] == b"\x03\x00\x08\x00"):
        return "axml"
    if len(data) > 4 and data[:4] == b"\x02\x00\x0c\x00":
        return "arsc"
    return "unknown"


def magic_to_onehot(magic: str) -> np.ndarray:
    """10-dim one-hot for magic type."""
    types = ["dex", "elf", "zip", "png", "jpg", "gif", "gzip", "axml", "arsc", "unknown"]
    vec = np.zeros(10, dtype=np.float32)
    idx = types.index(magic) if magic in types else 9
    vec[idx] = 1.0
    return vec


def path_type_onehot(entry_path: str) -> np.ndarray:
    """12-dim one-hot for path-based coarse type (leakage-free)."""
    types = [
        "classes_dex", "secondary_dex", "native_lib", "assets",
        "res_raw", "res_other", "meta_inf", "resources_arsc",
        "manifest", "embedded_archive", "root_file", "unknown"
    ]
    vec = np.zeros(12, dtype=np.float32)
    p = entry_path.lower().replace("\\", "/")

    if p == "classes.dex":
        idx = 0
    elif p.startswith("classes") and p.endswith(".dex"):
        idx = 1
    elif p.startswith("lib/") or p.endswith(".so"):
        idx = 2
    elif p.startswith("assets/"):
        idx = 3
    elif p.startswith("res/raw/"):
        idx = 4
    elif p.startswith("res/"):
        idx = 5
    elif p.startswith("meta-inf/"):
        idx = 6
    elif p == "resources.arsc":
        idx = 7
    elif p == "androidmanifest.xml":
        idx = 8
    elif p.endswith((".apk", ".jar", ".zip")):
        idx = 9
    elif "/" not in p:
        idx = 10
    else:
        idx = 11

    vec[idx] = 1.0
    return vec


def extract_entry_features(entry_path: str, data: bytes, apk_context: Dict) -> np.ndarray:
    """Extract ~310-dim feature vector for a single ZIP entry.

    Groups:
      A: byte_histogram (256) + entropy (1) + rolling_entropy_stats (4) = 261
      B: printable_ratio (1) + zero_ratio (1) + compression_ratio (1) + size_log (1)
         + magic_onehot (10) + path_type_onehot (12) + is_dex (1) + is_elf (1) = 28
      C: DEX-specific (7) + ELF-specific (7) + Asset-specific (4) + padding (2) = 20
      D: apk context (4)
    Total: 261 + 28 + 20 + 4 = 313
    """
    # Group A: byte distribution
    hist = compute_byte_histogram(data)  # 256
    entropy = compute_entropy(data)  # 1
    roll_stats = compute_rolling_entropy_stats(data)  # 4

    # Group B: structural context
    n = len(data)
    printable = sum(1 for b in data if 32 <= b <= 126) / max(n, 1)
    zero_ratio = data.count(0) / max(n, 1)
    comp_ratio = apk_context.get("compress_ratio", 1.0)
    size_log = math.log2(max(n, 1))

    magic = detect_magic(data)
    magic_oh = magic_to_onehot(magic)  # 10
    path_oh = path_type_onehot(entry_path)  # 12
    is_dex = 1.0 if magic == "dex" else 0.0
    is_elf = 1.0 if magic == "elf" else 0.0

    # Group C: format-specific (simplified for speed)
    dex_feats = np.zeros(7, dtype=np.float32)
    elf_feats = np.zeros(7, dtype=np.float32)
    asset_feats = np.zeros(4, dtype=np.float32)
    padding = np.zeros(2, dtype=np.float32)

    if magic == "dex" and len(data) > 112:
        try:
            # DEX header parsing (simplified)
            string_ids_size = int.from_bytes(data[56:60], "little")
            type_ids_size = int.from_bytes(data[64:68], "little")
            method_ids_size = int.from_bytes(data[88:92], "little")
            class_defs_size = int.from_bytes(data[96:100], "little")
            dex_feats[0] = math.log2(max(string_ids_size, 1))  # string_count_log
            dex_feats[1] = math.log2(max(type_ids_size, 1))  # type_count_log
            dex_feats[2] = math.log2(max(method_ids_size, 1))  # method_count_log
            dex_feats[3] = math.log2(max(class_defs_size, 1))  # class_count_log
            dex_feats[4] = 1.0 if data[:4] == b"dex\n" else 0.0  # valid_header
            # native_ratio approximation: 0 for most DEX
            dex_feats[5] = 0.0
            # empty_method_ratio: hard to compute without full parse
            dex_feats[6] = 0.0
        except Exception:
            pass

    elif magic == "elf" and len(data) > 64:
        try:
            # ELF: check for packer-related indicators in string table
            elf_feats[0] = 1.0  # valid elf header
            elf_feats[1] = 1.0 if b"JNI_OnLoad" in data else 0.0
            elf_feats[2] = 1.0 if b"dlopen" in data else 0.0
            elf_feats[3] = 1.0 if b"mmap" in data else 0.0
            elf_feats[4] = 1.0 if b"mprotect" in data else 0.0
            # text_entropy (first 4KB after header)
            text_region = data[64:min(4160, len(data))]
            elf_feats[5] = compute_entropy(text_region)
            # custom sections count (heuristic: unusual strings)
            elf_feats[6] = min(data.count(b".packed") + data.count(b".jiagu"), 5.0)
        except Exception:
            pass

    else:
        # Asset/unknown
        asset_feats[0] = 1.0 if b"dex\n" in data[:4096] else 0.0  # embedded dex magic
        asset_feats[1] = 1.0 if b"\x7fELF" in data[:4096] else 0.0  # embedded elf magic
        asset_feats[2] = 1.0 if b"PK\x03\x04" in data[:4096] else 0.0  # embedded zip magic
        asset_feats[3] = printable  # text_like_ratio (same as printable)

    # Group D: APK global context
    ctx = np.array([
        apk_context.get("dex_count", 0),
        apk_context.get("so_count", 0),
        apk_context.get("asset_count", 0),
        apk_context.get("total_entries", 0),
    ], dtype=np.float32)

    # Concatenate all
    feature = np.concatenate([
        hist,                    # 256
        np.array([entropy], dtype=np.float32),  # 1
        roll_stats,              # 4
        np.array([printable, zero_ratio, comp_ratio, size_log], dtype=np.float32),  # 4
        magic_oh,                # 10
        path_oh,                 # 12
        np.array([is_dex, is_elf], dtype=np.float32),  # 2
        dex_feats,               # 7
        elf_feats,               # 7
        asset_feats,             # 4
        padding,                 # 2
        ctx,                     # 4
    ])
    # Total: 256+1+4+4+10+12+2+7+7+4+2+4 = 313
    return feature


# ---------------------------------------------------------------------------
# APK parsing
# ---------------------------------------------------------------------------


def parse_apk_entries(apk_path: Path) -> Optional[List[Dict]]:
    """Parse an APK and extract per-entry features.

    Returns list of dicts: {name, features (313-dim), size, entropy, ...}
    """
    try:
        with zipfile.ZipFile(apk_path) as zf:
            infos = zf.infolist()

            # Build APK context
            dex_count = sum(1 for i in infos if i.filename.lower().endswith(".dex"))
            so_count = sum(1 for i in infos if i.filename.lower().endswith(".so"))
            asset_count = sum(1 for i in infos if i.filename.lower().startswith("assets/"))
            apk_context = {
                "dex_count": dex_count,
                "so_count": so_count,
                "asset_count": asset_count,
                "total_entries": len(infos),
            }

            entries = []
            for info in infos:
                if info.is_dir():
                    continue
                if info.file_size == 0:
                    continue
                # Skip very large entries (> 50MB) to avoid OOM
                if info.file_size > 50 * 1024 * 1024:
                    continue

                try:
                    data = zf.read(info.filename)
                except Exception:
                    continue

                compress_ratio = (info.compress_size / max(info.file_size, 1)
                                  if info.file_size > 0 else 1.0)
                apk_context["compress_ratio"] = compress_ratio

                features = extract_entry_features(info.filename, data, apk_context)
                entropy = compute_entropy(data)

                entries.append({
                    "name": info.filename,
                    "features": features,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "entropy": entropy,
                })

            return entries

    except (zipfile.BadZipFile, Exception) as e:
        print(f"  ERROR parsing {apk_path.name}: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Diff label generation
# ---------------------------------------------------------------------------


def compute_diff_labels(
    origin_entries: List[Dict],
    packed_entries: List[Dict],
) -> Dict[str, float]:
    """Compute per-entry diff labels between origin and packed APK.

    Returns: {entry_name: diff_score} where:
      - 1.0 = new entry (packer-injected)
      - 0.8 = significantly modified (entropy change > 2 or size change > 50%)
      - 0.3 = lightly modified (CRC/content differs but similar structure)
      - 0.0 = unchanged
    """
    origin_by_name = {e["name"]: e for e in origin_entries}
    labels = {}

    for entry in packed_entries:
        name = entry["name"]
        if name not in origin_by_name:
            # New entry — packer injected
            labels[name] = 1.0
        else:
            orig = origin_by_name[name]
            entropy_delta = abs(entry["entropy"] - orig["entropy"])
            size_ratio = entry["size"] / max(orig["size"], 1)

            if entropy_delta > 2.0 or size_ratio > 2.0 or size_ratio < 0.3:
                # Significantly modified
                labels[name] = 0.8
            elif entry["size"] != orig["size"]:
                # Lightly modified
                labels[name] = 0.3
            else:
                # Likely unchanged (same size — could still differ in content
                # but without CRC we assume similar)
                labels[name] = 0.0

    return labels


# ---------------------------------------------------------------------------
# Origin-to-packed filename matching
# ---------------------------------------------------------------------------


def find_origin_for_packed(packed_path: Path, origin_entries_by_stem: Dict) -> Optional[str]:
    """Match a packed APK filename to its origin counterpart.

    Happer naming conventions:
      Origin: 2048_release_signed.apk
      Ali: 2048_release_signed_unsign_sign.apk
      Qihoo: 2048_release_signed_208_jiagu_sign.apk
      Tencent: 2048_release_signed_legu_signed.apk
      etc.

    Strategy: try progressively shorter prefixes of the packed filename
    until we find a match in the origin stems.
    """
    stem = packed_path.stem

    # Known suffix patterns to strip
    suffixes_to_strip = [
        "_unsign_sign",      # Ali
        "_legu_signed",      # Tencent-16
        "_legu_sign",        # Tencent-18
        "_protected_sign",   # Bangcle
        "_unsigned_sign",    # Baidu-16
    ]

    # Try stripping known suffixes first
    for suffix in suffixes_to_strip:
        if stem.endswith(suffix):
            candidate = stem[:-len(suffix)]
            if candidate in origin_entries_by_stem:
                return candidate

    # For Qihoo/Ijiami/Kiwi/Baidu-18: pattern is {name}_{digits}_{packer}_sign
    # Try: remove everything after last known packer keyword
    for keyword in ["_jiagu_sign", "_ijiami_sign", "_kiwi_sign", "_baidu_sign"]:
        if keyword in stem:
            idx = stem.index(keyword)
            # The part before keyword may have _NNN appended
            prefix = stem[:idx]
            # Try stripping trailing _NNN (digits)
            parts = prefix.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                candidate = parts[0]
            else:
                candidate = prefix
            if candidate in origin_entries_by_stem:
                return candidate

    # Fallback: try longest common prefix matching
    best_match = None
    best_len = 0
    for origin_stem in origin_entries_by_stem:
        # Check if packed stem starts with origin stem
        if stem.startswith(origin_stem) and len(origin_stem) > best_len:
            best_match = origin_stem
            best_len = len(origin_stem)

    # Also check if origin stem starts with packed stem (package name based)
    if best_match is None:
        for origin_stem in origin_entries_by_stem:
            # Package-name matching: com.app.name → com.app.name_NNN_sign
            if stem.startswith(origin_stem.split("_")[0]) and len(origin_stem) > best_len:
                best_match = origin_stem
                best_len = len(origin_stem)

    return best_match


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of APKs per packer (0=all, for debugging)")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Happer DS-AMIL Feature Extraction (Block 0)", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Parse all Origin APKs
    print("\n[Step 1] Parsing Origin APKs...", flush=True)
    origin_data: Dict[str, Dict] = {}  # stem -> {path, entries}

    for origin_dir_name in ORIGIN_DIRS:
        origin_dir = HAPPER_DIR / origin_dir_name
        if not origin_dir.exists():
            print(f"  WARNING: {origin_dir} not found, skipping", flush=True)
            continue
        apks = sorted(origin_dir.glob("*.apk"))
        print(f"  {origin_dir_name}: {len(apks)} APKs", flush=True)

        for apk_path in apks:
            if args.limit and len(origin_data) >= args.limit:
                break
            entries = parse_apk_entries(apk_path)
            if entries is not None:
                origin_data[apk_path.stem] = {
                    "path": str(apk_path),
                    "entries": entries,
                    "dir": origin_dir_name,
                }

    print(f"  Total origins parsed: {len(origin_data)}", flush=True)

    # Step 2: Parse packed APKs and compute diff labels
    print("\n[Step 2] Parsing packed APKs + computing diff labels...", flush=True)

    all_apk_records = []  # List of {apk_id, family, features, diff_labels, ...}
    match_stats = {"matched": 0, "unmatched": 0}

    for family, dirs in PACKER_FAMILIES.items():
        family_count = 0
        for dir_name in dirs:
            packer_dir = HAPPER_DIR / dir_name
            if not packer_dir.exists():
                continue

            # Determine which origin set to use (16 or 18)
            year = "16" if "16" in dir_name else "18"
            origin_subset = {k: v for k, v in origin_data.items()
                            if year in v["dir"]}

            apks = sorted(packer_dir.glob("*.apk"))
            for apk_path in apks:
                if args.limit and family_count >= args.limit:
                    break

                packed_entries = parse_apk_entries(apk_path)
                if packed_entries is None:
                    continue

                # Find matching origin
                origin_stem = find_origin_for_packed(apk_path, origin_subset)

                diff_labels = {}
                if origin_stem and origin_stem in origin_data:
                    origin_entries = origin_data[origin_stem]["entries"]
                    diff_labels = compute_diff_labels(origin_entries, packed_entries)
                    match_stats["matched"] += 1
                else:
                    # No match found — all entries get label 0.5 (uncertain)
                    diff_labels = {e["name"]: 0.5 for e in packed_entries}
                    match_stats["unmatched"] += 1

                apk_id = f"{family}__{dir_name}__{apk_path.stem}"
                all_apk_records.append({
                    "apk_id": apk_id,
                    "family": family,
                    "dir": dir_name,
                    "filename": apk_path.name,
                    "path": str(apk_path),
                    "is_packed": True,
                    "entries": packed_entries,
                    "diff_labels": diff_labels,
                })
                family_count += 1

        print(f"  {family}: {family_count} APKs parsed", flush=True)

    # Add origin APKs as benign samples
    for stem, data in origin_data.items():
        apk_id = f"benign__{data['dir']}__{stem}"
        all_apk_records.append({
            "apk_id": apk_id,
            "family": "benign",
            "dir": data["dir"],
            "filename": f"{stem}.apk",
            "path": data["path"],
            "is_packed": False,
            "entries": data["entries"],
            "diff_labels": {e["name"]: 0.0 for e in data["entries"]},
        })

    print(f"\n  Total APKs: {len(all_apk_records)} "
          f"(packed: {sum(1 for r in all_apk_records if r['is_packed'])}, "
          f"benign: {sum(1 for r in all_apk_records if not r['is_packed'])})", flush=True)
    print(f"  Origin matching: {match_stats['matched']} matched, "
          f"{match_stats['unmatched']} unmatched", flush=True)

    # Step 3: Build LOPO splits
    print("\n[Step 3] Building 7-fold LOPO splits...", flush=True)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    families = sorted(PACKER_FAMILIES.keys())
    splits = {}
    for held_out in families:
        train_ids = [r["apk_id"] for r in all_apk_records
                     if r["family"] != held_out]
        test_ids = [r["apk_id"] for r in all_apk_records
                    if r["family"] == held_out or r["family"] == "benign"]
        splits[held_out] = {"train": train_ids, "test": test_ids}
        print(f"  Fold {held_out}: train={len(train_ids)}, test={len(test_ids)}", flush=True)

    with open(splits_dir / "lopo_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Step 4: Save features and labels
    print("\n[Step 4] Saving features and labels...", flush=True)

    # Save as JSONL (features as lists for JSON serialization)
    features_file = out_dir / "entry_features.jsonl"
    labels_file = out_dir / "diff_labels.json"
    apk_index_file = out_dir / "apk_index.json"

    apk_index = {}
    all_diff_labels = {}
    entry_count = 0

    with open(features_file, "w", encoding="utf-8") as fh:
        for record in all_apk_records:
            apk_id = record["apk_id"]
            apk_index[apk_id] = {
                "family": record["family"],
                "path": record["path"],
                "is_packed": record["is_packed"],
                "n_entries": len(record["entries"]),
            }
            all_diff_labels[apk_id] = record["diff_labels"]

            for entry in record["entries"]:
                row = {
                    "apk_id": apk_id,
                    "entry_name": entry["name"],
                    "features": entry["features"].tolist(),
                    "size": entry["size"],
                    "entropy": entry["entropy"],
                    "diff_label": record["diff_labels"].get(entry["name"], 0.0),
                    "apk_label": 1 if record["is_packed"] else 0,
                    "family": record["family"],
                }
                fh.write(json.dumps(row) + "\n")
                entry_count += 1

    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(all_diff_labels, f)

    with open(apk_index_file, "w", encoding="utf-8") as f:
        json.dump(apk_index, f, indent=2)

    # Summary stats
    print(f"\n  Saved {entry_count} entry records to {features_file}", flush=True)
    print(f"  Saved diff labels for {len(all_diff_labels)} APKs to {labels_file}", flush=True)
    print(f"  Saved APK index ({len(apk_index)} APKs) to {apk_index_file}", flush=True)

    # Step 5: Quick diff label statistics
    print("\n[Step 5] Diff label statistics...", flush=True)
    for family in families:
        family_records = [r for r in all_apk_records if r["family"] == family]
        if not family_records:
            continue
        all_labels = []
        for r in family_records:
            all_labels.extend(r["diff_labels"].values())
        n_new = sum(1 for l in all_labels if l >= 0.9)
        n_mod = sum(1 for l in all_labels if 0.3 <= l < 0.9)
        n_unch = sum(1 for l in all_labels if l < 0.3)
        total = len(all_labels)
        print(f"  {family:10s}: {total:4d} entries | "
              f"new(1.0)={n_new:3d} ({100*n_new/max(total,1):.1f}%) | "
              f"mod(0.3-0.8)={n_mod:3d} ({100*n_mod/max(total,1):.1f}%) | "
              f"unchanged={n_unch:3d} ({100*n_unch/max(total,1):.1f}%)",
              flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Block 0 COMPLETE. Next: run Block 1 (XGBoost baseline)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
