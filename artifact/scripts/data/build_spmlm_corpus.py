"""Build spMLM pretraining corpus from benign APKs.

Decodes all DEX/ELF/asset regions in benign APKs into three token streams
(Dalvik, Native, Byte) and caches them for fast spMLM training.

Usage:
    python scripts/data/build_spmlm_corpus.py [--apk-dir PATH] [--out-dir PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android_packer.apkio.objects import iter_apk_objects
from android_packer.decoders.pseudo_tokenizer import (
    BYTE_REPRESENTATIONS,
    BYTE_REPRESENTATION_LEGACY_RAW,
    PseudoCodeTokenizer,
)
from android_packer.regioning.typed_slicer import iter_typed_regions

DEFAULT_APK_DIRS = [
    ROOT / "data" / "androzoo" / "benign_corpus",
    ROOT / "data" / "happer_dataset" / "FSet" / "Origin-16",
    ROOT / "data" / "happer_dataset" / "FSet" / "Oirgin-18",
]
DEFAULT_OUT_DIR = ROOT / "data" / "pretrain_cache"


def get_dex_header_counts(data: bytes) -> tuple:
    """Extract string/type/method/field counts from DEX header."""
    if len(data) < 100 or data[:4] != b"dex\n":
        return (0, 0, 0, 0)
    try:
        string_ids = struct.unpack_from("<I", data, 56)[0]
        type_ids = struct.unpack_from("<I", data, 64)[0]
        field_ids = struct.unpack_from("<I", data, 80)[0]
        method_ids = struct.unpack_from("<I", data, 88)[0]
        return (string_ids, type_ids, method_ids, field_ids)
    except (struct.error, IndexError):
        return (0, 0, 0, 0)


def process_apk(
    apk_path: Path,
    tokenizer: PseudoCodeTokenizer,
    max_regions_per_entry: int = 10,
) -> List[Dict]:
    """Process one APK into spMLM training sequences."""
    sequences = []

    try:
        entries = []
        dex_counts = {}  # entry_path → header counts

        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) < 256:
                continue
            entries.append((obj_meta, obj_bytes))
            # Cache DEX header counts for validation
            if obj_bytes[:4] == b"dex\n":
                dex_counts[obj_meta.object_path] = get_dex_header_counts(obj_bytes)

        for entry_idx, (obj_meta, obj_bytes) in enumerate(entries):
            regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=entry_idx)

            # Limit regions per entry for corpus diversity
            for region in regions[:max_regions_per_entry]:
                region_data = obj_bytes[region.offset_start:region.offset_end]
                if len(region_data) < 64:
                    continue

                # Get DEX header counts if this is a DEX region
                header_counts = dex_counts.get(obj_meta.object_path, (0, 0, 0, 0))

                # Encode through all three decoders
                dalvik_enc, native_enc, byte_enc = tokenizer.encode_region(
                    region_data,
                    entry_type=region.entry_type,
                    dex_header_counts=header_counts,
                )

                # Build abnormal masks from decoder output
                dalvik_abnormal = [0] * len(dalvik_enc.token_ids)
                native_abnormal = [0] * len(native_enc.token_ids)

                # Add all three streams
                sequences.append({
                    "token_ids": dalvik_enc.token_ids,
                    "token_type_ids": dalvik_enc.token_type_ids,
                    "attention_mask": dalvik_enc.attention_mask,
                    "abnormal_mask": dalvik_abnormal,
                    "stream": "dalvik",
                    "entry_type": region.entry_type,
                    "section_type": region.section_type,
                })
                sequences.append({
                    "token_ids": native_enc.token_ids,
                    "token_type_ids": native_enc.token_type_ids,
                    "attention_mask": native_enc.attention_mask,
                    "abnormal_mask": native_abnormal,
                    "stream": "native",
                    "entry_type": region.entry_type,
                    "section_type": region.section_type,
                })
                sequences.append({
                    "token_ids": byte_enc.token_ids,
                    "token_type_ids": byte_enc.token_type_ids,
                    "attention_mask": byte_enc.attention_mask,
                    "abnormal_mask": [0] * len(byte_enc.token_ids),
                    "stream": "byte",
                    "entry_type": region.entry_type,
                    "section_type": region.section_type,
                })

    except Exception as e:
        pass  # Skip broken APKs

    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk-dirs", nargs="+", type=Path, default=None,
                        help="Directories containing benign APKs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Limit APKs (0=all)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-regions-per-entry", type=int, default=10)
    parser.add_argument("--byte-representation", type=str,
                        default=BYTE_REPRESENTATION_LEGACY_RAW,
                        choices=BYTE_REPRESENTATIONS,
                        help="Byte-path representation to encode in the corpus")
    args = parser.parse_args()

    apk_dirs = args.apk_dirs or DEFAULT_APK_DIRS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PseudoCodeTokenizer(
        max_length=args.max_length,
        byte_representation=args.byte_representation,
    )

    # Collect all APK paths
    apk_paths = []
    for d in apk_dirs:
        if not d.exists():
            print(f"  SKIP (not found): {d}")
            continue
        # Handle both flat and sharded directories
        for p in sorted(d.rglob("*.apk")):
            apk_paths.append(p)

    if args.limit:
        apk_paths = apk_paths[:args.limit]

    print(f"[Corpus] Found {len(apk_paths)} APKs from {len(apk_dirs)} directories")
    print(f"[Corpus] max_length={args.max_length}, max_regions_per_entry={args.max_regions_per_entry}")
    print(f"[Corpus] byte_representation={args.byte_representation}")
    print()

    # Process APKs
    all_sequences = []
    t0 = time.time()

    for i, apk_path in enumerate(apk_paths):
        seqs = process_apk(apk_path, tokenizer, args.max_regions_per_entry)
        all_sequences.extend(seqs)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(apk_paths) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(apk_paths)}] {len(all_sequences)} seqs, "
                  f"{rate:.1f} APK/s, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n[Corpus] Processed {len(apk_paths)} APKs in {elapsed:.1f}s")
    print(f"[Corpus] Total sequences: {len(all_sequences)}")

    # Save as numpy arrays for fast loading
    print(f"[Corpus] Saving to {args.out_dir} ...", flush=True)

    # Split into chunks for memory efficiency
    chunk_size = 10000
    n_chunks = (len(all_sequences) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(all_sequences))
        chunk = all_sequences[start:end]

        token_ids = np.array([s["token_ids"] for s in chunk], dtype=np.int16)
        token_types = np.array([s["token_type_ids"] for s in chunk], dtype=np.int8)
        attn_masks = np.array([s["attention_mask"] for s in chunk], dtype=np.int8)
        abnormal = np.array([s["abnormal_mask"] for s in chunk], dtype=np.int8)

        np.savez_compressed(
            args.out_dir / f"corpus_chunk_{chunk_idx:04d}.npz",
            token_ids=token_ids,
            token_type_ids=token_types,
            attention_mask=attn_masks,
            abnormal_mask=abnormal,
        )

    # Save metadata
    meta = {
        "n_sequences": len(all_sequences),
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "max_length": args.max_length,
        "byte_representation": args.byte_representation,
        "vocab_size": tokenizer.vocab_size,
        "n_apks": len(apk_paths),
        "streams": {
            "dalvik": sum(1 for s in all_sequences if s["stream"] == "dalvik"),
            "native": sum(1 for s in all_sequences if s["stream"] == "native"),
            "byte": sum(1 for s in all_sequences if s["stream"] == "byte"),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out_dir / "corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Corpus] Saved {n_chunks} chunks + metadata")
    print(f"  Dalvik: {meta['streams']['dalvik']}")
    print(f"  Native: {meta['streams']['native']}")
    print(f"  Byte:   {meta['streams']['byte']}")


if __name__ == "__main__":
    main()
