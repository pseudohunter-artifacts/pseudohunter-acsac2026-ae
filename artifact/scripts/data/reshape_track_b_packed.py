#!/usr/bin/env python3
"""Reshape Track B packed directory from flat to hierarchical layout.

Flat layout (produced by build_track_b_corpus.py + S6 manual loop):

    data/real_world/track_b/packed/
        <packer_id>__<benign_stem>.apk
        <packer_id>__<benign_stem>.inject_labels.jsonl

Hierarchical layout (expected by run_track_b_labeling.py):

    data/real_world/track_b/packed/
        <packer_id>/
            <benign_stem>/
                packed.apk
                inject_labels.jsonl

Copies files (does not move) so the flat layout stays usable for any
other consumer. Run as:

    python scripts/data/reshape_track_b_packed.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed-dir", type=Path,
                    default=Path("data/real_world/track_b/packed"))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite destination files if they already exist.")
    args = ap.parse_args(argv)

    pdir = args.packed_dir.resolve()
    if not pdir.is_dir():
        print(f"ERROR: packed-dir not found: {pdir}", file=sys.stderr)
        return 2

    apks = sorted(pdir.glob("*__*.apk"))
    if not apks:
        print(f"WARN: no flat <packer>__<benign>.apk files under {pdir}",
              file=sys.stderr)
        return 1

    n_apk = 0
    n_jsonl = 0
    for apk in apks:
        stem = apk.stem  # "s5_timscriptov_..._multiplatform__com.fsck.k9_39035"
        if "__" not in stem:
            continue
        packer_id, benign_stem = stem.split("__", 1)
        out_subdir = pdir / packer_id / benign_stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        dst_apk = out_subdir / "packed.apk"
        if dst_apk.exists() and not args.force:
            print(f"  skip (exists): {dst_apk.relative_to(pdir)}")
        else:
            shutil.copy2(apk, dst_apk)
            print(f"  copied: {apk.name} -> {dst_apk.relative_to(pdir)}")
            n_apk += 1

        # companion jsonl
        jsonl_src = pdir / f"{stem}.inject_labels.jsonl"
        if jsonl_src.exists():
            dst_jsonl = out_subdir / "inject_labels.jsonl"
            if dst_jsonl.exists() and not args.force:
                print(f"  skip (exists): {dst_jsonl.relative_to(pdir)}")
            else:
                shutil.copy2(jsonl_src, dst_jsonl)
                print(f"  copied: {jsonl_src.name} -> {dst_jsonl.relative_to(pdir)}")
                n_jsonl += 1

    print(f"\nDONE: {n_apk} APK(s) + {n_jsonl} inject_labels.jsonl file(s) reshaped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
