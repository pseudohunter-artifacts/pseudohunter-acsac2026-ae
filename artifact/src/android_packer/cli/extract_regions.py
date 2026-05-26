"""CLI: extract APK objects and byte-window regions into JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.apkio import iter_apk_objects
from android_packer.regioning import iter_regions
from android_packer.utils.jsonl import write_jsonl


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract object metadata and byte-window regions from an APK."
    )
    parser.add_argument("apk", type=Path, help="Path to the input APK/ZIP file.")
    parser.add_argument(
        "--objects-out",
        type=Path,
        default=Path("data/processed/objects/objects.jsonl"),
        help="Output JSONL path for object metadata.",
    )
    parser.add_argument(
        "--regions-out",
        type=Path,
        default=Path("data/processed/regions/regions.jsonl"),
        help="Output JSONL path for region metadata.",
    )
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--min-region-size", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-member-bytes", type=int, default=None)
    parser.add_argument(
        "--no-include-tail",
        action="store_true",
        help="Do not add a final tail-aligned window.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    include_tail = not args.no_include_tail

    object_rows: list[dict] = []
    region_rows: list[dict] = []
    for metadata, data in iter_apk_objects(
        args.apk,
        max_depth=args.max_depth,
        max_member_bytes=args.max_member_bytes,
    ):
        object_rows.append(metadata.to_dict())
        for region in iter_regions(
            metadata,
            data,
            window_size=args.window_size,
            stride=args.stride,
            min_region_size=args.min_region_size,
            include_tail=include_tail,
        ):
            region_rows.append(region.to_dict())

    object_count = write_jsonl(args.objects_out, object_rows)
    region_count = write_jsonl(args.regions_out, region_rows)
    print(f"objects={object_count} regions={region_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
