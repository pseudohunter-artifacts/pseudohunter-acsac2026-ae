"""CLI: generate a synthetic packed APK with strong payload labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from android_packer.apkio.objects import file_sha256
from android_packer.synthetic import SUPPORTED_TRANSFORMS, build_synthetic_apk


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic APK by injecting a transformed DEX payload."
    )
    parser.add_argument("seed_apk", type=Path, help="Benign seed APK/ZIP.")
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="Optional payload file. If omitted, a DEX object is selected from seed_apk.",
    )
    parser.add_argument(
        "--transform-family",
        choices=SUPPORTED_TRANSFORMS,
        default="xor",
        help="Synthetic payload transform to apply.",
    )
    parser.add_argument(
        "--generated-apk-out",
        type=Path,
        default=None,
        help="Output APK path. Defaults under data/synthetic/generated_apks/.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output manifest JSON path. Defaults under data/synthetic/manifests/.",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Output labels JSONL path. Defaults under data/synthetic/labels/.",
    )
    parser.add_argument("--asset-prefix", default="assets/synthetic")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument(
        "--xor-key",
        type=_parse_int_auto,
        default=None,
        help="XOR key in decimal or hex. Defaults to a deterministic random key.",
    )
    parser.add_argument("--split-count", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    output_paths = _default_output_paths(args.seed_apk, args.transform_family)
    generated_apk_out = args.generated_apk_out or output_paths["generated_apk"]
    manifest_out = args.manifest_out or output_paths["manifest"]
    labels_out = args.labels_out or output_paths["labels"]

    result = build_synthetic_apk(
        seed_apk=args.seed_apk,
        generated_apk_out=generated_apk_out,
        manifest_out=manifest_out,
        labels_out=labels_out,
        payload_path=args.payload,
        transform_family=args.transform_family,
        rng_seed=args.seed,
        asset_prefix=args.asset_prefix,
        xor_key=args.xor_key,
        split_count=args.split_count,
    )

    print(
        " ".join(
            [
                f"generated_apk={result.generated_apk_path}",
                f"manifest={result.manifest_path}",
                f"labels={result.labels_path}",
                f"apk_id={result.manifest['generated_apk_id']}",
                f"injected_objects={len(result.manifest['injected_objects'])}",
            ]
        )
    )
    return 0


def _default_output_paths(seed_apk: Path, transform_family: str) -> dict[str, Path]:
    seed_id = file_sha256(seed_apk)[:12]
    stem = f"{seed_apk.stem}.{seed_id}.{transform_family}"
    return {
        "generated_apk": Path("data/synthetic/generated_apks") / f"{stem}.apk",
        "manifest": Path("data/synthetic/manifests") / f"{stem}.manifest.json",
        "labels": Path("data/synthetic/labels") / f"{stem}.labels.jsonl",
    }


def _parse_int_auto(value: str) -> int:
    return int(value, 0)


if __name__ == "__main__":
    raise SystemExit(main())
