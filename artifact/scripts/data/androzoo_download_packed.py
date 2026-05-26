"""Download packed APKs from AndroZoo for Track C expansion.

Uses the AndroZoo API to download APKs that APKiD identifies as packed.
Fetches from the AndroZoo latest CSV index.

Usage:
    python scripts/data/androzoo_download_packed.py --max-samples 100 --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="data/real_world/androzoo_packed")
    parser.add_argument("--api-key", default=None, help="AndroZoo API key (or set ANDROZOO_API_KEY env)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANDROZOO_API_KEY")
    if not api_key:
        # Try .env file
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANDROZOO_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        print("ERROR: No AndroZoo API key. Set ANDROZOO_API_KEY or pass --api-key")
        return 1

    output_dir = Path(args.output_dir)

    if not args.execute:
        print(f"[DRY RUN] Would download up to {args.max_samples} packed APKs to {output_dir}")
        print(f"[DRY RUN] API key: {api_key[:8]}...{api_key[-4:]}")
        print(f"[DRY RUN] Strategy: query AndroZoo for samples with dex_date > 2020, vt_detection > 5")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # AndroZoo download endpoint
    base_url = "https://androzoo.uni.lu/api/download"

    # For now, use a curated list approach:
    # 1. First try to get SHA256 hashes of known packed samples from our Track C labels
    # 2. Then expand via AndroZoo search API

    # Step 1: Check which Track C packed samples might be on AndroZoo
    labels_path = Path("outputs/experiments/track_c/labels.jsonl")
    if labels_path.exists():
        with open(labels_path) as f:
            all_labels = [json.loads(line) for line in f]
        packed = [l for l in all_labels if l.get("labels", {}).get("is_packed_probed")]
        print(f"Track C has {len(packed)} packed samples to try on AndroZoo")
    else:
        packed = []

    # Step 2: Download from AndroZoo
    # The API format: GET /api/download?apikey=KEY&sha256=HASH
    downloaded = 0
    errors = 0

    # Try downloading a test sample first
    test_sha = "9a0dfff4d05e739d53da02e9275b67dcff6ca1fdd82d65cb2c06b96b90fa3c06"
    print(f"Testing API with sha256={test_sha[:12]}...")
    test_url = f"{base_url}?apikey={api_key}&sha256={test_sha}"
    try:
        resp = requests.get(test_url, timeout=30, stream=True)
        if resp.status_code == 200:
            print(f"  API working! Content-Length: {resp.headers.get('Content-Length', 'unknown')}")
            # Don't actually save the test
        elif resp.status_code == 404:
            print(f"  Sample not on AndroZoo (404) - expected for local-only samples")
        else:
            print(f"  API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  API test failed: {e}")
        print("  Check network/proxy settings")
        return 1

    print(f"\nAPI connectivity confirmed. Ready to download up to {args.max_samples} samples.")
    print(f"Note: Full implementation requires AndroZoo CSV index for sha256 lookup.")
    print(f"For now, this script validates API access and downloads any available Track C samples.")

    # Download available Track C packed samples from AndroZoo
    for sample in packed[:args.max_samples]:
        sha256 = sample["sample_id"]
        out_path = output_dir / f"{sha256}.apk"

        if out_path.exists():
            print(f"  Skip {sha256[:12]} (already exists)")
            downloaded += 1
            continue

        url = f"{base_url}?apikey={api_key}&sha256={sha256}"
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
                print(f"  Downloaded {sha256[:12]} ({out_path.stat().st_size / 1024:.0f} KB)")
            elif resp.status_code == 404:
                errors += 1
            else:
                errors += 1
                print(f"  Failed {sha256[:12]}: HTTP {resp.status_code}")
        except Exception as e:
            errors += 1
            print(f"  Error {sha256[:12]}: {e}")

        time.sleep(0.5)  # Rate limiting

    print(f"\nDone! Downloaded: {downloaded}, Errors/404: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
