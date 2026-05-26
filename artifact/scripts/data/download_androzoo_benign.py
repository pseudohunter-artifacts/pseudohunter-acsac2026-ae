"""Download benign APKs from AndroZoo for spMLM pretraining corpus.

Uses the existing AndroZoo CSV index to select clean APKs (vt_detection=0),
then downloads via the existing fetch_androzoo.py infrastructure.

Selection criteria:
- vt_detection == 0 (no malware detection on VirusTotal)
- dex_date >= 2020 (modern APKs)
- apk_size within a configurable range
- markets contains 'play' when requested; otherwise Play Store apps are preferred

Usage:
    python scripts/data/download_androzoo_benign.py --target 1500 --execute
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ANDROZOO_DIR = ROOT / "data" / "androzoo"
CSV_PATH = ANDROZOO_DIR / "latest_with-added-date.csv.gz"
APK_DIR = ANDROZOO_DIR / "apks"
BENIGN_DIR = ANDROZOO_DIR / "benign_corpus"
DOWNLOAD_URL = "https://androzoo.uni.lu/api/download"
USER_AGENT = "android-packer-androzoo-benign-fetcher/1.0 (+research)"


def _has_market(markets: str, market_substr: str) -> bool:
    if not market_substr:
        return True
    return market_substr.lower() in markets.lower()


def _candidate_rank(candidate: dict, *, preferred_market: str = "play") -> tuple:
    """Higher is better: preferred market first, then newer, then larger."""
    markets = str(candidate.get("markets", ""))
    return (
        1 if _has_market(markets, preferred_market) else 0,
        str(candidate.get("dex_date", "")),
        int(candidate.get("apk_size", 0)),
    )


def _sha256_of(path: Path, *, buf_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_candidates(
    csv_path: Path,
    target: int = 1500,
    min_dex_date: str = "2020",
    min_size_mb: float = 0.1,
    max_size_mb: float = 30.0,
    require_market: str = "",
    preferred_market: str = "play",
    candidate_multiplier: int = 20,
    max_scan_rows: int = 0,
) -> List[dict]:
    """Select benign APK candidates from AndroZoo CSV index."""
    print(f"[Select] Reading {csv_path.name} ...", flush=True)
    if target <= 0:
        raise ValueError("target must be positive")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")

    pool_size = max(target, target * candidate_multiplier)
    heap: list[tuple[tuple, int, dict]] = []
    matched = 0
    scanned = 0

    # CSV columns: sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,vt_detection,
    #              vt_scan_date,dex_size,markets
    with gzip.open(csv_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_scan_rows and i >= max_scan_rows:
                break
            scanned = i + 1

            # Filter: vt_detection == 0
            vt = row.get("vt_detection", "")
            if vt != "0":
                continue

            # Filter: dex_date >= 2020
            dex_date = row.get("dex_date", "")
            if not dex_date or dex_date < min_dex_date:
                continue

            # Filter: size < max_size_mb
            try:
                apk_size = int(row.get("apk_size", "0"))
            except ValueError:
                continue
            if apk_size > max_size_mb * 1024 * 1024:
                continue
            if apk_size < min_size_mb * 1024 * 1024:
                continue

            markets = row.get("markets", "")
            if require_market and not _has_market(markets, require_market):
                continue

            candidate = {
                "sha256": row.get("sha256", ""),
                "pkg_name": row.get("pkg_name", ""),
                "apk_size": apk_size,
                "dex_date": dex_date,
                "markets": markets,
                "vt_detection": 0,
            }
            rank = _candidate_rank(candidate, preferred_market=preferred_market)
            item = (rank, matched, candidate)
            matched += 1
            if len(heap) < pool_size:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)

            if scanned % 1000000 == 0:
                print(
                    f"  Scanned {scanned:,} rows, matched {matched:,}, "
                    f"kept {len(heap):,} candidates ...",
                    flush=True,
                )

    ranked = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
    candidates = [candidate for _, _, candidate in ranked]

    # Take top N
    selected = candidates[:target]
    print(
        f"[Select] Selected {len(selected)} candidates from "
        f"{scanned:,} rows scanned ({matched:,} matched)",
        flush=True,
    )
    return selected


def download_apks(
    candidates: List[dict],
    out_dir: Path,
    api_key: str,
    *,
    sleep_sec: float = 0.3,
    dry_run: bool = True,
) -> dict:
    """Download APKs from AndroZoo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total": len(candidates)}

    for i, cand in enumerate(candidates):
        sha256 = cand["sha256"]
        if not sha256 or len(sha256) != 64:
            stats["failed"] += 1
            continue

        # Shard by first 2 chars
        shard_dir = out_dir / sha256[:2]
        apk_path = shard_dir / f"{sha256}.apk"

        if apk_path.exists():
            if _sha256_of(apk_path) == sha256.lower():
                stats["skipped"] += 1
                continue
            apk_path.unlink()

        if dry_run:
            stats["skipped"] += 1
            continue

        # Download
        shard_dir.mkdir(parents=True, exist_ok=True)
        url = f"{DOWNLOAD_URL}?sha256={sha256}&apikey={api_key}"
        tmp_path = apk_path.with_suffix(apk_path.suffix + ".part")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp, tmp_path.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
            actual_sha = _sha256_of(tmp_path)
            if actual_sha != sha256.lower():
                stats["failed"] += 1
                tmp_path.unlink(missing_ok=True)
                print(
                    f"  SHA256 mismatch for {sha256}: got {actual_sha}",
                    flush=True,
                )
                time.sleep(sleep_sec)
                continue
            tmp_path.replace(apk_path)
            stats["downloaded"] += 1
            if stats["downloaded"] % 10 == 0:
                print(f"  Downloaded {stats['downloaded']}/{stats['total']} "
                      f"(skip={stats['skipped']}, fail={stats['failed']})", flush=True)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            stats["failed"] += 1
            tmp_path.unlink(missing_ok=True)

        time.sleep(sleep_sec)

    return stats


def write_candidate_files(candidates: List[dict], sha_file: Path, jsonl_file: Path) -> None:
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with sha_file.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(f"{c['sha256']}\n")
    with jsonl_file.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1500,
                        help="Target number of benign APKs to download")
    parser.add_argument("--execute", action="store_true",
                        help="Actually download (default is dry-run)")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("ANDROZOO_API_KEY", ""),
                        help="AndroZoo API key (or set ANDROZOO_API_KEY env var)")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--min-dex-date", default="2020")
    parser.add_argument("--min-size-mb", type=float, default=0.1)
    parser.add_argument("--max-size-mb", type=float, default=30.0)
    parser.add_argument("--require-market", default="",
                        help="Only keep rows whose markets field contains this substring")
    parser.add_argument("--preferred-market", default="play",
                        help="Rank rows containing this market substring first")
    parser.add_argument("--candidate-multiplier", type=int, default=20,
                        help="Keep this many target-sized candidates while scanning")
    parser.add_argument("--max-scan-rows", type=int, default=0,
                        help="Debug/testing limit; 0 scans the full CSV")
    parser.add_argument("--out-dir", type=Path, default=BENIGN_DIR)
    parser.add_argument("--sha-file", type=Path,
                        default=ANDROZOO_DIR / "benign_candidates.txt")
    parser.add_argument("--candidate-jsonl", type=Path,
                        default=ANDROZOO_DIR / "benign_candidates.jsonl")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: AndroZoo CSV not found at {CSV_PATH}")
        print(f"  Download it first: curl -o {CSV_PATH} https://androzoo.uni.lu/static/lists/latest_with-added-date.csv.gz")
        sys.exit(1)

    if args.execute and not args.api_key:
        print("ERROR: --execute requires --api-key or ANDROZOO_API_KEY env var")
        sys.exit(1)

    # Select candidates
    candidates = select_candidates(
        CSV_PATH,
        target=args.target,
        min_dex_date=args.min_dex_date,
        min_size_mb=args.min_size_mb,
        max_size_mb=args.max_size_mb,
        require_market=args.require_market,
        preferred_market=args.preferred_market,
        candidate_multiplier=args.candidate_multiplier,
        max_scan_rows=args.max_scan_rows,
    )

    if not candidates:
        print("No candidates found!")
        sys.exit(1)

    write_candidate_files(candidates, args.sha_file, args.candidate_jsonl)
    print(f"[Select] Saved SHA256s to {args.sha_file}")
    print(f"[Select] Saved metadata to {args.candidate_jsonl}")

    # Check how many already downloaded
    already = 0
    for cand in candidates:
        sha = cand["sha256"]
        if (args.out_dir / sha[:2] / f"{sha}.apk").exists():
            already += 1
    print(f"[Status] Already have {already}/{len(candidates)} APKs", flush=True)

    if not args.execute:
        print(f"\n[DRY RUN] Would download {len(candidates) - already} APKs to {args.out_dir}")
        print(f"  Run with --execute to actually download")
        return

    # Download
    print(f"\n[Download] Fetching {len(candidates) - already} APKs ...", flush=True)
    stats = download_apks(
        candidates, args.out_dir, args.api_key,
        sleep_sec=args.sleep, dry_run=False,
    )
    print(f"\n[Done] {stats}")


if __name__ == "__main__":
    main()
