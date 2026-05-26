"""Build an APKiD-audited hard benign APK manifest.

The manifest contains metadata only: local paths, hashes, APK structure counts,
hardness flags, and APKiD audit summaries. APK bytes remain under gitignored
data directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android_packer.apkio.objects import ApkReadError, iter_apk_objects
from android_packer.labeling.apkid_cross_check import run_apkid


STRICT_DPT_BENIGN_DIR = ROOT / "data" / "real_world" / "track_b_v2" / "benign"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _apk_structure(path: Path) -> Dict:
    counts = Counter()
    native_abis = set()
    max_entropy = 0.0
    high_entropy_entries = 0
    largest_entry = 0
    compressed_entries = 0
    total_entries = 0

    for meta, data in iter_apk_objects(path, max_depth=1):
        total_entries += 1
        counts[meta.object_type] += 1
        largest_entry = max(largest_entry, meta.size)
        if meta.compression != "stored":
            compressed_entries += 1
        ent = _entropy(data[: min(len(data), 1024 * 1024)])
        max_entropy = max(max_entropy, ent)
        if ent >= 7.2 and meta.size >= 64 * 1024:
            high_entropy_entries += 1
        lower = meta.object_path.lower().replace("\\", "/")
        if lower.startswith("lib/") and lower.endswith(".so"):
            parts = lower.split("/")
            if len(parts) >= 3:
                native_abis.add(parts[1])

    size = path.stat().st_size
    flags = {
        "large_apk": size >= 20 * 1024 * 1024,
        "native_heavy": counts["native_lib"] >= 4 or len(native_abis) >= 2,
        "asset_heavy": counts["asset_blob"] + counts["resource"] >= 40,
        "multi_dex": counts["dex"] >= 2,
        "high_entropy": high_entropy_entries >= 1 or max_entropy >= 7.5,
        "large_entry": largest_entry >= 5 * 1024 * 1024,
        "many_entries": total_entries >= 200,
    }
    hardness_score = sum(1 for enabled in flags.values() if enabled)
    return {
        "entry_count": total_entries,
        "dex_count": counts["dex"],
        "native_lib_count": counts["native_lib"],
        "asset_count": counts["asset_blob"],
        "resource_count": counts["resource"],
        "archive_count": counts["embedded_archive"],
        "unknown_count": counts["unknown_blob"],
        "compressed_entry_count": compressed_entries,
        "native_abis": sorted(native_abis),
        "largest_entry_size": largest_entry,
        "max_sampled_entry_entropy": round(max_entropy, 4),
        "high_entropy_entry_count": high_entropy_entries,
        "hardness_flags": flags,
        "hardness_score": hardness_score,
    }


def _apkid_summary(path: Path, apkid_cmd: str, timeout: float) -> Dict:
    result = run_apkid(path, apkid_cmd=apkid_cmd, timeout=timeout, graceful=True)
    packer_like = result.packer_like_matches()
    return {
        "apkid_version": result.apkid_version,
        "apkid_error": result.error,
        "apkid_clean": result.error is None and not packer_like,
        "packer_like_hits": [
            {"category": hit.category, "hit": hit.hit, "filename": hit.filename}
            for hit in packer_like
        ],
        "all_hits": [
            {"category": hit.category, "hit": hit.hit, "filename": hit.filename}
            for hit in result.matches
        ],
    }


def _candidate_paths(apk_dirs: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for apk_dir in apk_dirs:
        if apk_dir.is_file() and apk_dir.suffix.lower() == ".apk":
            paths = [apk_dir]
        else:
            paths = sorted(apk_dir.rglob("*.apk")) if apk_dir.exists() else []
        for path in paths:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            yield path


def _source_from_path(path: Path) -> str:
    p = path.as_posix().lower()
    if "track_b_v2" in p:
        return "track_b_v2_strict_benign"
    if "track_b/benign" in p:
        return "track_b_fdroid_benign"
    if "androzoo" in p:
        return "androzoo"
    if "happer_dataset" in p:
        return "happer_origin"
    return "unknown"


def build_manifest(args) -> Dict:
    records: List[Dict] = []
    failures: List[Dict] = []

    for i, path in enumerate(_candidate_paths(args.apk_dirs), 1):
        source = _source_from_path(path)
        is_strict_test = _is_under(path, STRICT_DPT_BENIGN_DIR)
        try:
            structure = _apk_structure(path)
            apkid = _apkid_summary(path, args.apkid_cmd, args.apkid_timeout)
            sha = _sha256(path)
            label_class = (
                "benign-hard-clean"
                if apkid["apkid_clean"] and structure["hardness_score"] >= args.hard_threshold
                else "benign-clean"
                if apkid["apkid_clean"]
                else "benign-borderline"
            )
            train_allowed = bool(apkid["apkid_clean"] and not is_strict_test)
            records.append(
                {
                    "schema_version": 1,
                    "sha256": sha,
                    "local_path": _relative_to_root(path),
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "source": source,
                    "strict_dpt_test_set": is_strict_test,
                    "train_allowed": train_allowed,
                    "label_class": label_class,
                    **structure,
                    **apkid,
                }
            )
        except (ApkReadError, OSError, RuntimeError) as exc:
            failures.append(
                {
                    "local_path": _relative_to_root(path),
                    "source": source,
                    "error": str(exc),
                }
            )
        if i % 10 == 0:
            print(f"  scanned {i} APKs -> {len(records)} records", flush=True)

    counts = Counter(record["label_class"] for record in records)
    train_allowed = sum(1 for record in records if record["train_allowed"])
    hard_train = sum(
        1 for record in records
        if record["train_allowed"] and record["label_class"] == "benign-hard-clean"
    )
    return {
        "schema_version": 1,
        "generated_by": "scripts/data/build_hard_benign_manifest.py",
        "selection": {
            "apk_dirs": [_relative_to_root(path) for path in args.apk_dirs],
            "hard_threshold": args.hard_threshold,
            "apkid_cmd": args.apkid_cmd,
            "apkid_timeout": args.apkid_timeout,
        },
        "summary": {
            "records": len(records),
            "failures": len(failures),
            "train_allowed": train_allowed,
            "hard_train_allowed": hard_train,
            "label_class_counts": dict(counts),
        },
        "records": records,
        "failures": failures,
    }


def _write_jsonl(records: Sequence[Mapping], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _write_markdown(payload: Mapping, path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Hard Benign Manifest Audit",
        "",
        f"Records: {summary['records']}",
        f"Failures: {summary['failures']}",
        f"Train-allowed clean APKs: {summary['train_allowed']}",
        f"Train-allowed hard-clean APKs: {summary['hard_train_allowed']}",
        "",
        "| Label class | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(summary["label_class_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "| APK | Source | Class | Train | Size MB | Hardness | Flags | APKiD packer hits |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for record in payload["records"]:
        flags = [
            name for name, enabled in record["hardness_flags"].items()
            if enabled
        ]
        hits = ", ".join(hit["hit"] for hit in record["packer_like_hits"])
        lines.append(
            "| `{apk}` | {source} | {klass} | {train} | {size:.1f} | {hardness} | {flags} | {hits} |".format(
                apk=record["file_name"],
                source=record["source"],
                klass=record["label_class"],
                train="yes" if record["train_allowed"] else "no",
                size=record["size_bytes"] / (1024 * 1024),
                hardness=record["hardness_score"],
                flags=", ".join(flags) if flags else "-",
                hits=hits if hits else "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk-dir", dest="apk_dirs", type=Path, action="append", required=True)
    parser.add_argument("--apkid-cmd", default="apkid")
    parser.add_argument("--apkid-timeout", type=float, default=120.0)
    parser.add_argument("--hard-threshold", type=int, default=2)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "hard_benign" / "manifest.json",
    )
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    payload = build_manifest(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_jsonl = args.out_jsonl or args.out_json.with_suffix(".jsonl")
    out_md = args.out_md or args.out_json.with_suffix(".md")
    _write_jsonl(payload["records"], out_jsonl)
    _write_markdown(payload, out_md)

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"JSON: {args.out_json}")
    print(f"JSONL: {out_jsonl}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
