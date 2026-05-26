"""Build an auditable packed/unpacked pair manifest with optional APKiD checks.

This script does not download APKs and does not run a packer. It stitches
already-downloaded benign seed APKs (AndroZoo or F-Droid) together with packed
outputs produced by an external packer pipeline, then records APKiD evidence for
both sides.

Expected packed layout by default:

    <packed-dir>/<packer-id>/<unpacked-stem>/packed.apk

Flat names are also accepted when they start with ``<packer-id>__`` and contain
the unpacked APK stem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    packer_id: str
    source: str
    package_hint: str
    unpacked_apk: str
    packed_apk: Optional[str]
    unpacked_sha256: str
    packed_sha256: Optional[str]
    status: str
    apkid_unpacked_clean: Optional[bool]
    apkid_packed_has_packer: Optional[bool]
    apkid_unpacked: Optional[Dict[str, Any]]
    apkid_packed: Optional[Dict[str, Any]]
    notes: List[str]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_apks(paths: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for root in paths:
        if root.is_file() and root.suffix.lower() == ".apk":
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
        elif root.is_dir():
            for apk in sorted(root.rglob("*.apk")):
                resolved = apk.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def stem_matches(stem: str, text: str, *, min_prefix: int = 12) -> bool:
    """Return true when ``text`` contains the full stem or a stable prefix."""
    stem_l = stem.lower()
    text_l = text.lower()
    if stem_l in text_l:
        return True
    if len(stem_l) >= min_prefix and stem_l[:min_prefix] in text_l:
        return True
    return False


def find_packed_apk(packed_dir: Path, packer_id: str, unpacked: Path) -> Optional[Path]:
    stem = unpacked.stem
    hierarchical = packed_dir / packer_id / stem / "packed.apk"
    if hierarchical.exists():
        return hierarchical

    candidates = []
    for apk in packed_dir.rglob("*.apk"):
        name = apk.name
        rel = apk.relative_to(packed_dir).as_posix()
        if packer_id in rel and stem_matches(stem, rel):
            candidates.append(apk)
        elif name.startswith(f"{packer_id}__") and stem_matches(stem, name):
            candidates.append(apk)
    if len(candidates) == 1:
        return candidates[0]
    return None


def run_apkid_report(apk: Path, *, apkid_cmd: str, timeout: float):
    from android_packer.labeling.apkid_cross_check import (
        cross_check_apk,
        load_apkid_family_map,
    )

    family_map_path = ROOT / "configs" / "data" / "apkid_family_map.yaml"
    family_map = load_apkid_family_map(family_map_path) if family_map_path.exists() else None
    return cross_check_apk(
        apk,
        apk_id=apk.stem,
        expected_family=None,
        family_map=family_map,
        apkid_cmd=apkid_cmd,
        timeout=timeout,
        graceful=True,
    )


def apkid_summary(report) -> Dict[str, Any]:
    result = report.apkid_result
    packer_like = result.packer_like_matches()
    return {
        "agreement": report.agreement,
        "needs_manual_review": report.needs_manual_review,
        "has_packer_hit": report.has_packer_hit,
        "has_protector_hit": report.has_protector_hit,
        "detected_families": list(report.detected_families),
        "hits": [
            {"category": m.category, "hit": m.hit, "filename": m.filename}
            for m in packer_like
        ],
        "error": result.error,
        "apkid_version": result.apkid_version,
        "rules_sha256": result.rules_sha256,
    }


def build_records(args: argparse.Namespace) -> List[PairRecord]:
    records: List[PairRecord] = []
    for unpacked in iter_apks(args.unpacked_dirs):
        unpacked_sha = sha256_file(unpacked)
        for packer_id in args.packers:
            notes: List[str] = []
            packed = find_packed_apk(args.packed_dir, packer_id, unpacked)
            packed_sha = sha256_file(packed) if packed is not None else None

            unpacked_report = packed_report = None
            if args.run_apkid:
                unpacked_report = run_apkid_report(
                    unpacked, apkid_cmd=args.apkid_cmd, timeout=args.apkid_timeout
                )
                if packed is not None:
                    packed_report = run_apkid_report(
                        packed, apkid_cmd=args.apkid_cmd, timeout=args.apkid_timeout
                    )

            unpacked_clean = None
            packed_has_packer = None
            if unpacked_report is not None:
                unpacked_clean = not (
                    unpacked_report.has_packer_hit or unpacked_report.has_protector_hit
                )
                if not unpacked_clean:
                    notes.append("unpacked seed has APKiD packer/protector hit")
            if packed_report is not None:
                packed_has_packer = packed_report.has_packer_hit or packed_report.has_protector_hit
                if not packed_has_packer:
                    notes.append("packed APK has no APKiD packer/protector hit")
            if packed is None:
                notes.append("packed counterpart not found")

            status = "paired"
            if packed is None:
                status = "missing_packed"
            elif unpacked_clean is False:
                status = "unpacked_not_clean"
            elif packed_has_packer is False:
                status = "packed_not_apkid_confirmed"

            pair_id = f"{packer_id}__{unpacked.stem}"
            records.append(
                PairRecord(
                    pair_id=pair_id,
                    packer_id=packer_id,
                    source=args.source,
                    package_hint=unpacked.stem,
                    unpacked_apk=str(unpacked),
                    packed_apk=str(packed) if packed is not None else None,
                    unpacked_sha256=unpacked_sha,
                    packed_sha256=packed_sha,
                    status=status,
                    apkid_unpacked_clean=unpacked_clean,
                    apkid_packed_has_packer=packed_has_packer,
                    apkid_unpacked=apkid_summary(unpacked_report) if unpacked_report else None,
                    apkid_packed=apkid_summary(packed_report) if packed_report else None,
                    notes=notes,
                )
            )
    return records


def write_jsonl(path: Path, rows: Iterable[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_summary(path: Path, rows: List[PairRecord]) -> None:
    counts: Dict[str, int] = {}
    by_packer: Dict[str, Dict[str, int]] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        bucket = by_packer.setdefault(row.packer_id, {})
        bucket[row.status] = bucket.get(row.status, 0) + 1
    summary = {
        "n_records": len(rows),
        "status_counts": counts,
        "by_packer": by_packer,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unpacked-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories or APK files containing unpacked/benign seed APKs.",
    )
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--packers", nargs="+", required=True)
    parser.add_argument("--source", default="unknown", choices=["androzoo", "fdroid", "mixed", "unknown"])
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/real_world/paired_packed_apks/pairs.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/experiments/paired_packed_apks/summary.json"),
    )
    parser.add_argument("--run-apkid", action="store_true")
    parser.add_argument("--apkid-cmd", default="apkid")
    parser.add_argument("--apkid-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    records = build_records(args)
    write_jsonl(args.out_jsonl, records)
    write_summary(args.summary_out, records)

    print(f"records={len(records)}")
    print(f"jsonl={args.out_jsonl}")
    print(f"summary={args.summary_out}")
    if args.run_apkid:
        clean = sum(1 for r in records if r.apkid_unpacked_clean is True)
        packed_hit = sum(1 for r in records if r.apkid_packed_has_packer is True)
        print(f"apkid_unpacked_clean={clean}/{len(records)}")
        print(f"apkid_packed_has_packer={packed_hit}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
