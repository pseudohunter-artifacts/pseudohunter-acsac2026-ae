"""Track C wild-malware corpus inventory.

Walks ``data/real_world/wild/`` and emits a machine-readable JSON summary of
what raw material is present today. Output schema is documented in
``docs/workstreams/track_c/corpus_schema.md``.

Two sources are supported out of the box:

* ``ashishb_android_malware/``
    One directory per malware family; APKs (plus supporting artefacts like
    ``.txt`` notes, ``.bin`` payloads) are directly under the family dir.
* ``sk3ptre_AndroidMalware_2020/``
    One ``<family>.zip`` per family at the top level. We peek inside the zip
    central directory to count APKs without actually extracting, so the
    inventory is cheap even though the archives are several GB combined.

The script is **read-only** -- it never extracts, hashes the zip contents,
or moves files. It answers "what do we have?" for the `2026-05-05 dataset
recovery report <docs/progress/sessions/2026-05-05_dataset_recovery.md>`_.

Usage::

    python scripts/data/inventory_track_c.py \\
        --wild-dir data/real_world/wild \\
        --out-json outputs/experiments/track_c/inventory.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

APK_EXTS = {".apk"}

# Many wild-malware dumps (notably sk3ptre/) name samples after their MD5/SHA1
# hash with no extension. We sniff the first 4 bytes against the ZIP/APK magic
# number ``PK\x03\x04`` to pick those up too. See docs/workstreams/track_c/
# corpus_schema.md for the rationale.
APK_MAGIC = b"PK\x03\x04"


def _looks_like_apk_by_name(name: str) -> bool:
    return name.lower().endswith(tuple(APK_EXTS))


def _looks_like_apk_in_zip(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    """Check the first 4 bytes of a zip member against the APK magic.

    APKs are themselves zip archives, so the magic is ``PK\x03\x04``.
    We only read 4 bytes per member; for very large archives this still
    keeps the inventory O(n_members * few-bytes) rather than O(total_bytes).
    """
    if info.is_dir() or info.file_size < 4:
        return False
    try:
        with zf.open(info, "r") as fh:
            return fh.read(4) == APK_MAGIC
    except Exception:
        return False


@dataclasses.dataclass
class FamilyRecord:
    """One malware family as it sits on disk today."""

    source: str  # "ashishb" | "sk3ptre"
    family: str  # directory or zip base name
    container: str  # "directory" | "zip"
    path: str  # absolute path to the directory or zip
    apk_count: int
    apk_bytes: int
    sample_apks: List[str]  # up to 3 relative-to-container names
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Source walkers
# ---------------------------------------------------------------------------


def _scan_directory_family(family_dir: Path, source: str) -> FamilyRecord:
    apks: List[Path] = []
    name_only_hits = 0
    magic_only_hits = 0
    for p in sorted(family_dir.rglob("*")):
        if not p.is_file():
            continue
        if _looks_like_apk_by_name(p.name):
            apks.append(p)
            name_only_hits += 1
            continue
        # Sniff magic for hash-named samples in the ashishb tree too.
        try:
            with p.open("rb") as fh:
                head = fh.read(4)
        except Exception:
            continue
        if head == APK_MAGIC:
            apks.append(p)
            magic_only_hits += 1
    total_bytes = sum(p.stat().st_size for p in apks)
    sample = [p.name for p in apks[:3]]
    notes = None
    if magic_only_hits:
        notes = (
            f"{name_only_hits} apk(s) by name + {magic_only_hits} by PK-magic"
        )
    if not apks:
        # Some ashishb dirs ship stripped samples (hex dumps, ida scripts,
        # txt write-ups) instead of APKs. Record so the labeller knows it
        # needs a manual carve step.
        aux = sorted(p for p in family_dir.iterdir() if p.is_file())
        notes = f"no-apk family; {len(aux)} auxiliary files (first={aux[0].name!r})" if aux else "empty"
    return FamilyRecord(
        source=source,
        family=family_dir.name,
        container="directory",
        path=str(family_dir),
        apk_count=len(apks),
        apk_bytes=total_bytes,
        sample_apks=sample,
        notes=notes,
    )


def _scan_zip_family(zip_path: Path, source: str) -> FamilyRecord:
    apks_in_zip: List[zipfile.ZipInfo] = []
    name_only_hits = 0
    magic_only_hits = 0
    encrypted_members = 0
    notes: Optional[str] = None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                by_name = _looks_like_apk_by_name(info.filename)
                if by_name:
                    apks_in_zip.append(info)
                    name_only_hits += 1
                    continue
                # Many malware-sample zips (notably sk3ptre) are password-
                # protected with "infected" per industry convention. Detect
                # that state explicitly; we'll still record the members so the
                # labeller knows they need decryption before sniffing.
                if info.flag_bits & 0x1:  # ZIP encryption flag
                    encrypted_members += 1
                    continue
                # Only run the magic-byte sniff on candidates where the name
                # gives no hint: hash-like names, no extension, or generic
                # extensions like .bin.
                if _looks_like_apk_in_zip(zf, info):
                    apks_in_zip.append(info)
                    magic_only_hits += 1
    except zipfile.BadZipFile as exc:
        notes = f"BAD ZIP: {exc}"
    except Exception as exc:  # pragma: no cover (defensive)
        notes = f"zip scan failed: {exc}"
    total_bytes = sum(info.file_size for info in apks_in_zip)
    sample = [info.filename for info in apks_in_zip[:3]]
    # Build a notes string summarising how the apk count was determined.
    breadcrumbs = []
    if name_only_hits:
        breadcrumbs.append(f"{name_only_hits} apk(s) by name")
    if magic_only_hits:
        breadcrumbs.append(f"{magic_only_hits} by PK-magic")
    if encrypted_members:
        breadcrumbs.append(f"{encrypted_members} encrypted member(s) unclassified (password-protected zip; common convention: 'infected')")
    if not notes and breadcrumbs:
        notes = "; ".join(breadcrumbs)
    return FamilyRecord(
        source=source,
        family=zip_path.stem,
        container="zip",
        path=str(zip_path),
        apk_count=len(apks_in_zip),
        apk_bytes=total_bytes,
        sample_apks=sample,
        notes=notes,
    )


def _iter_ashishb(source_root: Path, source_label: str) -> Iterable[FamilyRecord]:
    if not source_root.exists():
        return
    for entry in sorted(source_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):  # skip .github etc.
            continue
        yield _scan_directory_family(entry, source_label)


def _iter_sk3ptre(source_root: Path, source_label: str) -> Iterable[FamilyRecord]:
    if not source_root.exists():
        return
    for entry in sorted(source_root.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".zip":
            yield _scan_zip_family(entry, source_label)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_inventory(wild_dir: Path) -> dict:
    records: List[FamilyRecord] = []
    records.extend(_iter_ashishb(wild_dir / "ashishb_android_malware", "ashishb"))
    records.extend(_iter_sk3ptre(wild_dir / "sk3ptre_AndroidMalware_2020", "sk3ptre"))

    # Summaries
    def _sum(src: str, cond=lambda r: True) -> tuple[int, int, int]:
        subset = [r for r in records if r.source == src and cond(r)]
        return (
            len(subset),
            sum(r.apk_count for r in subset),
            sum(r.apk_bytes for r in subset),
        )

    def _encrypted_member_count(records_: Iterable[FamilyRecord]) -> int:
        # Parse the "N encrypted member(s) ..." breadcrumb we embed in .notes.
        import re
        pat = re.compile(r"(\d+) encrypted member\(s\)")
        total = 0
        for r in records_:
            if r.notes:
                m = pat.search(r.notes)
                if m:
                    total += int(m.group(1))
        return total

    ash_fams, ash_apks, ash_bytes = _sum("ashishb")
    ash_empty_fams = _sum("ashishb", lambda r: r.apk_count == 0)[0]
    sk_fams, sk_apks, sk_bytes = _sum("sk3ptre")
    sk_bad_fams = _sum("sk3ptre", lambda r: (r.notes or "").startswith("BAD ZIP"))[0]
    sk_encrypted_members = _encrypted_member_count(r for r in records if r.source == "sk3ptre")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wild_dir": str(wild_dir),
        "summary": {
            "ashishb": {
                "family_count": ash_fams,
                "families_with_no_apk": ash_empty_fams,
                "apk_count": ash_apks,
                "apk_bytes": ash_bytes,
            },
            "sk3ptre": {
                "family_count": sk_fams,
                "bad_zips": sk_bad_fams,
                "apk_count": sk_apks,
                "apk_bytes": sk_bytes,
                "encrypted_members_unclassified": sk_encrypted_members,
                "note": (
                    "APK counts come from zip central-directory listings; archives are NOT extracted. "
                    "sk3ptre zips are password-protected per industry convention (password 'infected'); "
                    "encrypted members cannot be magic-sniffed and are reported separately."
                ),
            },
            "total_families": ash_fams + sk_fams,
            "total_apks_confirmed": ash_apks + sk_apks,
            "total_apks_pending_decryption": sk_encrypted_members,
            "total_apk_bytes": ash_bytes + sk_bytes,
        },
        "families": [r.to_dict() for r in records],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip())
    repo_root = Path(__file__).resolve().parents[2]
    default_wild = repo_root / "data" / "real_world" / "wild"
    default_out = repo_root / "outputs" / "experiments" / "track_c" / "inventory.json"
    p.add_argument("--wild-dir", type=Path, default=default_wild,
                   help=f"Track C raw dumps root (default: {default_wild})")
    p.add_argument("--out-json", type=Path, default=default_out,
                   help=f"where to write the inventory JSON (default: {default_out})")
    p.add_argument("--print-summary", action="store_true",
                   help="also pretty-print the summary block to stdout")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    wild_dir: Path = args.wild_dir
    if not wild_dir.exists():
        print(f"[FATAL] wild dir missing: {wild_dir}", file=sys.stderr)
        return 2
    inv = build_inventory(wild_dir)
    out: Path = args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(inv, fh, indent=2, ensure_ascii=False)
    print(f"[OK] wrote inventory: {out}")
    print(f"     families: ashishb={inv['summary']['ashishb']['family_count']}  "
          f"sk3ptre={inv['summary']['sk3ptre']['family_count']}  "
          f"total={inv['summary']['total_families']}")
    confirmed = inv['summary']['total_apks_confirmed']
    pending = inv['summary']['total_apks_pending_decryption']
    print(f"     apks   : confirmed={confirmed}  pending-decryption={pending}  "
          f"(~{inv['summary']['total_apk_bytes']/1024/1024:.1f} MB on-disk)")
    if args.print_summary:
        print(json.dumps(inv["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
