"""Track C wild-malware labelling pipeline (3 stages, idempotent).

Consumes ``data/real_world/wild/`` (see ``docs/workstreams/track_c/
corpus_schema.md``) and produces ``outputs/experiments/track_c/
labels.jsonl`` -- one JSON line per unique APK (deduped by sha256),
conforming to the Path-C-lite schema v1 documented in that file.

Three stages, each independently re-runnable:

* **Stage 1 -- materialise & hash.** Walks both sources, hashes every
  APK (identified by ``.apk`` extension OR by ``PK\\x03\\x04`` magic
  for hash-named samples), hard-copies / symlinks unique samples into
  ``data/real_world/track_c/samples/<sha256[:2]>/<sha256>.apk``. Writes
  the ``schema_version / track / source / family / sample_id /
  sample_path / provenance`` fields; the ``labels`` sub-dict is left
  entirely null. Supports ``--include-encrypted`` to opt in to the
  sk3ptre password-protected zips (password defaults to ``infected``).

* **Stage 2 -- apkid packer sniffer.** For every JSONL row written by
  Stage 1, invokes ``apkid_cross_check.cross_check_apk`` with
  ``expected_family=None`` (we never know the true family for wild
  samples). Fills ``labels.is_packed_probed`` + ``labels.suspected_packer``
  from the apkid hits, and also records the raw apkid detected-families
  list in a side ``apkid_debug`` field for reviewer auditability.

* **Stage 3 -- structural probes.** One zip walk per sample fills
  ``labels.has_native_libs`` (any ``lib/<abi>/*.so``) and
  ``labels.has_assets_dex`` (any ``assets/**/*.dex``). These are cheap
  signals -- Gen-2 whole-DEX packers often drop the encrypted classes.dex
  into ``assets/`` while native-stub-based packers (e.g. CS1 * 360-Jiagu)
  ship a ``libjiagu.so`` fingerprint under ``lib/``.

Usage::

    # Dry-run: print what would happen, do not touch disk.
    python scripts/data/label_track_c.py --stage 1 --dry-run

    # Stage 1 on ashishb only, default output path:
    python scripts/data/label_track_c.py --stage 1 --execute --sources ashishb

    # Stage 1 for everything including sk3ptre decrypted:
    python scripts/data/label_track_c.py --stage 1 --execute --include-encrypted

    # Stage 2 + 3 after Stage 1 has produced rows:
    python scripts/data/label_track_c.py --stage 2 --execute
    python scripts/data/label_track_c.py --stage 3 --execute

    # All three at once:
    python scripts/data/label_track_c.py --stage all --execute
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


APK_MAGIC = b"PK\x03\x04"
SK3PTRE_DEFAULT_PASSWORD = b"infected"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _blank_label_record(
    *,
    source: str,
    family: str,
    sample_id: str,
    sample_path: Path,
    upstream_note_path: Optional[Path],
    container: str,
    container_path: Path,
    decryption_password: Optional[str],
    fetched_from: str,
) -> Dict[str, Any]:
    """Produce one JSONL row in Path-C-lite v1 shape."""
    return {
        "schema_version": 1,
        "track": "C",
        "source": source,
        "family": family,
        "sample_id": sample_id,
        "sample_path": str(sample_path),
        "upstream_note_path": str(upstream_note_path) if upstream_note_path else None,
        "labels": {
            "family_coarse": family,
            "is_packed_probed": None,
            "suspected_packer": None,
            "has_native_libs": None,
            "has_assets_dex": None,
            "dex_payload_hint_offset": None,
        },
        "provenance": {
            "fetched_from": fetched_from,
            "container": container,
            "container_path": str(container_path),
            "decryption_password": decryption_password,
        },
    }


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _looks_like_apk(path: Path) -> bool:
    """True if ``path.suffix == '.apk'`` OR the first 4 bytes are PK magic."""
    if path.suffix.lower() == ".apk":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(4) == APK_MAGIC
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage 1 -- materialise & hash
# ---------------------------------------------------------------------------


def _iter_ashishb_apks(ashishb_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield ``(family, apk_path)`` for every real APK in the ashishb tree.

    ``family`` is the name of the immediate child directory of ``ashishb_root``
    containing the APK. Nested structure beneath the family dir is flattened.
    """
    if not ashishb_root.exists():
        return
    for family_dir in sorted(p for p in ashishb_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for candidate in sorted(family_dir.rglob("*")):
            if not candidate.is_file():
                continue
            if _looks_like_apk(candidate):
                yield family_dir.name, candidate


def _stage1_sk3ptre_extract_one(
    zip_path: Path,
    samples_pool: Path,
    *,
    password: bytes,
    dry_run: bool,
) -> Iterable[Tuple[str, str, Path, Path]]:
    """Iterate apk-like members of one sk3ptre zip.

    Yields ``(family, sample_id, extracted_path, origin_info_path_like)``.
    For encrypted members we decrypt on the fly, sha256-hash in-memory,
    then write to the samples pool only if the sample is new.
    """
    family = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        zf.setpassword(password)
        for info in zf.infolist():
            if info.is_dir() or info.file_size < 4:
                continue
            # Read first 4 bytes for magic check; if member is encrypted,
            # this tries the password -- bad password -> RuntimeError.
            try:
                with zf.open(info, "r") as fh:
                    head = fh.read(4)
                    if head != APK_MAGIC and not info.filename.lower().endswith(".apk"):
                        # Not an APK; skip silently.
                        continue
                    # Re-open and read the full payload to compute sha256.
                    rest = fh.read()
                    payload = head + rest
            except RuntimeError as exc:
                # Two failure modes surface as RuntimeError from the stdlib
                # ``zipfile`` module:
                #   (a) "Bad password for file" -> password mismatch
                #   (b) "That compression method is not supported" -> the zip
                #       uses LZMA / bzip2 / deflate64 / stored-encrypted which
                #       the stdlib can't open. Requires ``pyzipper`` or 7z.
                # The two need different remediation, so we surface the raw
                # message rather than a misleading "wrong password?" catch-all.
                raise RuntimeError(
                    f"sk3ptre zip {zip_path.name} member {info.filename!r}: "
                    f"{exc}"
                ) from exc
            sample_id = _sha256_bytes(payload)
            target = samples_pool / sample_id[:2] / f"{sample_id}.apk"
            if not dry_run and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            yield family, sample_id, target, zip_path


def _stage1_ashishb_materialise_one(
    apk_path: Path,
    samples_pool: Path,
    *,
    dry_run: bool,
) -> Tuple[str, Path]:
    """Hash one ashishb APK and (non-destructively) copy it into the pool.

    Returns ``(sample_id, pooled_path)``. Idempotent: if a file with the
    matching sha256 already lives in the pool, no copy happens.
    """
    sample_id = _sha256_file(apk_path)
    target = samples_pool / sample_id[:2] / f"{sample_id}.apk"
    if not dry_run and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Prefer hardlink on the same volume to save disk; fall back to copy.
        try:
            os.link(apk_path, target)
        except OSError:
            shutil.copy2(apk_path, target)
    return sample_id, target


def run_stage1_materialise(
    wild_dir: Path,
    samples_pool: Path,
    labels_jsonl: Path,
    *,
    sources: Tuple[str, ...],
    include_encrypted: bool,
    sk3ptre_password: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Stage 1 entry. Idempotent; rewrites ``labels_jsonl`` atomically.

    Existing Stage 2 / Stage 3 enrichments on rows that are still in the
    (source, sample_id) set are *preserved* from the old file so a re-run
    of Stage 1 does not wipe downstream work.
    """
    prior = _load_existing_rows(labels_jsonl)

    seen: Dict[str, Dict[str, Any]] = {}
    skipped_duplicate = 0
    sk3ptre_skipped_reasons: Dict[str, int] = {}

    if "ashishb" in sources:
        ashishb_root = wild_dir / "ashishb_android_malware"
        for family, apk_path in _iter_ashishb_apks(ashishb_root):
            try:
                sample_id, pooled = _stage1_ashishb_materialise_one(
                    apk_path, samples_pool, dry_run=dry_run
                )
            except Exception as exc:  # pragma: no cover
                sk3ptre_skipped_reasons[f"ashishb/{family}: {type(exc).__name__}"] = \
                    sk3ptre_skipped_reasons.get(f"ashishb/{family}: {type(exc).__name__}", 0) + 1
                continue
            row = _blank_label_record(
                source="ashishb",
                family=family,
                sample_id=sample_id,
                sample_path=pooled,
                upstream_note_path=None,
                container="directory",
                container_path=apk_path,
                decryption_password=None,
                fetched_from="https://github.com/ashishb/android-malware",
            )
            if sample_id in seen:
                skipped_duplicate += 1
                continue
            seen[sample_id] = _merge_with_prior(row, prior.get(sample_id))

    if "sk3ptre" in sources and include_encrypted:
        sk_root = wild_dir / "sk3ptre_AndroidMalware_2020"
        if sk_root.exists():
            password_bytes = sk3ptre_password.encode("ascii")
            zip_list = sorted(sk_root.glob("*.zip"))
            for idx, zip_path in enumerate(zip_list, start=1):
                # Progress heartbeat so long runs don't look frozen.
                print(
                    f"[stage1:sk3ptre] {idx:3d}/{len(zip_list)}  {zip_path.name}",
                    file=sys.stderr, flush=True,
                )
                try:
                    for family, sample_id, pooled, container in _stage1_sk3ptre_extract_one(
                        zip_path, samples_pool,
                        password=password_bytes,
                        dry_run=dry_run,
                    ):
                        row = _blank_label_record(
                            source="sk3ptre",
                            family=family,
                            sample_id=sample_id,
                            sample_path=pooled,
                            upstream_note_path=None,
                            container="zip",
                            container_path=container,
                            decryption_password=sk3ptre_password,
                            fetched_from="https://github.com/sk3ptre/AndroidMalware_2020",
                        )
                        if sample_id in seen:
                            skipped_duplicate += 1
                            continue
                        seen[sample_id] = _merge_with_prior(row, prior.get(sample_id))
                except RuntimeError as exc:
                    key = f"sk3ptre/{zip_path.stem}: decrypt"
                    sk3ptre_skipped_reasons[key] = sk3ptre_skipped_reasons.get(key, 0) + 1
                    print(f"[WARN] {exc}", file=sys.stderr, flush=True)
    elif "sk3ptre" in sources and not include_encrypted:
        # User listed sk3ptre but forgot the opt-in flag. Be explicit.
        sk3ptre_skipped_reasons["sk3ptre: skipped (need --include-encrypted)"] = 1

    rows = sorted(seen.values(), key=lambda r: (r["source"], r["family"], r["sample_id"]))
    if not dry_run:
        _atomic_write_jsonl(labels_jsonl, rows)

    return {
        "stage": 1,
        "dry_run": dry_run,
        "labels_jsonl": str(labels_jsonl),
        "samples_pool": str(samples_pool),
        "counts": {
            "unique_samples": len(rows),
            "by_source": {
                s: sum(1 for r in rows if r["source"] == s)
                for s in sorted({r["source"] for r in rows})
            },
            "skipped_duplicate_by_sha256": skipped_duplicate,
        },
        "warnings": sk3ptre_skipped_reasons,
    }


# ---------------------------------------------------------------------------
# Stage 2 -- apkid packer sniffer
# ---------------------------------------------------------------------------


def run_stage2_apkid(
    labels_jsonl: Path,
    *,
    apkid_cmd: str,
    timeout: float,
    limit: Optional[int],
    dry_run: bool,
) -> Dict[str, Any]:
    """Stage 2 entry. Fills labels.is_packed_probed / suspected_packer.

    Reuses :func:`android_packer.labeling.apkid_cross_check.cross_check_apk`
    with ``expected_family=None`` so wild samples are evaluated as "no a
    priori expectation" -- any hit becomes ``APKID_FALSE_POSITIVE`` in
    agreement terms, which for Track C semantically means "packer
    detected". We translate that back into our schema.
    """
    # Lazy import so Stage 1-only runs don't pay the dependency cost.
    from android_packer.labeling.apkid_cross_check import (
        cross_check_apk as _apkid_cross_check_apk,
        load_apkid_family_map,
    )

    all_rows = _load_ordered_rows(labels_jsonl)
    # ``--limit`` scopes which rows we process this invocation, but the
    # file we write back must contain EVERY row (processed + unchanged) or
    # a limited smoke run would truncate the corpus to ``limit`` rows.
    rows_to_process = all_rows if limit is None else all_rows[:limit]

    repo_root = Path(__file__).resolve().parents[2]
    family_map_path = repo_root / "configs" / "data" / "apkid_family_map.yaml"
    family_map = load_apkid_family_map(family_map_path) if family_map_path.exists() else None

    updated = 0
    apkid_failed = 0
    apkid_debug_list: List[Dict[str, Any]] = []

    total = len(rows_to_process)
    for idx, row in enumerate(rows_to_process, start=1):
        # Heartbeat every 20 samples so detached long runs don't look frozen.
        if idx == 1 or idx % 20 == 0 or idx == total:
            print(
                f"[stage2:apkid] {idx:4d}/{total}  {row.get('family','?')}/{row.get('sample_id','?')[:12]}",
                file=sys.stderr, flush=True,
            )
        sample_path = Path(row["sample_path"])
        if not sample_path.exists():
            row.setdefault("labels", {})["is_packed_probed"] = None
            row.setdefault("apkid_debug", {})["error"] = "sample_path missing"
            apkid_failed += 1
            continue
        if dry_run:
            continue
        report = _apkid_cross_check_apk(
            sample_path,
            apk_id=row["sample_id"],
            expected_family=None,
            family_map=family_map,
            apkid_cmd=apkid_cmd,
            timeout=timeout,
            graceful=True,
        )
        has_hit = report.has_packer_hit or report.has_protector_hit
        apkid_error = report.apkid_result.error if report.apkid_result else None
        if apkid_error:
            # Transport-level failure: record but don't claim a signal.
            row.setdefault("labels", {})["is_packed_probed"] = None
            row.setdefault("apkid_debug", {})["error"] = apkid_error
            apkid_failed += 1
        else:
            row.setdefault("labels", {})["is_packed_probed"] = bool(has_hit)
            # suspected_packer: mapped family names if any, else the raw hit
            # strings so a reviewer can grep.
            if report.detected_families:
                row["labels"]["suspected_packer"] = list(report.detected_families)
            elif has_hit:
                raw = sorted({
                    m.hit for m in report.apkid_result.packer_like_matches()
                })
                row["labels"]["suspected_packer"] = raw
            else:
                row["labels"]["suspected_packer"] = []
            row["apkid_debug"] = {
                "apkid_version": report.apkid_result.apkid_version,
                "agreement": report.agreement,
                "detected_families": list(report.detected_families),
                "has_packer_hit": report.has_packer_hit,
                "has_protector_hit": report.has_protector_hit,
                "notes": list(report.notes),
            }
            updated += 1
        apkid_debug_list.append(row.get("apkid_debug", {}))

    if not dry_run:
        # Write EVERY row back, not just the processed ones; ``all_rows``
        # still holds objects that were mutated in-place for the subset
        # we processed, so the on-disk file keeps Stage-1 provenance for
        # untouched rows while gaining Stage-2 enrichments for processed ones.
        _atomic_write_jsonl(labels_jsonl, all_rows)

    return {
        "stage": 2,
        "dry_run": dry_run,
        "labels_jsonl": str(labels_jsonl),
        "counts": {
            "rows_total": len(all_rows),
            "rows_processed_this_run": len(rows_to_process),
            "apkid_ran_ok": updated,
            "apkid_failed": apkid_failed,
            "suspected_packed": sum(
                1 for r in all_rows if r.get("labels", {}).get("is_packed_probed") is True
            ),
            "suspected_clean": sum(
                1 for r in all_rows if r.get("labels", {}).get("is_packed_probed") is False
            ),
        },
    }


# ---------------------------------------------------------------------------
# Stage 3 -- structural probes
# ---------------------------------------------------------------------------


def _structural_probes(apk_path: Path) -> Tuple[Optional[bool], Optional[bool]]:
    """Return ``(has_native_libs, has_assets_dex)`` for one APK.

    A single zip walk. On parse failure returns ``(None, None)`` so the
    caller can tell "not probed" from "probed but clean".
    """
    try:
        with zipfile.ZipFile(apk_path) as zf:
            has_native = False
            has_assets_dex = False
            for name in zf.namelist():
                lower = name.lower()
                # lib/<abi>/*.so -- note APKs carry arm64-v8a / armeabi-v7a / x86 / x86_64.
                if lower.startswith("lib/") and lower.endswith(".so"):
                    has_native = True
                # assets/**/*.dex (or .jar with dex inside -- not probed here).
                if lower.startswith("assets/") and lower.endswith(".dex"):
                    has_assets_dex = True
                if has_native and has_assets_dex:
                    break
            return has_native, has_assets_dex
    except Exception:
        return None, None


def run_stage3_structural(
    labels_jsonl: Path,
    *,
    limit: Optional[int],
    dry_run: bool,
) -> Dict[str, Any]:
    """Stage 3 entry. Fills labels.has_native_libs / has_assets_dex."""
    all_rows = _load_ordered_rows(labels_jsonl)
    rows_to_process = all_rows if limit is None else all_rows[:limit]

    updated = 0
    total = len(rows_to_process)
    for idx, row in enumerate(rows_to_process, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(
                f"[stage3:struct] {idx:4d}/{total}  {row.get('family','?')}/{row.get('sample_id','?')[:12]}",
                file=sys.stderr, flush=True,
            )
        sample_path = Path(row["sample_path"])
        if not sample_path.exists():
            continue
        if dry_run:
            continue
        has_native, has_assets_dex = _structural_probes(sample_path)
        labels = row.setdefault("labels", {})
        labels["has_native_libs"] = has_native
        labels["has_assets_dex"] = has_assets_dex
        updated += 1

    if not dry_run:
        _atomic_write_jsonl(labels_jsonl, all_rows)

    return {
        "stage": 3,
        "dry_run": dry_run,
        "labels_jsonl": str(labels_jsonl),
        "counts": {
            "rows_total": len(all_rows),
            "rows_processed_this_run": len(rows_to_process),
            "probed_ok": updated,
            "has_native_libs": sum(
                1 for r in all_rows if r.get("labels", {}).get("has_native_libs") is True
            ),
            "has_assets_dex": sum(
                1 for r in all_rows if r.get("labels", {}).get("has_assets_dex") is True
            ),
        },
    }


# ---------------------------------------------------------------------------
# JSONL atomic read / write helpers
# ---------------------------------------------------------------------------


def _load_existing_rows(labels_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing rows keyed by ``sample_id``; empty if file missing."""
    out: Dict[str, Dict[str, Any]] = {}
    if not labels_jsonl.exists():
        return out
    with labels_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if sid:
                out[sid] = obj
    return out


def _load_ordered_rows(labels_jsonl: Path) -> List[Dict[str, Any]]:
    """Read existing rows as a list preserving file order."""
    if not labels_jsonl.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with labels_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _merge_with_prior(fresh: Dict[str, Any], prior: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """When Stage 1 re-runs, preserve Stage 2 / Stage 3 enrichments.

    We overwrite ``schema_version``, ``source``, ``sample_path`` (pool
    may have moved), ``provenance`` (fresh wins) but keep any existing
    non-null ``labels.*`` and ``apkid_debug`` entries.
    """
    if prior is None:
        return fresh
    merged = dict(fresh)
    # Preserve labels that were filled in by later stages.
    prior_labels = (prior.get("labels") or {})
    for k, v in prior_labels.items():
        if v is not None and fresh["labels"].get(k) is None:
            merged["labels"][k] = v
    if "apkid_debug" in prior:
        merged["apkid_debug"] = prior["apkid_debug"]
    return merged


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_wild = repo_root / "data" / "real_world" / "wild"
    default_pool = repo_root / "data" / "real_world" / "track_c" / "samples"
    default_jsonl = repo_root / "outputs" / "experiments" / "track_c" / "labels.jsonl"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip())
    p.add_argument("--stage", choices=["1", "2", "3", "all"], default="all",
                   help="Which pipeline stage to run (default: all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk + plan; do not touch disk (no copies, no zip extracts, no apkid invocations).")
    p.add_argument("--execute", action="store_true",
                   help="Required to actually perform disk writes / subprocess calls. "
                        "Without --execute the tool runs in dry-run mode.")
    p.add_argument("--wild-dir", type=Path, default=default_wild,
                   help=f"Track C raw dumps root (default: {default_wild}).")
    p.add_argument("--samples-pool", type=Path, default=default_pool,
                   help=f"Deduped sample pool (default: {default_pool}).")
    p.add_argument("--labels-jsonl", type=Path, default=default_jsonl,
                   help=f"Per-sample JSONL output (default: {default_jsonl}).")
    p.add_argument("--sources", default="ashishb",
                   help="Comma-separated subset of {ashishb,sk3ptre}. Default: 'ashishb'.")
    p.add_argument("--include-encrypted", action="store_true",
                   help="Opt in to sk3ptre password-protected zip extraction.")
    p.add_argument("--sk3ptre-password", default="infected",
                   help="Password for sk3ptre zips (default: 'infected').")
    p.add_argument("--apkid-cmd", default="apkid",
                   help="apkid binary name (default: 'apkid'; override for tests).")
    p.add_argument("--apkid-timeout", type=float, default=120.0,
                   help="Per-sample apkid timeout in seconds (default: 120).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stage 2 / 3 only: cap the number of samples processed "
                        "(useful for smoke tests).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # ``--dry-run`` is the default unless ``--execute`` is explicitly set.
    dry_run = not args.execute or args.dry_run

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    for s in sources:
        if s not in {"ashishb", "sk3ptre"}:
            print(f"[FATAL] unknown --sources entry: {s!r} (accepted: ashishb, sk3ptre)", file=sys.stderr)
            return 2

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary: Dict[str, Any] = {
        "started_at": started_at,
        "stage": args.stage,
        "dry_run": dry_run,
        "wild_dir": str(args.wild_dir),
        "samples_pool": str(args.samples_pool),
        "labels_jsonl": str(args.labels_jsonl),
        "sources": list(sources),
        "include_encrypted": args.include_encrypted,
        "stages": [],
    }

    try:
        if args.stage in ("1", "all"):
            summary["stages"].append(run_stage1_materialise(
                args.wild_dir, args.samples_pool, args.labels_jsonl,
                sources=sources,
                include_encrypted=args.include_encrypted,
                sk3ptre_password=args.sk3ptre_password,
                dry_run=dry_run,
            ))
        if args.stage in ("2", "all"):
            summary["stages"].append(run_stage2_apkid(
                args.labels_jsonl,
                apkid_cmd=args.apkid_cmd,
                timeout=args.apkid_timeout,
                limit=args.limit,
                dry_run=dry_run,
            ))
        if args.stage in ("3", "all"):
            summary["stages"].append(run_stage3_structural(
                args.labels_jsonl,
                limit=args.limit,
                dry_run=dry_run,
            ))
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    summary["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
