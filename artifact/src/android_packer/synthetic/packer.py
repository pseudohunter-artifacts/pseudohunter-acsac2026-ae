"""Synthetic APK generation for DEX-only payload localization.

The heavy lifting is split between this module (orchestration, seed APK I/O,
manifest / label emission) and :mod:`android_packer.synthetic.transforms`
(the transform registry that knows how to turn a payload into one or more
``InjectedPayload`` objects).
"""

from __future__ import annotations

import json
import posixpath
import random
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Optional, Sequence

from android_packer.apkio import ApkObject, iter_apk_objects
from android_packer.apkio.objects import file_sha256
from android_packer.labeling import HIDDEN_EXECUTABLE_PAYLOAD, SyntheticLabel
from android_packer.synthetic.records import (
    InjectedPayload,
    PayloadSource,
    SyntheticBuildResult,
    SyntheticPackerError,
)
from android_packer.synthetic.transforms import (
    SUPPORTED_TRANSFORMS,
    TRANSFORMS,
    SeedNamingProfile,
    TransformContext,
    build_injected_payloads,
    normalize_asset_prefix,
)
from android_packer.utils.jsonl import write_jsonl


def build_synthetic_apk(
    *,
    seed_apk: Path,
    generated_apk_out: Path,
    manifest_out: Optional[Path] = None,
    labels_out: Optional[Path] = None,
    payload_path: Optional[Path] = None,
    transform_family: str = "xor",
    rng_seed: int = 0,
    asset_prefix: str = "assets/synthetic",
    xor_key: Optional[int] = None,
    split_count: int = 2,
    enforce_payload_size_range: bool = True,
) -> SyntheticBuildResult:
    """Create a synthetic packed APK and its strong labels.

    The generated APK is a copy of ``seed_apk`` with one or more additional
    asset-like objects that contain transformed DEX payload bytes. Labels point
    to byte ranges inside those injected APK objects.

    B1 (2026-04-29): when ``enforce_payload_size_range`` is True (the
    production default), each transform rejects payloads smaller than its
    per-family lower bound (see
    :data:`android_packer.synthetic.transforms._PAYLOAD_SIZE_WHOLE_OBJECT`
    / ``_PAYLOAD_SIZE_SUB_RANGE``). Unit-test fixtures that rely on tiny
    placeholder DEXes pass ``enforce_payload_size_range=False`` to keep
    the legacy fast-path. Production CLI keeps the default.
    """

    seed_apk = Path(seed_apk)
    generated_apk_out = Path(generated_apk_out)
    if seed_apk.resolve() == generated_apk_out.resolve():
        raise ValueError("generated_apk_out must not overwrite seed_apk")
    # ``TRANSFORMS`` is the live registry; ``SUPPORTED_TRANSFORMS`` is a
    # snapshot taken at import time that can lag after ``register_transform``.
    if transform_family not in TRANSFORMS:
        supported = ", ".join(sorted(TRANSFORMS))
        raise ValueError(f"unsupported transform_family={transform_family!r}; use {supported}")

    seed_apk_id = file_sha256(seed_apk)
    # ``_prepare_seed_state`` performs a *single* pass over the seed ZIP and
    # returns the payload, the existing member paths, and a pool of host
    # candidates for sub-range transforms.
    payload_source, existing_paths, seed_objects = _prepare_seed_state(
        seed_apk, payload_path
    )
    if not payload_source.data:
        raise SyntheticPackerError("payload bytes are empty")

    # A-v3 leakage fix (2026-05-07): build a SeedNamingProfile from the
    # seed APK so synthetic injection paths and ZipInfo metadata mimic
    # native distributions. Without this profile, ``unique_payload_path``
    # falls back to a legacy ``<asset_prefix>/<token>.bin`` allocator that
    # remains usable from unit tests with no seed APK.
    naming_profile = build_seed_naming_profile(seed_apk)

    rng = random.Random(rng_seed)
    ctx = TransformContext(
        payload=payload_source.data,
        rng=rng,
        existing_paths=existing_paths,
        asset_prefix=asset_prefix,
        xor_key=xor_key,
        split_count=split_count,
        seed_objects=seed_objects,
        enforce_payload_size_range=enforce_payload_size_range,
        naming_profile=naming_profile,
    )
    injected_payloads = build_injected_payloads(transform_family, ctx)

    _write_augmented_apk(
        seed_apk,
        generated_apk_out,
        injected_payloads,
        naming_profile=naming_profile,
        rng=rng,
    )
    generated_apk_id = file_sha256(generated_apk_out)
    payload_sha = sha256(payload_source.data).hexdigest()

    labels = [
        SyntheticLabel(
            apk_id=generated_apk_id,
            object_path=injected.object_path,
            # Sub-range transforms (``host_object_path is not None``) place
            # the payload at ``[payload_offset_start, payload_offset_end)``
            # inside the host object's full byte stream (which is stored at
            # ``data``). Whole-object transforms leave the payload filling
            # the entire member, so the range degenerates to ``0..len(data)``.
            offset_start=injected.payload_offset_start if injected.host_object_path else 0,
            offset_end=injected.payload_offset_end if injected.host_object_path else len(injected.data),
            label=HIDDEN_EXECUTABLE_PAYLOAD,
            transform_family=injected.transform_family,
            payload_sha256=payload_sha,
            source_apk_id=payload_source.source_apk_id,
            source_object_path=payload_source.source_object_path,
            transformed_sha256=sha256(injected.data).hexdigest(),
            part_index=injected.part_index,
            part_count=injected.part_count,
            source_offset_start=injected.payload_offset_start,
            source_offset_end=injected.payload_offset_end,
        )
        for injected in injected_payloads
    ]
    manifest = _build_manifest(
        seed_apk=seed_apk,
        seed_apk_id=seed_apk_id,
        generated_apk_out=generated_apk_out,
        generated_apk_id=generated_apk_id,
        payload_source=payload_source,
        transform_family=transform_family,
        rng_seed=rng_seed,
        asset_prefix=asset_prefix,
        split_count=split_count,
        injected_payloads=injected_payloads,
        labels_out=labels_out,
        payload_sha=payload_sha,
    )

    if labels_out is not None:
        write_jsonl(Path(labels_out), [label.to_dict() for label in labels])
    if manifest_out is not None:
        _write_json(Path(manifest_out), manifest)

    return SyntheticBuildResult(
        generated_apk_path=generated_apk_out,
        manifest_path=Path(manifest_out) if manifest_out is not None else None,
        labels_path=Path(labels_out) if labels_out is not None else None,
        manifest=manifest,
        labels=labels,
    )


def _prepare_seed_state(
    seed_apk: Path,
    payload_path: Optional[Path],
) -> tuple[PayloadSource, set[str], list[tuple[ApkObject, bytes]]]:
    """Open the seed APK once and collect everything we need from it.

    Returns ``(payload_source, existing_paths, seed_objects)``:

    - ``payload_source`` describes the DEX/file we will transform.
    - ``existing_paths`` is the set of ZIP member paths already present in the
      seed APK, used to allocate collision-free injection paths.
    - ``seed_objects`` is the full list of top-level ``(ApkObject, bytes)``
      tuples from the seed, used by sub-range transforms (``embedded_asset``,
      ``so_embedded``, ``dex_method_inlined``) to select a host object in
      which to embed the payload. The list preserves the original ZIP
      iteration order so transform outputs are deterministic given
      ``rng_seed``.
    """

    if payload_path is not None:
        payload_path = Path(payload_path)
        payload_source = PayloadSource(
            kind="external_file",
            data=payload_path.read_bytes(),
            source_apk_id=None,
            source_object_path=str(payload_path),
        )
        # We still need ``seed_objects`` for sub-range transforms even when
        # ``payload_path`` is externally provided. Walking the ZIP once here
        # is cheaper than re-opening it inside each sub-range transform.
        existing_paths: set[str] = set()
        seed_objects: list[tuple[ApkObject, bytes]] = []
        for metadata, data in iter_apk_objects(seed_apk, max_depth=0):
            existing_paths.add(metadata.object_path)
            seed_objects.append((metadata, data))
        return payload_source, existing_paths, seed_objects

    # Walk the ZIP once: pick a suitable DEX payload *and* collect the member
    # path set *and* the seed object pool from the same
    # ``iter_apk_objects`` iteration.
    dex_candidates: list[tuple[ApkObject, bytes]] = []
    existing_paths = set()
    seed_objects = []
    for metadata, data in iter_apk_objects(seed_apk, max_depth=0):
        existing_paths.add(metadata.object_path)
        seed_objects.append((metadata, data))
        if metadata.object_type == "dex":
            dex_candidates.append((metadata, data))

    if not dex_candidates:
        raise SyntheticPackerError(
            "seed APK has no top-level DEX object; pass --payload explicitly"
        )

    # Prefer ``classes2.dex`` etc. over ``classes.dex`` so synthetic payloads
    # don't collide with the primary DEX a runtime is most likely to load.
    secondary = [
        item
        for item in dex_candidates
        if posixpath.basename(item[0].object_path.lower()) != "classes.dex"
    ]
    selected = (secondary or dex_candidates)[0]
    payload_source = PayloadSource(
        kind="seed_apk_dex",
        data=selected[1],
        source_apk_id=selected[0].apk_id,
        source_object_path=selected[0].object_path,
    )
    return payload_source, existing_paths, seed_objects


def _load_payload_source(seed_apk: Path, payload_path: Optional[Path]) -> PayloadSource:
    """Backwards-compatible wrapper kept for external callers.

    ``build_synthetic_apk`` itself now calls :func:`_prepare_seed_state` to
    avoid re-opening the seed APK.
    """

    payload_source, _, _ = _prepare_seed_state(seed_apk, payload_path)
    return payload_source


def _select_default_dex_payload(seed_apk: Path) -> tuple[ApkObject, bytes]:
    """Deprecated: kept so any external caller continues to work.

    Prefer :func:`_prepare_seed_state` which shares the single ZIP pass with
    ``build_synthetic_apk``.
    """

    dex_objects = [
        (metadata, data)
        for metadata, data in iter_apk_objects(seed_apk, max_depth=0)
        if metadata.object_type == "dex"
    ]
    if not dex_objects:
        raise SyntheticPackerError(
            "seed APK has no top-level DEX object; pass --payload explicitly"
        )

    secondary = [
        item
        for item in dex_objects
        if posixpath.basename(item[0].object_path.lower()) != "classes.dex"
    ]
    return (secondary or dex_objects)[0]


def _write_augmented_apk(
    seed_apk: Path,
    generated_apk_out: Path,
    injected_payloads: Sequence[InjectedPayload],
    *,
    naming_profile: Optional[SeedNamingProfile] = None,
    rng: Optional[random.Random] = None,
) -> None:
    generated_apk_out.parent.mkdir(parents=True, exist_ok=True)
    # Partition injections into two classes:
    #   - ``overwrites``: sub-range transforms that replace an existing member
    #     (host_object_path is set). Keyed by the host path for O(1) lookup.
    #     Multiple InjectedPayload records per host are supported (B2's
    #     dex_method_inlined emits k segments, all pointing at the same
    #     host DEX). Contract: every record for a given host MUST carry the
    #     same ``data`` bytes (i.e. the transform has already merged all
    #     segments into one host-wide buffer); we check and fail loudly if
    #     they disagree, because divergent ``data`` would make the
    #     generated APK non-deterministic w.r.t. which segment "wins".
    #   - ``appended``: whole-object transforms that add a brand-new member.
    overwrites: dict[str, InjectedPayload] = {}
    appended: list[InjectedPayload] = []
    for injected in injected_payloads:
        if injected.host_object_path is not None:
            host = injected.host_object_path.replace("\\", "/")
            existing = overwrites.get(host)
            if existing is None:
                overwrites[host] = injected
            elif existing.data != injected.data:
                # B2-shaped multi-segment sub-range transforms must all
                # merge their segments into a single ``data`` buffer
                # before emitting InjectedPayload records. A mismatch
                # here means the transform author forgot that step.
                raise SyntheticPackerError(
                    f"sub-range transform emitted conflicting data for "
                    f"host {host!r}: segments must share a merged buffer"
                )
            # else: identical data => this is a legitimate multi-segment
            # record sharing the same host buffer; keep the first.
        else:
            appended.append(injected)

    # A-v3 leakage fix (2026-05-07): rather than serialising native entries
    # first and concatenating ``appended`` at the tail (the old behaviour,
    # which made injected entries 100% land in the last contiguous block),
    # we interleave each ``appended`` payload at a random insertion point
    # in the native entry stream. The shuffle is RNG-deterministic for
    # reproducibility.
    rng_local = rng if rng is not None else random.Random(0)

    with zipfile.ZipFile(seed_apk, "r") as source:
        native_infos = [info for info in source.infolist() if not info.is_dir()]

        # Pick insertion indices (0..len(native_infos)) for each appended
        # payload; multiple payloads may share an index, in which case
        # they emit consecutively at that point — natural since
        # ``random.randint`` is independent.
        insertion_points: list[tuple[int, InjectedPayload]] = []
        for inj in appended:
            idx = rng_local.randint(0, len(native_infos))
            insertion_points.append((idx, inj))
        # Group insertions by index, preserving relative order of payloads
        # that share an index (split_xor's ``part`` ordering must survive).
        insertions_by_idx: dict[int, list[InjectedPayload]] = {}
        for idx, inj in insertion_points:
            insertions_by_idx.setdefault(idx, []).append(inj)

        with zipfile.ZipFile(generated_apk_out, "w") as target:
            # Emit any insertions scheduled before the first native entry.
            for inj in insertions_by_idx.pop(0, []):
                target.writestr(
                    _zip_info_for_injected(inj.object_path, naming_profile, rng_local),
                    inj.data,
                )
            for native_idx, info in enumerate(native_infos):
                member_path = info.filename.replace("\\", "/")
                override = overwrites.pop(member_path, None)
                if override is not None:
                    # Sub-range transform: replace this member's bytes with
                    # ``override.data`` (host_prefix + payload bytes).
                    data = override.data
                else:
                    with source.open(info) as handle:
                        data = handle.read()
                # Build a fresh ZipInfo instead of re-using ``info`` from the
                # source archive. Re-using it leaks the original
                # ``extra`` / ``compress_size`` / ``CRC`` fields into the new
                # ZIP, which can desynchronise when the member is re-encoded.
                # Preserve only the stable metadata callers rely on.
                new_info = zipfile.ZipInfo(
                    filename=info.filename,
                    date_time=info.date_time,
                )
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                new_info.create_system = info.create_system
                new_info.flag_bits = info.flag_bits & 0x800  # keep UTF-8 flag only
                target.writestr(new_info, data)
                # Emit any insertions scheduled to land *after* this native
                # entry (i.e. at insertion index ``native_idx + 1``).
                for inj in insertions_by_idx.pop(native_idx + 1, []):
                    target.writestr(
                        _zip_info_for_injected(
                            inj.object_path, naming_profile, rng_local
                        ),
                        inj.data,
                    )
            # Sanity: no insertion-index slots should remain unprocessed.
            if insertions_by_idx:
                # Should not happen — if randint produced > len(native_infos)
                # we already would have caught it above. Defensive: emit
                # them at the tail to avoid silent payload loss.
                for slots in insertions_by_idx.values():
                    for inj in slots:
                        target.writestr(
                            _zip_info_for_injected(
                                inj.object_path, naming_profile, rng_local
                            ),
                            inj.data,
                        )

    if overwrites:
        # A sub-range transform declared a host_object_path that did not
        # exist in the seed APK. Fail loudly rather than silently drop the
        # payload, which would make ``region_labels.jsonl`` inconsistent
        # with the actual APK bytes.
        missing = ", ".join(sorted(overwrites))
        raise SyntheticPackerError(
            f"sub-range transform references unknown host object(s): {missing}"
        )


def _build_manifest(
    *,
    seed_apk: Path,
    seed_apk_id: str,
    generated_apk_out: Path,
    generated_apk_id: str,
    payload_source: PayloadSource,
    transform_family: str,
    rng_seed: int,
    asset_prefix: str,
    split_count: int,
    injected_payloads: Sequence[InjectedPayload],
    labels_out: Optional[Path],
    payload_sha: str,
) -> dict:
    xor_keys = sorted(
        {injected.xor_key for injected in injected_payloads if injected.xor_key is not None}
    )
    return {
        "schema_version": 1,
        "seed_apk_path": str(seed_apk),
        "seed_apk_id": seed_apk_id,
        "generated_apk_path": str(generated_apk_out),
        "generated_apk_id": generated_apk_id,
        "labels_path": str(labels_out) if labels_out is not None else None,
        "transform_family": transform_family,
        "parameters": {
            "rng_seed": rng_seed,
            "asset_prefix": normalize_asset_prefix(asset_prefix),
            "split_count": split_count if transform_family == "split_xor" else None,
            "xor_keys": xor_keys,
        },
        "payload": {
            "source_kind": payload_source.kind,
            "source_apk_id": payload_source.source_apk_id,
            "source_object_path": payload_source.source_object_path,
            "payload_sha256": payload_sha,
            "payload_size": len(payload_source.data),
        },
        # ``offset_start`` / ``offset_end`` below are **object-local** byte
        # offsets describing where the payload lives within the injected
        # APK member. Whole-object transforms (``host_object_path is None``)
        # place the payload at ``[0, len(data))``; sub-range transforms
        # (``embedded_asset``, ``so_embedded``, ``dex_method_inlined``)
        # place it at ``[payload_offset_start, payload_offset_end)`` within
        # the overwritten host's full byte stream, which is the semantics
        # ``region_labels.jsonl`` downstream assumes.
        # ``source_offset_start`` / ``source_offset_end`` describe the byte
        # range of the *pre-transform* payload that this member represents;
        # ``split_xor`` is the current user that actually exercises them.
        "injected_objects": [
            {
                "object_path": injected.object_path,
                "host_object_path": injected.host_object_path,
                "offset_start": (
                    injected.payload_offset_start
                    if injected.host_object_path
                    else 0
                ),
                "offset_end": (
                    injected.payload_offset_end
                    if injected.host_object_path
                    else len(injected.data)
                ),
                "size": len(injected.data),
                "sha256": sha256(injected.data).hexdigest(),
                "transform_family": injected.transform_family,
                "part_index": injected.part_index,
                "part_count": injected.part_count,
                "source_offset_start": injected.payload_offset_start,
                "source_offset_end": injected.payload_offset_end,
                # A-v2 Fix-1 (2026-04-30): record per-segment xor_key in
                # the manifest. Necessary so reviewers / consumers can
                # recover (payload_segment -> xor_key) mapping for
                # families like ``split_xor`` where each segment now
                # carries an **independent** key (aligning with Qihoo
                # 360 multi-chunk encryption). ``None`` for transforms
                # that do not use an XOR stage (e.g. ``base64``).
                "xor_key": injected.xor_key,
            }
            for injected in injected_payloads
        ],
    }


def _zip_member_paths(apk_path: Path) -> set[str]:
    with zipfile.ZipFile(apk_path, "r") as archive:
        return {info.filename.replace("\\", "/") for info in archive.infolist()}


def _zip_info(object_path: str) -> zipfile.ZipInfo:
    """Legacy ZipInfo factory kept for unit-test compatibility.

    Production callers should prefer :func:`_zip_info_for_injected` so the
    emitted entry inherits the seed APK's date_time / external_attr /
    create_system distribution and avoids the L2-a / L14 leaks documented
    in ``scripts/diag_synthetic_leakage_v2.py``.
    """

    info = zipfile.ZipInfo(object_path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _zip_info_for_injected(
    object_path: str,
    naming_profile: Optional[SeedNamingProfile],
    rng: random.Random,
) -> zipfile.ZipInfo:
    """Build a ZipInfo that mimics the seed APK's native entry distribution.

    A-v3 leakage fix (2026-05-07): the previous ``_zip_info`` wrote a
    fixed ``date_time=(1980,1,1,0,0,0)`` and ``external_attr=0o644<<16``,
    which 100% separated injected entries from native ones in seed APKs
    that use a different normalised timestamp / attr (e.g. AOSP signed
    APKs use ``(1981,1,1,1,1,2)`` and ``0o600<<16``).

    A-v4 leakage fix (2026-05-08): v3 probe revealed two residual leaks
    this function was creating:

    * L21: because we used ``naming_profile.date_time_mode`` verbatim,
      100% of multi-injection tasks had all injected entries carry the
      IDENTICAL second-resolution timestamp (16/16 multi-injection
      tasks in v3 corpus). Native entries for the same APK instead
      spread across several date_time values. We now sample from
      ``date_time_pool`` per entry.
    * L22: same root cause — ``compress_type_mode`` meant every
      injected entry used DEFLATE (the seed's mode), whereas native
      entries have a ~25% STORE rate driven by pre-compressed content
      types (.png, .arsc, .so). We now sample ``compress_type`` from
      the per-entry frequency pool.

    ``external_attr`` / ``create_system`` remain scalar modes because
    native APKs nearly-always carry a single value for each (no
    intra-APK variance to mimic).
    """

    if naming_profile is None:
        return _zip_info(object_path)

    # A-v4: per-entry frequency-weighted sampling from the seed APK's
    # native pools. Falls back to ``*_mode`` when the profile was
    # constructed without the pool (e.g. unit-test shortcut).
    date_time = naming_profile.random_date_time(rng)
    compress_type = naming_profile.random_compress_type(rng)

    info = zipfile.ZipInfo(object_path, date_time=date_time)
    info.compress_type = compress_type
    info.external_attr = naming_profile.external_attr_mode
    info.create_system = naming_profile.create_system_mode
    return info


def build_seed_naming_profile(seed_apk: Path) -> SeedNamingProfile:
    """Snapshot the seed APK's ZIP-entry naming / metadata distributions.

    The returned :class:`SeedNamingProfile` is consumed by
    :func:`unique_payload_path` (to mint paths that mimic native naming)
    and :func:`_zip_info_for_injected` (to emit ZipInfo records whose
    date_time / external_attr / create_system land in the seed APK's
    mode bucket).

    Implementation notes:

    * ``directory_pool`` is the unique set of parent directories present
      in the seed, excluding empty (top-level) and ``META-INF/`` (which
      is signature territory we don't want to taint with synthetic
      entries — Android's signature verifier would refuse the resulting
      APK if we did, and reviewers would correctly flag it as
      unrealistic). Top-level injection is allowed (empty directory).
    * ``stem_lengths`` and ``extension_pool`` retain duplicates so
      :func:`SeedNamingProfile.random_*` samples reflect frequency, not
      uniqueness (i.e. the abundant ``.png`` extension is more likely to
      be picked than the single ``.kotlin_module``).
    * ``*_mode`` fields are the most common value across native entries,
      computed via :class:`collections.Counter`.
    """

    directories: list[str] = []
    stem_lengths: list[int] = []
    extensions: list[str] = []
    dir_ext_pairs: list[tuple[str, str]] = []  # A-v4: joint (dir, ext) pool (L19)
    date_time_samples: list[tuple[int, int, int, int, int, int]] = []  # A-v4 (L21)
    compress_type_samples: list[int] = []  # A-v4 (L22)
    date_times: Counter = Counter()
    external_attrs: Counter = Counter()
    create_systems: Counter = Counter()
    compress_types: Counter = Counter()

    with zipfile.ZipFile(seed_apk, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if "/" in name:
                directory, stem_with_ext = name.rsplit("/", 1)
            else:
                directory, stem_with_ext = "", name
            # Skip META-INF/ — we never inject there (signature territory).
            if directory.startswith("META-INF"):
                continue
            directories.append(directory)
            # Split stem and extension. Use the *last* dot so multi-dot
            # native names like ``foo.bar.png`` contribute ``.png``, not
            # ``.bar.png`` — which keeps extension_pool tight and the
            # synthetic stems realistic.
            if "." in stem_with_ext:
                stem, ext = stem_with_ext.rsplit(".", 1)
                ext = "." + ext
            else:
                stem, ext = stem_with_ext, ""
            if stem:
                stem_lengths.append(len(stem))
            extensions.append(ext)
            dir_ext_pairs.append((directory, ext))
            date_times[info.date_time] += 1
            external_attrs[info.external_attr] += 1
            create_systems[info.create_system] += 1
            compress_types[info.compress_type] += 1
            # A-v4 (2026-05-08): accumulate per-entry samples so downstream
            # per-entry sampling reflects the native frequency distribution
            # (not just the mode), closing L21 (date_time) and L22
            # (compress_type STORE vs DEFLATE split).
            date_time_samples.append(info.date_time)
            compress_type_samples.append(info.compress_type)

    if not directories:
        # Defensive fallback: a seed APK with no usable directories is
        # unrealistic; emit a minimal profile so callers don't crash.
        return SeedNamingProfile(
            directory_pool=("assets",),
            stem_lengths=(6,),
            extension_pool=(".bin",),
            date_time_mode=(1980, 1, 1, 0, 0, 0),
            external_attr_mode=0o644 << 16,
            create_system_mode=0,
            compress_type_mode=zipfile.ZIP_DEFLATED,
            dir_ext_pairs=(("assets", ".bin"),),
            date_time_pool=((1980, 1, 1, 0, 0, 0),),
            compress_type_pool=(zipfile.ZIP_DEFLATED,),
        )

    return SeedNamingProfile(
        directory_pool=tuple(directories),
        stem_lengths=tuple(stem_lengths),
        extension_pool=tuple(extensions),
        date_time_mode=date_times.most_common(1)[0][0],
        external_attr_mode=external_attrs.most_common(1)[0][0],
        create_system_mode=create_systems.most_common(1)[0][0],
        compress_type_mode=compress_types.most_common(1)[0][0],
        dir_ext_pairs=tuple(dir_ext_pairs),
        date_time_pool=tuple(date_time_samples),
        compress_type_pool=tuple(compress_type_samples),
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "InjectedPayload",
    "PayloadSource",
    "SUPPORTED_TRANSFORMS",
    "SyntheticBuildResult",
    "SyntheticPackerError",
    "build_seed_naming_profile",
    "build_synthetic_apk",
]
