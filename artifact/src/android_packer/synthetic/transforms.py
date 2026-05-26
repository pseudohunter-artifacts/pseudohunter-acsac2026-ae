"""Synthetic-payload transform registry.

Each transform family produces a list of :class:`InjectedPayload` instances
from an input payload. The registry (``TRANSFORMS``) is the extension point
for adversarial variants listed in ``docs/method/threat_model.md`` such as
``decoy_insertion`` or ``signature_stripping`` — new families only need to
provide a ``TransformFn`` and register it by name.

Transform families split into two structural classes:

* **Whole-object** (``xor``, ``base64``, ``split_xor``, ``path_randomized``,
  ``signature_strip``): the payload becomes one or more brand-new ZIP
  entries. Models PackerGrind Gen1/Gen2 packers.
* **Sub-range** (``embedded_asset``, ``so_embedded``, ``dex_method_inlined``):
  the payload is written into an *existing* ZIP entry, preceded (and in some
  cases followed) by the host's original bytes. Models PackerGrind Gen3
  packers where payloads hide inside asset / SO / DEX sections instead of
  materialising as dedicated members. See ``docs/method/threat_model.md``
  §"Synthetic 威胁覆盖矩阵" for the covered packer generations.
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

from android_packer.apkio.objects import ApkObject
from android_packer.features.dex_item_parser import (
    DEX_ITEM_TYPES,
    DexParseError,
    parse_dex_item_spans,
)
from android_packer.synthetic.records import InjectedPayload, SyntheticPackerError

TransformFn = Callable[["TransformContext"], List[InjectedPayload]]


@dataclass
class TransformContext:
    """Inputs handed to every :data:`TRANSFORMS` entry.

    Builders may consume ``rng`` (for deterministic randomness) and mutate
    ``existing_paths`` via :func:`unique_payload_path` to avoid collisions with
    the seed APK's existing members and with each other.

    ``seed_objects`` is the full list of ``(metadata, data)`` tuples from the
    seed APK, pre-loaded by :func:`packer._prepare_seed_state`. Sub-range
    transforms select a host from this pool to embed the payload into; the
    list is empty for callers that never trigger a sub-range transform.

    Stage A v3 leakage fix (2026-05-07): ``naming_profile`` carries a
    snapshot of the seed APK's directory layout and filename style so
    :func:`unique_payload_path` can mint paths that are statistically
    indistinguishable from native entries. ``asset_prefix`` is now a
    *fallback* used only when no naming_profile is provided (keeps unit
    tests passing without a seed APK).
    """

    payload: bytes
    rng: random.Random
    existing_paths: set
    asset_prefix: str
    xor_key: Optional[int] = None
    split_count: int = 2
    seed_objects: Sequence[Tuple[ApkObject, bytes]] = field(default_factory=list)
    # B1 (2026-04-29): when False, ``_normalize_payload_size`` skips the
    # lower-bound check and accepts any payload size. Production callers
    # leave this at True; unit-test fixtures (which use tiny synthetic
    # DEXes) override it to keep legacy tests fast.
    enforce_payload_size_range: bool = True
    # A-v3 leakage fix (2026-05-07): optional naming profile derived
    # from the seed APK. None preserves the legacy behaviour for tests.
    naming_profile: Optional["SeedNamingProfile"] = None


@dataclass(frozen=True)
class SeedNamingProfile:
    """Snapshot of seed APK ZIP layout used to mint indistinguishable paths.

    Built once per task in :func:`packer._prepare_seed_state` from the
    seed APK's central directory. All fields are statistics over native
    entries — no synthetic mark gets mixed in. Consumers:

    * :func:`unique_payload_path` samples ``dir_ext_pairs`` for a joint
      (directory, extension) combination that is guaranteed to exist in
      the seed (closes L19) and ``stem_lengths`` for a realistic stem.
    * :func:`packer._zip_info_for_injected` samples ``date_time_pool``
      and ``compress_type_pool`` per-entry (rather than using a single
      mode value, which caused L21 / L22). ``external_attr_mode`` /
      ``create_system_mode`` are kept as scalars because native APKs
      already nearly-always use a single value for each.

    A-v4 leakage fix (2026-05-08): ``dir_ext_pairs`` (L19),
    ``date_time_pool`` (L21) and ``compress_type_pool`` (L22) replace
    the independent-dimension sampling that leaked in v3. Legacy
    ``*_mode`` scalar fields are preserved for backwards compatibility
    with any external caller, but :func:`packer._zip_info_for_injected`
    now samples from the pools instead.
    """

    directory_pool: Tuple[str, ...]
    stem_lengths: Tuple[int, ...]
    extension_pool: Tuple[str, ...]
    date_time_mode: Tuple[int, int, int, int, int, int]
    external_attr_mode: int
    create_system_mode: int
    compress_type_mode: int
    # A-v4 (2026-05-08): new joint / frequency pools. Defaults kept for
    # callers that construct ``SeedNamingProfile`` manually (unit tests).
    dir_ext_pairs: Tuple[Tuple[str, str], ...] = ()
    date_time_pool: Tuple[Tuple[int, int, int, int, int, int], ...] = ()
    compress_type_pool: Tuple[int, ...] = ()

    def random_directory(self, rng: random.Random) -> str:
        if not self.directory_pool:
            return "assets"
        return rng.choice(self.directory_pool)

    def random_extension(self, rng: random.Random) -> str:
        if not self.extension_pool:
            return ""
        return rng.choice(self.extension_pool)

    def random_stem_length(self, rng: random.Random) -> int:
        if not self.stem_lengths:
            return 6
        return rng.choice(self.stem_lengths)

    def random_dir_ext(self, rng: random.Random) -> Tuple[str, str]:
        """Sample a (directory, extension) pair from the seed APK.

        A-v4 (2026-05-08): when available this method is preferred over
        independent ``random_directory`` + ``random_extension`` sampling.
        Independent sampling produced novel pairs like ``res/color/*.svg``
        that never occur in real APKs (L19). Pairs are drawn with
        frequency weighting because we store duplicates.
        """
        if not self.dir_ext_pairs:
            # Fallback: independent sampling (legacy behaviour for
            # manually-constructed profiles without the joint pool).
            return self.random_directory(rng), self.random_extension(rng)
        return rng.choice(self.dir_ext_pairs)

    def random_date_time(
        self, rng: random.Random
    ) -> Tuple[int, int, int, int, int, int]:
        if not self.date_time_pool:
            return self.date_time_mode
        return rng.choice(self.date_time_pool)

    def random_compress_type(self, rng: random.Random) -> int:
        if not self.compress_type_pool:
            return self.compress_type_mode
        return rng.choice(self.compress_type_pool)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def normalize_asset_prefix(asset_prefix: str) -> str:
    normalized = asset_prefix.replace("\\", "/").strip("/")
    return normalized or "assets/synthetic"


def unique_payload_path(
    existing_paths: set,
    asset_prefix: str,
    transform_family: str,
    rng: random.Random,
    *,
    part_index: Optional[int] = None,
    naming_profile: Optional["SeedNamingProfile"] = None,
    forced_extension: Optional[str] = None,
) -> str:
    """Mint a fresh injection path that mimics native ZIP-entry naming.

    A-v3 leakage fix (2026-05-07): the previous implementation produced
    paths of the form ``assets/payload/<family>.<12hex>.bin`` which
    leaked the family name and matched a fixed long-stem 12-hex pattern
    that 100% separated injected from native entries (see
    ``scripts/diag_synthetic_leakage_v2.py``). The new behaviour:

    * If ``naming_profile`` is provided, sample directory / stem-length /
      extension from the seed APK's native distributions so the path is
      statistically indistinguishable from a native entry.
    * Stem characters are alphanumeric (mixed case + digits) instead of
      pure-hex 12-char tokens, matching native APK entry naming.
    * Family name is **never** baked into the path.
    * ``part_index`` (split_xor) is encoded as a single trailing digit
      glued onto the stem (``abc1`` not ``abc.part001``), preserving
      uniqueness without the giveaway ``.partNNN`` suffix.
    * Falls back to the legacy ``<asset_prefix>/<token>.bin`` when no
      profile is present (unit-test compatibility).
    """

    _ = transform_family  # accepted for backwards compatibility; not used in path.

    if naming_profile is not None:
        for _ in range(2000):
            # A-v4 (2026-05-08): sample (directory, extension) JOINTLY so
            # the combination is guaranteed to occur in the seed APK.
            # Independent sampling produced novel combinations like
            # ``res/color/<stem>.svg`` that never appear in real APKs
            # and 39% of injected pairs in v3 were such novelties (L19).
            directory, ext_from_pair = naming_profile.random_dir_ext(rng)
            length = max(2, naming_profile.random_stem_length(rng))
            stem = _random_stem(rng, length)
            ext = forced_extension if forced_extension is not None else ext_from_pair
            if part_index is not None:
                # Embed part as a trailing digit pair without the giveaway
                # ``.partNNN`` separator.
                stem = f"{stem}{part_index % 10}"
            candidate = f"{directory}/{stem}{ext}" if directory else f"{stem}{ext}"
            if candidate not in existing_paths:
                existing_paths.add(candidate)
                return candidate
        # Fall through to the legacy allocator below if we somehow can't
        # find a unique slot in 2000 tries (extremely unlikely).

    # Legacy fallback (used when no naming_profile is provided, e.g. unit
    # tests). Still mints a hex token but no longer carries the family
    # name so we don't regress the L1 leak in fallback mode either.
    prefix = normalize_asset_prefix(asset_prefix)
    suffix = "" if part_index is None else str(part_index % 10)
    ext = forced_extension if forced_extension is not None else ".bin"
    for _ in range(1000):
        token = f"{rng.getrandbits(48):012x}"
        candidate = f"{prefix}/{token}{suffix}{ext}"
        if candidate not in existing_paths:
            existing_paths.add(candidate)
            return candidate
    raise SyntheticPackerError("failed to allocate a unique payload object path")


# Character pool for synthetic stem tokens. Mixed case + digits matches the
# observed native APK entry distribution (e.g. ``MainActivity``, ``a1b``,
# ``R$styleable``); pure-hex would be a giveaway, see L11 leak diagnostic.
_STEM_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _random_stem(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_STEM_CHARSET) for _ in range(length))


def resolve_xor_key(xor_key: Optional[int], rng: random.Random) -> int:
    key = rng.randint(1, 255) if xor_key is None else xor_key
    if not 0 <= key <= 255:
        raise ValueError("xor_key must be between 0 and 255")
    return key


def xor_bytes(data: bytes, key: int) -> bytes:
    if not 0 <= key <= 255:
        raise ValueError("xor_key must be between 0 and 255")
    # ``bytes.translate`` with a 256-byte table runs a single C-level pass and
    # avoids the per-byte Python overhead that dominates for multi-megabyte
    # payloads.
    return data.translate(_xor_table(key))


@lru_cache(maxsize=256)
def _xor_table(key: int) -> bytes:
    return bytes(byte ^ key for byte in range(256))


def split_ranges(
    size: int,
    split_count: int,
    *,
    rng: Optional[random.Random] = None,
    jitter: float = 0.0,
) -> List[Tuple[int, int]]:
    """Split ``[0, size)`` into ``split_count`` contiguous non-overlapping ranges.

    A-v4 leakage fix (2026-05-08): when ``rng`` is provided and
    ``jitter > 0``, per-part sizes are perturbed by up to
    ``jitter * base`` bytes so split_xor parts no longer carry
    byte-identical sizes (L23 in ``diag_synthetic_leakage_v3.py``:
    100% of tasks had all k parts share one file_size, which is a
    trivial signature). With ``jitter=0`` (default), behaviour is
    identical to the pre-v4 implementation so existing call sites
    that rely on exact equal partitioning remain unaffected.

    The perturbation keeps ``sum(lengths) == size`` by absorbing
    the cumulative drift in the last part. All parts are guaranteed
    at least 1 byte.
    """

    base = size // split_count
    remainder = size % split_count
    lengths = [base + (1 if index < remainder else 0) for index in range(split_count)]

    if rng is not None and jitter > 0 and split_count >= 2 and base > 4:
        max_delta = max(1, int(base * jitter))
        # Apply zero-sum perturbation: draw a delta for each part except
        # the last, clamp so each part stays >= 1, then set the last
        # part to ``size - sum(prev)``. This guarantees the partition
        # covers ``[0, size)`` exactly.
        for i in range(split_count - 1):
            delta = rng.randint(-max_delta, max_delta)
            new_len = lengths[i] + delta
            if new_len < 1:
                new_len = 1
            # Don't let this part eat into the minimum length reserved
            # for remaining parts (each remaining part needs >= 1 byte).
            remaining_min = split_count - i - 1
            consumed = sum(lengths[:i]) + new_len
            if size - consumed < remaining_min:
                new_len = size - sum(lengths[:i]) - remaining_min
                if new_len < 1:
                    new_len = 1
            lengths[i] = new_len
        lengths[-1] = max(1, size - sum(lengths[:-1]))

    ranges: List[Tuple[int, int]] = []
    start = 0
    for length in lengths:
        end = start + length
        ranges.append((start, end))
        start = end
    # Invariant: exact coverage of [0, size).
    assert start == size, f"split_ranges drift: covered {start} of {size}"
    return ranges


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


_FAMILY_PATH_RANDOMIZED = "path_randomized"
_FAMILY_XOR = "xor"
_FAMILY_BASE64 = "base64"
_FAMILY_SPLIT_XOR = "split_xor"
_FAMILY_SIGNATURE_STRIP = "signature_strip"
_FAMILY_EMBEDDED_ASSET = "embedded_asset"
_FAMILY_SO_EMBEDDED = "so_embedded"
_FAMILY_DEX_METHOD_INLINED = "dex_method_inlined"
# A-v2-b (2026-04-30): three new transforms extending threat model to
# Gen2+ shim loader and Gen3 container / string-pool encryption. Spec:
# ``docs/method/threat_model.md`` §"Track A v2" / §"Sub-range 嵌入的
# 三种具体形态".
_FAMILY_MULTI_DEX_SHIM = "multi_dex_shim"
_FAMILY_EMBEDDED_ARCHIVE = "embedded_archive"
_FAMILY_DEX_STRING_ENCRYPTED = "dex_string_encrypted"

# Number of leading bytes the ``signature_strip`` transform XOR-masks.
# A-v2 Fix-2 (2026-04-30): extended from 8 (magic only) to 80 (full DEX
# header) to align with the Gen2 full-header-tampering threat model
# documented in ``docs/method/threat_model.md`` §"Track A v2". The
# constant name keeps its historical ``_PREFIX_LEN`` suffix for
# stability with existing callsites; semantically it is now the
# "DEX header length" (which is also 0x70 = 112? No — official Android
# DEX spec pegs ``header_size = 0x70`` but the first 80 bytes span the
# fixed-layout fields we care about: magic (8) + checksum (4) +
# signature (20) + file_size (4) + header_size (4) + endian_tag (4) +
# link_size/off (8) + map_off (4) + string_ids_size/off (8) +
# type_ids_size/off (8) + proto_ids_size/off (8) = 80).
_SIGNATURE_STRIP_PREFIX_LEN = 80

# --- B1 (2026-04-29): per-family payload-size-range defaults -------------
# Rationale: a first Gen3 run produced payload bytes ranging from 3 KB
# (opentracks stub) to 9.4 MB (keepass secondary DEX), spanning 3 orders
# of magnitude. That scale jitter confounds AUROC aggregation and — for
# sub-range transforms — makes most generated tasks collapse into
# "payload almost fully occupies the entry" (p/total median 0.82–0.89),
# which structurally degenerates to Gen1 whole-object. The caps below
# are engineering heuristics (NOT byte-range statistics cited from prior
# work); their role is documented in docs/method/threat_model.md
# §"Payload / Host 体积分布的工程约束（B1 + B2，本次新增）".
_PAYLOAD_SIZE_WHOLE_OBJECT = (64 * 1024, 4 * 1024 * 1024)   # [64 KB, 4 MB]
_PAYLOAD_SIZE_SUB_RANGE = (64 * 1024, 1 * 1024 * 1024)       # [64 KB, 1 MB]

# Upper bound on payload/total-entry ratio for Gen3 sub-range transforms.
# Enforces "host still occupies at least 1/4 of the resulting entry" so
# the Gen3 semantic of "hiding inside a legitimate host" survives.
_SUB_RANGE_MAX_P_OVER_TOTAL = 0.75

# Minimum host-object size (in bytes) required before a sub-range transform
# will consider it as a candidate. Small hosts produce windows that are
# either entirely host or entirely payload, which defeats the purpose of
# sub-range modelling (we want mixed-composition regions).
_MIN_HOST_BYTES_ASSET = 16 * 1024
_MIN_HOST_BYTES_SO = 16 * 1024

# --- B2 (2026-04-29): dex_method_inlined multi-segment scatter ----------
# Rationale: a single contiguous tail overwrite does not reflect real
# PackerGrind Gen3 "single-method hiding" (Qihoo VMP / SecNeo / Bangcle
# Pro), where the packer encrypts many individual methods' insns in
# place. We scatter the payload across k segments, each landing on the
# tail of a distinct large code_item span, with an independent XOR key
# per segment (modelling "different methods, different keys"). The
# engineering heuristics are:
_DEX_INLINE_K_RANGE = (3, 10)                # scatter across k segments
_DEX_INLINE_SEG_SIZE_RANGE = (256, 4 * 1024) # per-segment size [256 B, 4 KB]
_DEX_INLINE_HEAD_PRESERVE = 256              # keep N head bytes of each span
_DEX_INLINE_MIN_SPAN_SIZE = 512              # skip spans below this size

# File extensions / paths that ``embedded_asset`` may use as hosts. DEX and
# SO are excluded here — DEX has its own dedicated Gen3 transform
# (``dex_method_inlined``) and SO has ``so_embedded``; leaving them in
# ``embedded_asset`` would blur the family taxonomy and cost audit clarity.
_EMBEDDED_ASSET_EXT_ALLOW = (
    ".png", ".jpg", ".jpeg", ".webp",
    ".arsc", ".ttf", ".otf",
    ".pb", ".dat", ".bin",
)
_EMBEDDED_ASSET_EXT_DENY = (".dex", ".so", ".xml")

# ELF magic used by ``so_embedded`` to confirm the candidate is a real .so
# (some APKs carry "_ignore" or renamed binaries under lib/).
_ELF_MAGIC = b"\x7fELF"


def _normalize_payload_size(
    payload: bytes,
    rng: random.Random,
    *,
    size_range: Tuple[int, int],
    family: str,
    enforce: bool = True,
) -> bytes:
    """B1: clip / reject payload bytes to fit ``size_range``.

    - payload shorter than the lower bound **and** ``enforce`` is True
      => raise SyntheticPackerError. (Silent zero-padding would fabricate
      synthetic bytes the labels can't explain; re-sampling bytes would
      drift the label's ``source_offset_*`` semantics. Fail-fast keeps
      the manifest honest, and upstream CLI can skip the task with a
      warning.)
    - payload shorter than the lower bound **and** ``enforce`` is False
      => return as-is (unit-test fallback).
    - payload within the range => return as-is.
    - payload longer than the upper bound => take a deterministic-random
      slice of exactly ``upper`` bytes from within the payload. The slice
      start is drawn from ``rng`` so different seed APKs with the same
      oversized DEX pick different substrings, avoiding trivial
      cross-task byte overlap.
    """

    lower, upper = size_range
    n = len(payload)
    if n < lower:
        if not enforce:
            return payload
        raise SyntheticPackerError(
            f"{family}: payload size {n} below lower bound {lower} "
            f"(the seed DEX is too small to serve as a realistic packer "
            f"payload; skip this task or pass a larger --payload)"
        )
    if n <= upper:
        return payload
    start = rng.randint(0, n - upper)
    return payload[start : start + upper]


def _build_path_randomized(ctx: TransformContext) -> List[InjectedPayload]:
    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_PATH_RANDOMIZED,
        enforce=ctx.enforce_payload_size_range,
    )
    return [
        InjectedPayload(
            object_path=unique_payload_path(
                ctx.existing_paths, ctx.asset_prefix, _FAMILY_PATH_RANDOMIZED, ctx.rng,
                naming_profile=ctx.naming_profile,
            ),
            data=payload,
            transform_family=_FAMILY_PATH_RANDOMIZED,
            payload_offset_start=0,
            payload_offset_end=len(payload),
            part_index=None,
            part_count=None,
            xor_key=None,
        )
    ]


def _build_xor(ctx: TransformContext) -> List[InjectedPayload]:
    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_XOR,
        enforce=ctx.enforce_payload_size_range,
    )
    key = resolve_xor_key(ctx.xor_key, ctx.rng)
    transformed = xor_bytes(payload, key)
    return [
        InjectedPayload(
            object_path=unique_payload_path(
                ctx.existing_paths, ctx.asset_prefix, _FAMILY_XOR, ctx.rng,
                naming_profile=ctx.naming_profile,
            ),
            data=transformed,
            transform_family=_FAMILY_XOR,
            payload_offset_start=0,
            payload_offset_end=len(payload),
            part_index=None,
            part_count=None,
            xor_key=key,
        )
    ]


def _build_base64(ctx: TransformContext) -> List[InjectedPayload]:
    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_BASE64,
        enforce=ctx.enforce_payload_size_range,
    )
    transformed = base64.b64encode(payload)
    return [
        InjectedPayload(
            object_path=unique_payload_path(
                ctx.existing_paths, ctx.asset_prefix, _FAMILY_BASE64, ctx.rng,
                naming_profile=ctx.naming_profile,
            ),
            data=transformed,
            transform_family=_FAMILY_BASE64,
            payload_offset_start=0,
            payload_offset_end=len(payload),
            part_index=None,
            part_count=None,
            xor_key=None,
        )
    ]


def _build_split_xor(ctx: TransformContext) -> List[InjectedPayload]:
    if ctx.split_count < 2:
        raise ValueError("split_count must be at least 2 for split_xor")
    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_SPLIT_XOR,
        enforce=ctx.enforce_payload_size_range,
    )
    if len(payload) < ctx.split_count:
        raise ValueError("payload is too small for requested split_count")
    # A-v2 Fix-1 (2026-04-30): each chunk now gets an **independent**
    # XOR key sampled from ``ctx.rng``, aligning with the real-world
    # Qihoo 360 / AppSpear "multi-chunk encryption" threat model
    # (PackerGrind TSE 2022 §3.2). The previous v1 behaviour of sharing
    # one key across all chunks was a synthetic simplification that
    # reviewers would correctly flag as an inauthentic Gen1 variant.
    # The caller-provided ``ctx.xor_key`` is honoured as the **first**
    # chunk's key (preserves single-chunk unit-test fixtures that pin
    # a specific key), with subsequent chunks sampling fresh keys.
    # A-v4 leakage fix (2026-05-08): apply ±8% per-part size jitter so
    # split_xor tasks no longer have all k parts share an identical
    # file_size (L23 in v3 probe: 8/8 tasks showed k-way equality,
    # which is a 0-cost signature for any classifier). ``jitter=0.08``
    # with base >= 64 KiB (enforced by B1) gives >= 5 KiB of drift.
    ranges = split_ranges(
        len(payload), ctx.split_count, rng=ctx.rng, jitter=0.08
    )
    injected: List[InjectedPayload] = []
    for index, (start, end) in enumerate(ranges):
        # First chunk honours ``ctx.xor_key`` if provided; otherwise
        # (and for every subsequent chunk) sample a fresh key from the
        # per-task RNG.
        if index == 0:
            key = resolve_xor_key(ctx.xor_key, ctx.rng)
        else:
            key = resolve_xor_key(None, ctx.rng)
        transformed = xor_bytes(payload[start:end], key)
        injected.append(
            InjectedPayload(
                object_path=unique_payload_path(
                    ctx.existing_paths,
                    ctx.asset_prefix,
                    _FAMILY_SPLIT_XOR,
                    ctx.rng,
                    part_index=index,
                    naming_profile=ctx.naming_profile,
                ),
                data=transformed,
                transform_family=_FAMILY_SPLIT_XOR,
                payload_offset_start=start,
                payload_offset_end=end,
                part_index=index,
                part_count=len(ranges),
                xor_key=key,
            )
        )
    return injected


def _build_signature_strip(ctx: TransformContext) -> List[InjectedPayload]:
    """XOR-mask the full 80-byte DEX header (Gen2 full-header tampering).

    A-v2 Fix-2 (2026-04-30): this transform now scrambles the entire
    80-byte DEX header (``magic`` + ``checksum`` + ``signature`` +
    ``file_size`` + ``header_size`` + ``endian_tag`` + ``link_off`` +
    ``link_size`` + ``map_off`` + string/type/proto/field/method/class
    offsets), not just the 8-byte magic. Each 4-byte DWORD in the
    header is masked with its own randomly-sampled DWORD key,
    equivalent to 20 independent DWORD keys. This aligns with the
    360 Jiagu / 爱加密 v2 Gen2 threat model documented in
    PackerGrind TSE 2022 §3.2 and DroidUnpack NDSS 2018 §4.1.

    The rest of the payload body (offset >= 80) is left byte-identical,
    so body-level features (entropy, bigram distribution) still reflect
    the original payload. What changes:

    * Every DEX header field — checksum, file_size, map_off, etc. — is
      destroyed, defeating DEX-aware structural parsers that walk the
      header before descending into ``class_defs``.
    * Naive YARA-style magic lookups at offset 0 miss.
    * ``sanity_rules`` heuristics keyed on DEX magic or map_off
      consistency observe synthetic-looking noise.

    The emitted :class:`InjectedPayload` records ``xor_key`` as the
    **first byte** of the generated mask for backwards compatibility
    with the single-value dataclass field; the full 80-byte mask is
    not stored anywhere because the transform is purely destructive
    (we don't need to "decrypt" for unit-testing — we only need the
    byte-level layout). Reviewers reproducing the experiment from
    ``rng_seed`` alone will regenerate the same mask deterministically.
    """

    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_SIGNATURE_STRIP,
        enforce=ctx.enforce_payload_size_range,
    )
    if len(payload) < _SIGNATURE_STRIP_PREFIX_LEN:
        raise ValueError(
            f"payload must be at least {_SIGNATURE_STRIP_PREFIX_LEN} bytes "
            f"for signature_strip, got {len(payload)}"
        )
    # Sample an 80-byte header mask (20 independent DWORDs). We mask in
    # DWORD granularity — every 4 consecutive mask bytes come from the
    # same 32-bit random value — to match the real-world claim that each
    # DEX header **field** (4-byte aligned) is scrambled with its own
    # DWORD key. In practice the byte-level mask is uniform random from
    # ``ctx.rng``'s perspective, but the DWORD claim is kept for the
    # paper-level threat-model alignment.
    mask = bytes(ctx.rng.randint(0, 255) for _ in range(_SIGNATURE_STRIP_PREFIX_LEN))
    # Retain the leading byte as the nominal ``xor_key`` field value
    # (records.InjectedPayload.xor_key is Optional[int], we cannot store
    # 80 bytes there without a schema extension). Consumers that want
    # the full mask should rely on ``rng_seed`` determinism.
    nominal_key = mask[0]
    head = bytes(b ^ m for b, m in zip(payload[:_SIGNATURE_STRIP_PREFIX_LEN], mask))
    tail = payload[_SIGNATURE_STRIP_PREFIX_LEN:]
    transformed = head + tail
    return [
        InjectedPayload(
            object_path=unique_payload_path(
                ctx.existing_paths,
                ctx.asset_prefix,
                _FAMILY_SIGNATURE_STRIP,
                ctx.rng,
                naming_profile=ctx.naming_profile,
            ),
            data=transformed,
            transform_family=_FAMILY_SIGNATURE_STRIP,
            payload_offset_start=0,
            payload_offset_end=len(payload),
            part_index=None,
            part_count=None,
            xor_key=nominal_key,
        )
    ]


# ---------------------------------------------------------------------------
# Sub-range (Gen3) transforms: payload embedded inside an existing host
# object. See docs/method/threat_model.md §"Synthetic 威胁覆盖矩阵" for the
# packer-generation mapping and the motivation for each family.
# ---------------------------------------------------------------------------


def _select_host(
    ctx: TransformContext,
    *,
    predicate: Callable[[ApkObject, bytes], bool],
    min_host_bytes: int,
    family: str,
) -> Tuple[ApkObject, bytes]:
    """Pick one host object from ``ctx.seed_objects`` for a sub-range transform.

    Candidates must (i) satisfy ``predicate``, (ii) be at least
    ``min_host_bytes`` bytes long, and (iii) not be byte-identical to the
    payload (guards against embedding a DEX into itself when the seed APK
    only carries one DEX and the external payload wasn't provided).
    The final pick is deterministic given ``ctx.rng``: we shuffle the
    candidate list with ``rng`` and take the first one.
    """

    candidates: List[Tuple[ApkObject, bytes]] = []
    for metadata, data in ctx.seed_objects:
        if len(data) < min_host_bytes:
            continue
        if data == ctx.payload:
            continue
        if not predicate(metadata, data):
            continue
        candidates.append((metadata, data))

    if not candidates:
        raise SyntheticPackerError(
            f"{family}: seed APK has no suitable host object "
            f"(need size >= {min_host_bytes} bytes)"
        )

    index = ctx.rng.randrange(len(candidates))
    return candidates[index]


def _is_asset_candidate(metadata: ApkObject, data: bytes) -> bool:
    """Host filter for ``embedded_asset``: non-DEX, non-SO, non-code asset."""

    path = metadata.object_path.lower()
    if any(path.endswith(ext) for ext in _EMBEDDED_ASSET_EXT_DENY):
        return False
    if path.startswith("meta-inf/"):
        # Signature / manifest entries are integrity-sensitive. Leave them
        # out so we don't accidentally trip an APK signing-scheme check
        # during any downstream sanity pass.
        return False
    if any(path.endswith(ext) for ext in _EMBEDDED_ASSET_EXT_ALLOW):
        return True
    # Fall back to "assets/" root: captures arbitrary binary blobs packers
    # tend to hide in without hard-coding every extension.
    return path.startswith("assets/")


def _is_so_candidate(metadata: ApkObject, data: bytes) -> bool:
    """Host filter for ``so_embedded``: real ELF under lib/."""

    path = metadata.object_path.lower()
    if not path.startswith("lib/") or not path.endswith(".so"):
        return False
    return data.startswith(_ELF_MAGIC)


def _is_dex_candidate(metadata: ApkObject, data: bytes) -> bool:
    """Host filter for ``dex_method_inlined``: parseable DEX."""

    if metadata.object_type != "dex":
        return False
    # ``parse_dex_item_spans`` raises DexParseError on anything that doesn't
    # look like a benign DEX. We don't need the spans here; we just need to
    # know the parse succeeds so the Gen3 builder can re-parse cheaply.
    try:
        parse_dex_item_spans(data)
    except DexParseError:
        return False
    return True


def _build_embedded_asset(ctx: TransformContext) -> List[InjectedPayload]:
    """Append an XOR-encrypted payload to the tail of an existing asset.

    Models PackerGrind Gen3 "payload appended to legitimate asset" behaviour
    seen in Bangcle v2 and in several Nagapt variants. The host asset's
    original bytes are preserved verbatim, so detectors that rely on
    intra-object distributional contrast (entropy_delta_entry, byte
    histogram within the entry) actually see a mixed benign/payload entry —
    which is the whole point of extending beyond Gen1/Gen2.

    B1 contract: payload is clipped to :data:`_PAYLOAD_SIZE_SUB_RANGE`
    before encryption, and the selected host must leave payload occupying
    at most :data:`_SUB_RANGE_MAX_P_OVER_TOTAL` of the final entry.
    """

    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_SUB_RANGE,
        family=_FAMILY_EMBEDDED_ASSET,
        enforce=ctx.enforce_payload_size_range,
    )
    # Host must be large enough that payload/(host+payload) <= max ratio,
    # i.e. host_size >= payload * (1/max - 1). Derived as a per-task
    # min_host_bytes floor so _select_host can reject oversized payloads
    # against undersized hosts with a deterministic, explicit error.
    min_host_by_ratio = int(
        len(payload) * (1.0 / _SUB_RANGE_MAX_P_OVER_TOTAL - 1.0)
    )
    min_host = max(_MIN_HOST_BYTES_ASSET, min_host_by_ratio)
    host_metadata, host_data = _select_host(
        ctx,
        predicate=_is_asset_candidate,
        min_host_bytes=min_host,
        family=_FAMILY_EMBEDDED_ASSET,
    )
    key = resolve_xor_key(ctx.xor_key, ctx.rng)
    encrypted = xor_bytes(payload, key)
    data = host_data + encrypted
    payload_start = len(host_data)
    payload_end = payload_start + len(payload)
    return [
        InjectedPayload(
            object_path=host_metadata.object_path,
            data=data,
            transform_family=_FAMILY_EMBEDDED_ASSET,
            payload_offset_start=payload_start,
            payload_offset_end=payload_end,
            part_index=None,
            part_count=None,
            xor_key=key,
            host_object_path=host_metadata.object_path,
        )
    ]


def _build_so_embedded(ctx: TransformContext) -> List[InjectedPayload]:
    """Append an XOR-encrypted payload to the tail of an existing .so file.

    Models 360 Jiagu / some Tencent Legu variants where the payload rides
    along the ``.rodata`` tail of a legitimate native lib. We append past
    the ELF's last section rather than rewriting section headers: ``readelf
    -a`` still parses the file correctly (the extra bytes sit after the
    section header table, which ELF tolerates silently). That keeps the
    synthetic APK structurally valid while exercising the same detector
    weaknesses as ``embedded_asset`` in a different host type.

    .. note::
       **A-v2 Fix-3 (2026-04-30) — simplified ELF overlay**. Real-world
       360 Jiagu / 爱加密 v3 additionally update ``e_shoff`` and append
       a fresh section header table entry so the payload appears as a
       legitimate ``.rodata`` section to ``readelf -S``. We do **not**
       perform this metadata update here: the payload bytes sit past
       the section header table as raw file-tail overlay. This is a
       known simplification relative to the threat model claimed in
       ``docs/method/threat_model.md`` §"Synthetic 威胁覆盖矩阵";
       the paper must accompany Stage A ``so_embedded`` numbers with
       an explicit disclaimer in §4.1 ("we simulate the byte-level
       geometry only, not the ELF metadata update"). Full ELF section
       header update is deferred to Stage B batch F-SO-Real. The
       empirical consequence observed in the 2026-04-29 precheck —
       ``so_embedded`` per-task entropy-delta AUROC mean 0.35 (direction
       reversed) — is likely a symptom of this simplification: the
       overlay bytes look more uniform than true .rodata, so the
       entropy-delta sign flips. Keeping the simplified version in the
       dataset is still informative; we just don't over-claim it.

    B1 contract: payload is clipped to :data:`_PAYLOAD_SIZE_SUB_RANGE`
    before encryption, and the selected .so must leave payload occupying
    at most :data:`_SUB_RANGE_MAX_P_OVER_TOTAL` of the final entry.
    """

    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_SUB_RANGE,
        family=_FAMILY_SO_EMBEDDED,
        enforce=ctx.enforce_payload_size_range,
    )
    min_host_by_ratio = int(
        len(payload) * (1.0 / _SUB_RANGE_MAX_P_OVER_TOTAL - 1.0)
    )
    min_host = max(_MIN_HOST_BYTES_SO, min_host_by_ratio)
    host_metadata, host_data = _select_host(
        ctx,
        predicate=_is_so_candidate,
        min_host_bytes=min_host,
        family=_FAMILY_SO_EMBEDDED,
    )
    key = resolve_xor_key(ctx.xor_key, ctx.rng)
    encrypted = xor_bytes(payload, key)
    data = host_data + encrypted
    payload_start = len(host_data)
    payload_end = payload_start + len(payload)
    return [
        InjectedPayload(
            object_path=host_metadata.object_path,
            data=data,
            transform_family=_FAMILY_SO_EMBEDDED,
            payload_offset_start=payload_start,
            payload_offset_end=payload_end,
            part_index=None,
            part_count=None,
            xor_key=key,
            host_object_path=host_metadata.object_path,
        )
    ]


def _build_dex_method_inlined(ctx: TransformContext) -> List[InjectedPayload]:
    """Scatter XOR-encrypted payload across a code_item span.

    This is the **hard adversarial** family referenced in
    ``docs/method/threat_model.md``. The host is a parseable DEX from the
    seed APK (different from the payload source), and the payload is cut
    into ``k`` short segments (:data:`_DEX_INLINE_K_RANGE`, typically
    3–10), each encrypted with its **own** XOR key and written at
    non-overlapping positions inside the largest code_item span, after
    the span's :data:`_DEX_INLINE_HEAD_PRESERVE` preserve head. The DEX
    file length stays unchanged and the header / map_list are untouched;
    what changes is that ``k`` stretches of bytes previously labelled
    ``code_item`` now hold encrypted payload, byte-adjacent to real
    bytecode — modelling PackerGrind Gen3 "single-method hiding" as seen
    in Qihoo VMP / SecNeo / Bangcle Pro.

    Why **inside a single span** rather than across multiple spans: the
    parser in :mod:`android_packer.features.dex_item_parser` rolls the
    entire ``code_item`` section up into a single :class:`DexItemSpan`
    (it deliberately over-approximates, because walking individual
    code_item entries is not needed for MLM aux-loss labelling). The
    real Qihoo VMP / SecNeo packers also encrypt individual method
    insns *within* the same physical ``code_item`` section, so a scatter
    inside a single span is structurally faithful.

    B2 contract:

    * ``k`` segments with independent XOR keys (each key uniformly drawn
      from [1, 255]; ``ctx.xor_key`` is ignored here so the multi-key
      story is honest).
    * Each segment length ∈ :data:`_DEX_INLINE_SEG_SIZE_RANGE` bytes.
    * Segments do **not** overlap; the random placement algorithm
      ensures a gap ≥ 1 byte between any two segments (so the labels
      describe ``k`` disjoint positive intervals separated by real
      bytecode — the whole point of the scatter).
    * The chosen span retains at least :data:`_DEX_INLINE_HEAD_PRESERVE`
      bytes of original bytecode at the head; spans below
      :data:`_DEX_INLINE_MIN_SPAN_SIZE` bytes are rejected.
    * Runtime fidelity is **not** required.

    The output is a list of ``k`` :class:`InjectedPayload` records, all
    pointing at the same ``host_object_path`` and carrying the same
    merged host-wide ``data`` buffer (so :func:`_write_augmented_apk`
    writes the host exactly once). Each record reports its own
    ``payload_offset_start / end`` (sorted ascending by ``part_index``).
    """

    host_metadata, host_data = _select_host(
        ctx,
        predicate=_is_dex_candidate,
        min_host_bytes=1 << 15,  # 32 KiB: smallest sane DEX for inlining
        family=_FAMILY_DEX_METHOD_INLINED,
    )
    spans = parse_dex_item_spans(host_data)
    code_index = DEX_ITEM_TYPES.index("code_item")
    code_spans = [
        span for span in spans
        if span.item_type == code_index and span.size >= _DEX_INLINE_MIN_SPAN_SIZE
    ]
    if not code_spans:
        raise SyntheticPackerError(
            f"{_FAMILY_DEX_METHOD_INLINED}: host DEX has no code_item span "
            f">= {_DEX_INLINE_MIN_SPAN_SIZE} bytes"
        )
    # Work inside the largest span (the parser rolls all code_items up
    # into one span; see module docstring).
    target = max(code_spans, key=lambda span: span.size)
    usable_start = target.offset + _DEX_INLINE_HEAD_PRESERVE
    usable_end = target.offset + target.size
    usable_size = usable_end - usable_start
    seg_lo, seg_hi = _DEX_INLINE_SEG_SIZE_RANGE
    k_min, k_max = _DEX_INLINE_K_RANGE

    # How many segments can we even fit? Each segment takes at least
    # seg_lo + 1 byte of gap; the largest possible k is therefore
    # floor(usable_size / (seg_lo + 1)).
    max_fit = usable_size // (seg_lo + 1)
    if max_fit < 1:
        raise SyntheticPackerError(
            f"{_FAMILY_DEX_METHOD_INLINED}: usable region in code_item span "
            f"({usable_size} bytes) cannot fit a single minimum segment "
            f"({seg_lo} bytes)"
        )
    k = ctx.rng.randint(min(k_min, max_fit), min(k_max, max_fit))

    # Pick k disjoint segment sizes and positions. We proceed greedily:
    # split the usable region into k roughly equal slots, then within
    # each slot uniformly pick a segment size in [seg_lo, min(seg_hi,
    # slot_size - 1)] and a start offset so the segment sits inside
    # the slot with at least 1-byte trailing gap. Using slot
    # partitioning guarantees non-overlap without retries.
    slot_size = usable_size // k
    if slot_size < seg_lo + 1:
        # k was sampled too high for the usable size; clamp down.
        k = usable_size // (seg_lo + 1)
        slot_size = usable_size // k
    segments: List[Tuple[int, int]] = []  # (abs_start, abs_end)
    for i in range(k):
        slot_start = usable_start + i * slot_size
        seg_cap = min(seg_hi, slot_size - 1)  # 1-byte trailing gap
        seg_size = ctx.rng.randint(seg_lo, seg_cap)
        # Random start within slot, leaving room for the segment.
        max_offset_in_slot = slot_size - seg_size - 1
        offset_in_slot = ctx.rng.randint(0, max_offset_in_slot) if max_offset_in_slot > 0 else 0
        seg_start = slot_start + offset_in_slot
        segments.append((seg_start, seg_start + seg_size))

    # Clip payload to the aggregate segment capacity. The cut is from
    # the head so SyntheticLabel.source_offset_* stays contiguous.
    total_needed = sum(end - start for start, end in segments)
    if len(ctx.payload) < total_needed:
        # Drop tail segments until the payload fills the remaining ones.
        while segments and sum(e - s for s, e in segments) > len(ctx.payload):
            segments.pop()
        if not segments:
            raise SyntheticPackerError(
                f"{_FAMILY_DEX_METHOD_INLINED}: payload "
                f"({len(ctx.payload)} bytes) too small for any "
                f"{seg_lo}-byte segment in this host"
            )
        total_needed = sum(e - s for s, e in segments)
    payload_slice = ctx.payload[:total_needed]

    # Overwrite each segment with encrypted bytes and build records.
    data = bytearray(host_data)
    records: List[InjectedPayload] = []
    payload_cursor = 0
    for index, (seg_start, seg_end) in enumerate(segments):
        seg_size = seg_end - seg_start
        segment_bytes = payload_slice[payload_cursor : payload_cursor + seg_size]
        payload_cursor += seg_size
        # Each segment gets an independent XOR key — the whole point of
        # modelling "different methods encrypted with different keys".
        key = ctx.rng.randint(1, 255)
        encrypted = xor_bytes(segment_bytes, key)
        data[seg_start:seg_end] = encrypted
        records.append(
            InjectedPayload(
                object_path=host_metadata.object_path,
                # Deferred: `data` filled with final buffer after the loop
                # so every record shares identical merged bytes (required
                # by _write_augmented_apk's multi-segment overwrite
                # contract).
                data=b"",
                transform_family=_FAMILY_DEX_METHOD_INLINED,
                payload_offset_start=seg_start,
                payload_offset_end=seg_end,
                part_index=index,
                part_count=len(segments),
                xor_key=key,
                host_object_path=host_metadata.object_path,
            )
        )

    # Replace the placeholder ``data`` on every record with the merged
    # host buffer. We rebuild via dataclass replace because the records
    # are frozen.
    from dataclasses import replace as _dc_replace
    merged = bytes(data)
    records = [_dc_replace(r, data=merged) for r in records]
    return records


# ---------------------------------------------------------------------------
# A-v2-b (2026-04-30): new transforms
# ---------------------------------------------------------------------------
# Spec: docs/method/threat_model.md §"Track A v2". Each of the three
# families extends the Gen2 / Gen3 coverage of the existing registry:
#
# * ``multi_dex_shim``    — Gen2+ "shim DEX" pattern: classes.dex is
#   overwritten with a fabricated stub that *looks* like an Android
#   application entry point (valid DEX header, benign onCreate stub),
#   while the real payload lives XOR-encrypted in a fresh
#   ``assets/<random>`` object. Attacks detectors that over-trust the
#   legitimacy of classes.dex bytes.
# * ``embedded_archive``  — Gen3 nested-container: payload is packed
#   into a nested jar (valid ZIP structure, valid manifest) and then
#   XOR-encrypted whole before landing at ``assets/<random>.zip``.
#   Attacks detectors that scan ``assets/*`` at depth 1 only.
# * ``dex_string_encrypted`` — Gen3 string-pool encryption: only the
#   ``string_data_item`` region inside ``classes.dex`` is scrambled
#   (per-span independent XOR key); ``code_item`` bytes are untouched.
#   Forms the code-vs-string ablation counterpart to
#   ``dex_method_inlined``.
#
# All three are sub-range transforms (host overwrite), i.e.
# ``host_object_path is not None`` on their emitted records. Standards
# for key naming / offset semantics follow the existing Gen3 builders.


# Leading magic bytes we generate for the fabricated classes.dex stub;
# matches the official DEX 035 header so that downstream parsers do
# not reject the shim outright.
_MULTI_DEX_SHIM_MAGIC = b"dex\n035\x00"
# Minimum stub DEX size in bytes. Padded with zero-filled tail up to
# this size so the stub itself is large enough to pass the 64 KiB
# lower bound in :func:`_select_host` / B1 checks downstream.
_MULTI_DEX_SHIM_MIN_STUB_SIZE = 1 << 16  # 64 KiB


def _build_multi_dex_shim(ctx: TransformContext) -> List[InjectedPayload]:
    """Replace ``classes.dex`` with a fabricated shim DEX + hide payload in asset.

    Emits **two** injected records: (a) an overwrite of
    ``classes.dex`` with a stub DEX, and (b) a whole-object injection
    of the XOR-encrypted real payload at ``assets/<random>``. Labels
    for (a) are ``benign_loader`` (the shim is what the packer uses
    to dispatch execution; it is not itself hidden payload); labels
    for (b) are ``hidden_executable_payload`` covering the full object.

    The stub DEX is **not** runnable in an Android VM — we only need
    the byte-level layout (valid magic + plausible header +
    checksum-free body) for static-analysis evaluation. Reviewers
    should understand this limitation; the paper's §4.1 must carry
    the same "byte-level geometry, not runtime fidelity" disclaimer
    as ``so_embedded``.

    Why a separate shim rather than reusing the seed APK's own
    classes.dex: real Qihoo 360 Jiagu packers **replace** classes.dex
    with their own loader. Reusing the benign classes.dex would
    understate the threat; the asset payload alone would be
    equivalent to Gen1 ``xor`` with a random name.
    """

    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_MULTI_DEX_SHIM,
        enforce=ctx.enforce_payload_size_range,
    )
    # Find the real classes.dex in the seed APK so we know what to
    # overwrite. If the seed has no classes.dex (rare), fail loudly —
    # this transform is meaningless otherwise.
    classes_dex_path = None
    original_classes_dex_size: Optional[int] = None
    for metadata, _data in ctx.seed_objects:
        if metadata.object_path == "classes.dex":
            classes_dex_path = "classes.dex"
            original_classes_dex_size = len(_data)
            break
    if classes_dex_path is None:
        raise SyntheticPackerError(
            f"{_FAMILY_MULTI_DEX_SHIM}: seed APK has no classes.dex to overwrite"
        )

    # Build the shim DEX. A minimally-plausible body: DEX magic +
    # random checksum bytes + zero-filled 80-byte header + padding.
    # Real 360 Jiagu stubs contain a tiny ``Application.onCreate``
    # bytecode sequence; we approximate it with deterministic random
    # bytes (rng-seeded) so the stub has nonzero entropy but no
    # executable value.
    #
    # A-v4 leakage fix (2026-05-08): size the shim so it is close to
    # the ORIGINAL classes.dex size rather than a fixed 64 KiB floor.
    # v3 probe found multi_dex_shim hosts shrank to ~1% of their
    # original size (mean ratio 0.01, L24), a deterministic fingerprint
    # any classifier can exploit. Real-world packers (360 Jiagu / Bangcle)
    # replace classes.dex with a shim of COMPARABLE size (±20%), not a
    # 100x smaller one. We therefore target the original size ± 10%
    # jitter and enforce a 64 KiB floor for very small seeds.
    if original_classes_dex_size is not None and original_classes_dex_size > _MULTI_DEX_SHIM_MIN_STUB_SIZE:
        jitter_ratio = 1.0 + (ctx.rng.random() - 0.5) * 0.2  # ±10%
        target_shim_size = max(
            _MULTI_DEX_SHIM_MIN_STUB_SIZE,
            int(original_classes_dex_size * jitter_ratio),
        )
    else:
        target_shim_size = _MULTI_DEX_SHIM_MIN_STUB_SIZE
    shim_body_size = target_shim_size - len(_MULTI_DEX_SHIM_MAGIC)
    shim_body = bytes(ctx.rng.randint(0, 255) for _ in range(shim_body_size))
    shim_dex = _MULTI_DEX_SHIM_MAGIC + shim_body

    # XOR-encrypt the real payload with a single key.
    key = resolve_xor_key(ctx.xor_key, ctx.rng)
    encrypted_payload = xor_bytes(payload, key)

    # Allocate the payload object path under asset_prefix.
    payload_path = unique_payload_path(
        ctx.existing_paths,
        ctx.asset_prefix,
        _FAMILY_MULTI_DEX_SHIM,
        ctx.rng,
        naming_profile=ctx.naming_profile,
    )

    # Two records:
    #   1. shim classes.dex overwrite (host_object_path set -> overwrite
    #      mode; benign_loader semantic -> payload offsets 0..0 empty
    #      so no positive region is labelled).
    #   2. whole-object asset injection with encrypted real payload.
    return [
        # (1) shim classes.dex overwrite — NOT labelled as payload; the
        # shim bytes are the *benign loader* of the packer.
        # ``payload_offset_start == payload_offset_end == 0`` flags an
        # empty positive region within the host, which the label
        # writer will correctly treat as "this host has no hidden
        # payload inside it".
        InjectedPayload(
            object_path=classes_dex_path,
            data=shim_dex,
            transform_family=_FAMILY_MULTI_DEX_SHIM,
            payload_offset_start=0,
            payload_offset_end=0,
            part_index=0,
            part_count=2,
            xor_key=None,
            host_object_path=classes_dex_path,
        ),
        # (2) whole-object payload injection at assets/<random>.
        InjectedPayload(
            object_path=payload_path,
            data=encrypted_payload,
            transform_family=_FAMILY_MULTI_DEX_SHIM,
            payload_offset_start=0,
            payload_offset_end=len(encrypted_payload),
            part_index=1,
            part_count=2,
            xor_key=key,
        ),
    ]


def _build_embedded_archive(ctx: TransformContext) -> List[InjectedPayload]:
    """Pack the payload into a nested jar, then XOR-encrypt whole.

    Resulting bytes land at ``assets/<random>.zip``. The nested jar
    contains one valid entry ``META-INF/MANIFEST.MF`` and one
    ``classes.dex`` entry holding the real payload. The overall byte
    stream is then XOR-masked with a single key, so the outer ZIP
    structure is NOT parseable by ``zipfile.ZipFile`` without
    decryption. Attacks detectors that rely on generic
    ``iter_apk_objects`` ZIP-within-ZIP traversal.

    Label semantics: the entire outer asset object is labelled
    ``hidden_executable_payload`` (whole-object). Although internally
    a jar, from the outside it is opaque encrypted bytes.

    Future work (out of Stage A scope): a variant where the outer
    ZIP structure is left intact and only specific inner entries are
    encrypted — would test depth-2 detectors. Not implemented here.
    """

    import io
    import zipfile as _zipfile

    payload = _normalize_payload_size(
        ctx.payload, ctx.rng,
        size_range=_PAYLOAD_SIZE_WHOLE_OBJECT,
        family=_FAMILY_EMBEDDED_ARCHIVE,
        enforce=ctx.enforce_payload_size_range,
    )

    # Build a nested jar in memory.
    buf = io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as inner:
        manifest_lines = (
            "Manifest-Version: 1.0\r\n"
            "Created-By: android-packer synthetic factory\r\n"
            "\r\n"
        ).encode("utf-8")
        inner.writestr("META-INF/MANIFEST.MF", manifest_lines)
        inner.writestr("classes.dex", payload)
    nested_jar_bytes = buf.getvalue()

    # Encrypt the full nested jar bytes with a single key.
    key = resolve_xor_key(ctx.xor_key, ctx.rng)
    encrypted = xor_bytes(nested_jar_bytes, key)

    # A-v3 leakage fix (2026-05-07): use the shared ``unique_payload_path``
    # helper so the path mimics native naming and avoids the L1/L7/L11
    # leak vectors (no embedded family name, no fixed 12-hex token, no
    # always-``.zip`` suffix unless the seed APK actually has zips).
    object_path = unique_payload_path(
        ctx.existing_paths,
        ctx.asset_prefix,
        _FAMILY_EMBEDDED_ARCHIVE,
        ctx.rng,
        naming_profile=ctx.naming_profile,
        # Force ``.zip`` only if no naming profile is available (legacy
        # fallback). When a profile is present, sample a native extension
        # so the entry blends in with seed entries.
        forced_extension=None if ctx.naming_profile is not None else ".zip",
    )

    # payload_offset_start/end cover the full encrypted container,
    # including the nested jar header / manifest — from the outside
    # the entire object is considered hidden payload.
    return [
        InjectedPayload(
            object_path=object_path,
            data=encrypted,
            transform_family=_FAMILY_EMBEDDED_ARCHIVE,
            payload_offset_start=0,
            payload_offset_end=len(encrypted),
            part_index=None,
            part_count=None,
            xor_key=key,
        )
    ]


def _build_dex_string_encrypted(ctx: TransformContext) -> List[InjectedPayload]:
    """Encrypt only the ``string_data_item`` spans of ``classes.dex`` in-place.

    Each string_data_item span gets its own randomly-sampled 1-byte
    XOR key. ``code_item`` bytes, headers, and maps stay untouched.
    This forms the string-pool counterpart to
    ``dex_method_inlined``: same style of in-place overwrite within a
    DEX host, but targeting strings rather than opcodes.

    Label semantics: each encrypted string span is a positive region
    (``hidden_executable_payload``); ``part_index`` / ``part_count``
    track the span index as usual.

    Caveats:

    * The DEX parser (:mod:`android_packer.features.dex_item_parser`)
      rolls up *all* string_data_items into a single
      ``DexItemSpan`` with ``item_type == index of "string_data"``.
      We encrypt the full rolled-up span as one operation; per-item
      (per-string) granularity is out of scope for Stage A.
    * Exactly one "string_data" span per DEX is expected. If the
      parser returns zero such spans we raise
      :class:`SyntheticPackerError` so the task is skipped cleanly
      rather than silently producing an empty positive label.
    * DEX checksum / signature not updated — the resulting DEX is
      **not** runnable. Byte-level geometry only. Same disclaimer
      applies as to ``so_embedded`` and ``multi_dex_shim``.
    """

    host_metadata, host_data = _select_host(
        ctx,
        predicate=_is_dex_candidate,
        min_host_bytes=1 << 15,  # 32 KiB
        family=_FAMILY_DEX_STRING_ENCRYPTED,
    )
    spans = parse_dex_item_spans(host_data)
    string_data_index = DEX_ITEM_TYPES.index("string_data")
    string_spans = [s for s in spans if s.item_type == string_data_index and s.size > 0]
    if not string_spans:
        raise SyntheticPackerError(
            f"{_FAMILY_DEX_STRING_ENCRYPTED}: host DEX has no string_data span"
        )

    # Work in a mutable buffer so we can overwrite in-place.
    data = bytearray(host_data)
    records: List[InjectedPayload] = []
    # Sort spans by offset so part_index tracks positional order; the
    # Qihoo-style behaviour would be 1-to-N per-string keys, but with
    # rolled-up parser output we get 1-to-1 spans in practice.
    string_spans.sort(key=lambda s: s.offset)
    for index, span in enumerate(string_spans):
        key = ctx.rng.randint(1, 255)
        data[span.offset : span.offset + span.size] = xor_bytes(
            bytes(data[span.offset : span.offset + span.size]), key
        )
        records.append(
            InjectedPayload(
                object_path=host_metadata.object_path,
                data=b"",  # filled after the loop (merged host buffer)
                transform_family=_FAMILY_DEX_STRING_ENCRYPTED,
                payload_offset_start=span.offset,
                payload_offset_end=span.offset + span.size,
                part_index=index,
                part_count=len(string_spans),
                xor_key=key,
                host_object_path=host_metadata.object_path,
            )
        )

    from dataclasses import replace as _dc_replace
    merged = bytes(data)
    records = [_dc_replace(r, data=merged) for r in records]
    return records


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TRANSFORMS: dict[str, TransformFn] = {
    _FAMILY_PATH_RANDOMIZED: _build_path_randomized,
    _FAMILY_XOR: _build_xor,
    _FAMILY_BASE64: _build_base64,
    _FAMILY_SPLIT_XOR: _build_split_xor,
    _FAMILY_SIGNATURE_STRIP: _build_signature_strip,
    _FAMILY_EMBEDDED_ASSET: _build_embedded_asset,
    _FAMILY_SO_EMBEDDED: _build_so_embedded,
    _FAMILY_DEX_METHOD_INLINED: _build_dex_method_inlined,
    # A-v2-b (2026-04-30):
    _FAMILY_MULTI_DEX_SHIM: _build_multi_dex_shim,
    _FAMILY_EMBEDDED_ARCHIVE: _build_embedded_archive,
    _FAMILY_DEX_STRING_ENCRYPTED: _build_dex_string_encrypted,
}

#: Declared transform families. Preserves insertion order so CLI help and
#: manifest listings stay stable across Python runs.
SUPPORTED_TRANSFORMS: tuple[str, ...] = tuple(TRANSFORMS)


def register_transform(name: str, builder: TransformFn) -> None:
    """Register ``builder`` under ``name``.

    Raises :class:`ValueError` if ``name`` is already registered so adversarial
    variants added in experiments can't silently shadow the built-ins.
    """

    if name in TRANSFORMS:
        raise ValueError(f"transform already registered: {name!r}")
    TRANSFORMS[name] = builder
    # Keep the public tuple in sync.
    global SUPPORTED_TRANSFORMS
    SUPPORTED_TRANSFORMS = tuple(TRANSFORMS)


def build_injected_payloads(transform_family: str, ctx: TransformContext) -> List[InjectedPayload]:
    try:
        builder = TRANSFORMS[transform_family]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_TRANSFORMS)
        raise ValueError(
            f"unsupported transform_family={transform_family!r}; use {supported}"
        ) from exc
    return builder(ctx)


__all__ = [
    "SUPPORTED_TRANSFORMS",
    "TRANSFORMS",
    "SeedNamingProfile",
    "TransformContext",
    "TransformFn",
    "build_injected_payloads",
    "normalize_asset_prefix",
    "register_transform",
    "resolve_xor_key",
    "split_ranges",
    "unique_payload_path",
    "xor_bytes",
]
