"""Byte-level feature extraction for region-granularity classifiers.

The features are deliberately **pure-Python, stdlib-only**. That keeps
``android_packer.features`` usable from any baseline / CLI without
pulling numpy in on a zero-dependency core install. Downstream
consumers that want dense arrays (e.g. sklearn's LogisticRegression)
build them at the edge via :meth:`RegionFeatureVector.to_dense`.

Feature families (each toggleable via :class:`ByteFeatureConfig`):

- ``unigram_histogram``: 256-dim normalised byte frequency vector. The
  single most informative feature family for distinguishing encrypted
  / compressed blobs from plain text or class file data.
- ``bigram_histogram`` (optional, hashed): 256*256 = 65 536 possible
  bigrams; stored with the hashing trick into ``bigram_hash_dim``
  buckets (default 1024) to keep the vector tractable. Disable when
  training a tiny model that would overfit the extra 1k features.
- ``scalars``: Shannon entropy, byte-chunk entropy (mean & std of
  entropy over fixed sub-windows), printable ASCII ratio, zero-byte
  ratio, longest zero run ratio, non-ASCII high-byte ratio,
  duplicate-block ratio (fraction of 16-byte blocks seen more than
  once). Each is deterministic and bounded, so the combined vector
  stays in the same numeric regime as the histogram.

All numeric outputs are plain Python floats and every feature name is
a str, which means vectors can be serialised to JSON for debugging
without further conversion.
"""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ByteFeatureConfig:
    """Toggle and tune each feature family.

    The defaults are chosen for the "n-gram + logistic regression"
    baseline: unigram histogram on (dominant signal), hashed bigram on
    with a modest bucket count, scalar features on. All toggles are
    stable across versions so saved models remain portable.
    """

    include_unigram: bool = True
    include_bigram: bool = True
    bigram_hash_dim: int = 1024
    include_scalars: bool = True
    # Sub-window size used for byte-chunk entropy statistics. 256 is a
    # pragmatic middle ground: small enough that a tampered region
    # with heterogeneous content shows visible variance, large enough
    # that the per-chunk histogram isn't dominated by sampling noise.
    entropy_chunk_size: int = 256
    # Block size used for duplicate-block ratio. 16 catches the padding
    # patterns that block cipher outputs and LZ-style compression
    # streams never produce; larger blocks would miss short repeats.
    duplicate_block_size: int = 16

    def __post_init__(self) -> None:
        if self.bigram_hash_dim <= 0:
            raise ValueError(
                f"bigram_hash_dim must be positive, got {self.bigram_hash_dim}"
            )
        if self.entropy_chunk_size <= 0:
            raise ValueError(
                f"entropy_chunk_size must be positive, got {self.entropy_chunk_size}"
            )
        if self.duplicate_block_size <= 0:
            raise ValueError(
                f"duplicate_block_size must be positive, got "
                f"{self.duplicate_block_size}"
            )


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionFeatureVector:
    """A sparse-by-convention feature vector for a single region.

    ``values`` maps feature name -> float. Missing keys are implicitly
    zero, which matches how sklearn's DictVectorizer treats absent
    entries and lets us leave all-zero unigram buckets off the wire.
    ``order`` preserves the deterministic feature order used when the
    vector was built, so consumers that want a stable dense layout
    (e.g. PyTorch dataloaders) can reproduce it without another pass.
    """

    values: Dict[str, float]
    order: Tuple[str, ...]

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def to_dense(self, feature_order: Optional[Sequence[str]] = None) -> List[float]:
        """Render the vector as a dense list of floats.

        When ``feature_order`` is provided the values are projected into
        that layout (missing keys become 0.0). Otherwise the vector's
        own :attr:`order` is used, which is convenient for smoke tests.
        """

        order = tuple(feature_order) if feature_order is not None else self.order
        return [float(self.values.get(name, 0.0)) for name in order]


# ---------------------------------------------------------------------------
# Feature extraction entry point
# ---------------------------------------------------------------------------


def region_byte_features(
    data: bytes,
    config: Optional[ByteFeatureConfig] = None,
) -> RegionFeatureVector:
    """Compute the full feature vector for a region's raw bytes.

    ``data`` may be empty; in that case every feature is 0.0 and the
    returned vector is still a valid ``RegionFeatureVector``. This
    mirrors how an upstream slice past end-of-object would collapse to
    zero length without raising, which keeps the feature pipeline
    composable with streaming readers that might over-read.
    """

    cfg = config or ByteFeatureConfig()

    values: Dict[str, float] = {}
    order: List[str] = []

    if cfg.include_unigram:
        _fill_unigram(data, values, order)
    if cfg.include_bigram:
        _fill_bigram(data, cfg.bigram_hash_dim, values, order)
    if cfg.include_scalars:
        _fill_scalars(data, cfg, values, order)

    return RegionFeatureVector(values=values, order=tuple(order))


# ---------------------------------------------------------------------------
# Individual feature builders (private)
# ---------------------------------------------------------------------------


def _fill_unigram(
    data: bytes,
    values: Dict[str, float],
    order: List[str],
) -> None:
    # Always reserve the 256 slots in the output order so that a
    # zero-byte region still produces a stable dense layout. Only
    # non-zero counts populate ``values`` to keep JSON payloads small.
    total = len(data)
    counts = Counter(data)
    inv_total = 1.0 / total if total else 0.0
    for b in range(256):
        name = f"u{b:03d}"
        order.append(name)
        c = counts.get(b)
        if c:
            values[name] = c * inv_total


def _fill_bigram(
    data: bytes,
    hash_dim: int,
    values: Dict[str, float],
    order: List[str],
) -> None:
    # Reserve deterministic bucket names first so the output order is
    # independent of data length.
    bucket_names = [f"b{i:04d}" for i in range(hash_dim)]
    order.extend(bucket_names)

    if len(data) < 2:
        return

    bucket_counts: Dict[int, int] = {}
    # Vectorise the bigram walk with zip on memoryview slices; this is
    # dramatically faster than a Python for-loop over indices on large
    # regions and is still stdlib-only.
    first = memoryview(data)[:-1]
    second = memoryview(data)[1:]
    for a, b in zip(first, second):
        # A classic, order-sensitive hash: (a << 8 | b) modulo dim.
        # Stable across Python versions and processes because it does
        # NOT use Python's randomised hash().
        idx = ((a << 8) | b) % hash_dim
        bucket_counts[idx] = bucket_counts.get(idx, 0) + 1

    total = len(data) - 1
    inv_total = 1.0 / total if total else 0.0
    for idx, c in bucket_counts.items():
        values[bucket_names[idx]] = c * inv_total


def _fill_scalars(
    data: bytes,
    config: ByteFeatureConfig,
    values: Dict[str, float],
    order: List[str],
) -> None:
    # Every scalar feature is always emitted (even when zero) because
    # the feature count is tiny and downstream consumers prefer a
    # guaranteed-present key to a defaulted zero.
    scalar_names = [
        "s_entropy",
        "s_chunk_entropy_mean",
        "s_chunk_entropy_std",
        "s_printable_ratio",
        "s_zero_ratio",
        "s_longest_zero_run_ratio",
        "s_high_byte_ratio",
        "s_duplicate_block_ratio",
        "s_length_log1p",
    ]
    order.extend(scalar_names)

    total = len(data)
    if total == 0:
        for name in scalar_names:
            values[name] = 0.0
        return

    counts = Counter(data)

    # Shannon entropy (whole region).
    entropy = 0.0
    inv_total = 1.0 / total
    for c in counts.values():
        p = c * inv_total
        entropy -= p * math.log2(p)
    values["s_entropy"] = round(entropy, 6)

    # Chunk entropy mean / std.
    chunk = config.entropy_chunk_size
    if total >= chunk:
        mem = memoryview(data)
        chunk_entropies: List[float] = []
        for start in range(0, total - chunk + 1, chunk):
            sub_counts = Counter(bytes(mem[start : start + chunk]))
            ent = 0.0
            inv_chunk = 1.0 / chunk
            for c in sub_counts.values():
                p = c * inv_chunk
                ent -= p * math.log2(p)
            chunk_entropies.append(ent)
        if chunk_entropies:
            mean = sum(chunk_entropies) / len(chunk_entropies)
            var = sum((e - mean) ** 2 for e in chunk_entropies) / len(chunk_entropies)
            values["s_chunk_entropy_mean"] = round(mean, 6)
            values["s_chunk_entropy_std"] = round(math.sqrt(var), 6)
        else:
            values["s_chunk_entropy_mean"] = 0.0
            values["s_chunk_entropy_std"] = 0.0
    else:
        # Fall back to the whole-region entropy and zero variance.
        values["s_chunk_entropy_mean"] = values["s_entropy"]
        values["s_chunk_entropy_std"] = 0.0

    # Printable / zero / high-byte ratios.
    printable = sum(
        c for b, c in counts.items() if 0x20 <= b < 0x7F or b in (0x09, 0x0A, 0x0D)
    )
    values["s_printable_ratio"] = round(printable * inv_total, 6)
    values["s_zero_ratio"] = round(counts.get(0, 0) * inv_total, 6)
    values["s_high_byte_ratio"] = round(
        sum(c for b, c in counts.items() if b >= 0x80) * inv_total,
        6,
    )

    # Longest run of 0x00, expressed as a fraction of the region size.
    longest = 0
    current = 0
    for b in data:
        if b == 0:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    values["s_longest_zero_run_ratio"] = round(longest * inv_total, 6)

    # Duplicate 16-byte block ratio. Compressed / encrypted streams
    # (very) rarely repeat 16-byte windows; padding, uncompressed
    # bitmaps and class constants do. The ratio is capped at 1.0.
    block = config.duplicate_block_size
    if total >= block * 2:
        mem = memoryview(data)
        block_counts: Dict[bytes, int] = {}
        block_n = total // block
        for i in range(block_n):
            key = bytes(mem[i * block : (i + 1) * block])
            block_counts[key] = block_counts.get(key, 0) + 1
        duplicates = sum(c - 1 for c in block_counts.values() if c > 1)
        values["s_duplicate_block_ratio"] = round(duplicates / block_n, 6)
    else:
        values["s_duplicate_block_ratio"] = 0.0

    # Length-aware calibration feature. log1p keeps the value in a
    # sane range for linear models even when region sizes span orders
    # of magnitude. Divided by log1p(1 MiB) so the typical range is
    # roughly [0, 1]; the LR model can still scale it via its weight.
    values["s_length_log1p"] = round(
        math.log1p(total) / math.log1p(1 << 20),
        6,
    )


# ---------------------------------------------------------------------------
# APK byte access with LRU cache on full object contents
# ---------------------------------------------------------------------------


def extract_region_bytes(
    apk_path: Path,
    object_path: str,
    offset_start: int,
    offset_end: int,
) -> bytes:
    """Read a single region's bytes out of an APK.

    This is the uncached primitive; prefer :class:`ObjectByteLoader`
    when multiple regions share an ``(apk_path, object_path)`` pair,
    which is the common case (a 4 KiB-stride slicing of a 1 MiB
    object creates hundreds of regions per object).
    """

    with zipfile.ZipFile(apk_path, "r") as archive:
        with archive.open(object_path, "r") as member:
            # Fast path: skip bytes we don't need. ``zipfile`` has no
            # seek on the decompressed stream, so we have to read and
            # discard the prefix, then read the region window.
            _skip(member, offset_start)
            length = max(0, offset_end - offset_start)
            return member.read(length)


def _skip(stream, amount: int, chunk: int = 1 << 16) -> None:
    # Stdlib zipfile's decompressing stream does not implement seek(),
    # so emulate it with a bounded read loop. ``amount`` is small in
    # practice because objects are the APK zip-entry boundary.
    remaining = amount
    while remaining > 0:
        buf = stream.read(min(chunk, remaining))
        if not buf:
            return
        remaining -= len(buf)


class ObjectByteLoader:
    """Per-APK / per-object byte cache.

    Opens each ``(apk_path, object_path)`` pair at most once per
    instance, holds the decompressed bytes in memory, and serves
    arbitrary ``[offset_start, offset_end)`` slices off that buffer.

    This is the "strategy A" from the batch E design note: cheap
    enough for MVP, easily replaced by a precomputed features file
    once regions number in the millions.
    """

    def __init__(self, cache_size: int = 64) -> None:
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        # Dynamically build an lru_cache-wrapped reader bound to this
        # instance so that different loader instances have independent
        # caches (important for tests, and for long-running pipelines
        # that want to reset the working set between phases).
        self._read_object = lru_cache(maxsize=cache_size)(self._read_object_uncached)

    @staticmethod
    def _read_object_uncached(apk_path_str: str, object_path: str) -> bytes:
        with zipfile.ZipFile(apk_path_str, "r") as archive:
            with archive.open(object_path, "r") as member:
                return member.read()

    def region_bytes(
        self,
        apk_path: Path,
        object_path: str,
        offset_start: int,
        offset_end: int,
    ) -> bytes:
        data = self._read_object(str(apk_path), object_path)
        if offset_end <= offset_start:
            return b""
        # Bound the slice to the object length so that a stale region
        # metadata row (object got re-generated with a shorter length)
        # collapses to a prefix rather than producing garbage bytes.
        return data[offset_start:offset_end]

    def cache_info(self):  # pragma: no cover - thin passthrough for debugging
        return self._read_object.cache_info()

    def cache_clear(self) -> None:
        self._read_object.cache_clear()


__all__ = [
    "ByteFeatureConfig",
    "ObjectByteLoader",
    "RegionFeatureVector",
    "extract_region_bytes",
    "region_byte_features",
]
