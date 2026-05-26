"""Byte-window region generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import log2
from typing import Iterator

from android_packer.apkio.objects import ApkObject

# Precomputed translation table mapping every printable byte (tab, LF, CR and
# 0x20..0x7E) to ``b"\x01"`` and everything else to ``b"\x00"``. Combined with
# ``bytes.count`` this lets ``printable_ratio`` be a single C-level pass.
_PRINTABLE_BYTES = bytes(
    [9, 10, 13] + list(range(32, 127))
)
_PRINTABLE_TABLE = bytes(
    1 if byte in _PRINTABLE_BYTES else 0 for byte in range(256)
)


@dataclass(frozen=True)
class Region:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    object_type: str
    offset_start: int
    offset_end: int
    size: int
    sha256: str
    entropy: float
    printable_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def iter_regions(
    metadata: ApkObject,
    data: bytes,
    *,
    window_size: int,
    stride: int,
    min_region_size: int = 1,
    include_tail: bool = True,
) -> Iterator[Region]:
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if stride <= 0:
        raise ValueError("stride must be greater than 0")
    if min_region_size <= 0:
        raise ValueError("min_region_size must be greater than 0")

    starts = list(_window_starts(len(data), window_size, stride, include_tail))
    for index, start in enumerate(starts):
        end = min(start + window_size, len(data))
        chunk = data[start:end]
        if len(chunk) < min_region_size:
            continue
        yield Region(
            apk_id=metadata.apk_id,
            object_id=metadata.object_id,
            region_id=f"{metadata.object_id}:r{index:06d}",
            object_path=metadata.object_path,
            object_type=metadata.object_type,
            offset_start=start,
            offset_end=end,
            size=len(chunk),
            sha256=sha256(chunk).hexdigest(),
            entropy=round(byte_entropy(chunk), 6),
            printable_ratio=round(printable_ratio(chunk), 6),
        )


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    total = len(data)
    # ``collections.Counter`` on a ``bytes`` object runs in C and only iterates
    # once even for multi-megabyte regions.
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return entropy


def printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    # Translate every byte to ``0x00``/``0x01`` and count the 1s; both ops are
    # implemented in C.
    return data.translate(_PRINTABLE_TABLE).count(b"\x01") / len(data)


def _window_starts(
    data_size: int,
    window_size: int,
    stride: int,
    include_tail: bool,
) -> Iterator[int]:
    if data_size <= 0:
        return
    if data_size <= window_size:
        yield 0
        return

    last_start = None
    start = 0
    while start + window_size <= data_size:
        yield start
        last_start = start
        start += stride

    tail_start = data_size - window_size
    if include_tail and tail_start != last_start:
        yield tail_start
