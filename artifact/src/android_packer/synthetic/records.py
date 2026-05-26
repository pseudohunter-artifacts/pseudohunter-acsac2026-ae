"""Synthetic-packer data records.

Kept in a dedicated module so ``synthetic.transforms`` can depend on the
record types without importing ``synthetic.packer`` (which would create a
cycle through ``transforms`` itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from android_packer.labeling import SyntheticLabel


class SyntheticPackerError(RuntimeError):
    """Raised when a synthetic APK cannot be generated."""


@dataclass(frozen=True)
class PayloadSource:
    kind: str
    data: bytes
    source_apk_id: Optional[str]
    source_object_path: str


@dataclass(frozen=True)
class InjectedPayload:
    object_path: str
    data: bytes
    transform_family: str
    payload_offset_start: int
    payload_offset_end: int
    part_index: Optional[int]
    part_count: Optional[int]
    xor_key: Optional[int]
    # ``host_object_path``: when set, the transform is a **sub-range**
    # embedding and the writer must *overwrite* the host ZIP entry rather
    # than create a new one. ``data`` is the full replacement bytes
    # (host_prefix + encrypted_payload [+ host_suffix]) and
    # ``payload_offset_start`` / ``payload_offset_end`` point to the
    # object-local byte range occupied by the payload within ``data``.
    # When ``None``, the transform is a Gen1/Gen2 whole-object injection
    # and ``data`` is written as a brand-new ZIP entry at
    # ``object_path`` (pre-existing behaviour).
    host_object_path: Optional[str] = None


@dataclass(frozen=True)
class SyntheticBuildResult:
    generated_apk_path: Path
    manifest_path: Optional[Path]
    labels_path: Optional[Path]
    manifest: dict
    labels: List[SyntheticLabel]


__all__ = [
    "InjectedPayload",
    "PayloadSource",
    "SyntheticBuildResult",
    "SyntheticPackerError",
]
