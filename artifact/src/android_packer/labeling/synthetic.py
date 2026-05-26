"""Synthetic payload label records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


HIDDEN_EXECUTABLE_PAYLOAD = "hidden_executable_payload"


@dataclass(frozen=True)
class SyntheticLabel:
    """Strong label for a synthetic packed APK payload.

    ``offset_start`` / ``offset_end`` are **object-local** byte offsets: they
    describe the byte range inside the injected APK object (``object_path``)
    that the payload occupies, not offsets into the raw APK file. They align
    with the ``offset_start`` / ``offset_end`` emitted by region metadata so
    downstream joins (``labeling.alignment``) only need to compare intervals
    with the same reference frame.

    ``source_offset_start`` / ``source_offset_end`` describe a byte range in
    the *pre-transform* payload (e.g. a DEX); they are only meaningful for
    transform families that split or slice the payload (``split_xor``).
    """

    apk_id: str
    object_path: str
    offset_start: int
    offset_end: int
    label: str
    transform_family: str
    payload_sha256: str
    source_apk_id: Optional[str] = None
    source_object_path: Optional[str] = None
    transformed_sha256: Optional[str] = None
    part_index: Optional[int] = None
    part_count: Optional[int] = None
    source_offset_start: Optional[int] = None
    source_offset_end: Optional[int] = None

    def to_dict(self) -> dict:
        row = asdict(self)
        return {key: value for key, value in row.items() if value is not None}
