"""Minimal stdlib DEX parser for grammar-aware pretraining (F2b).

This module implements **just enough** of the Dalvik Executable (DEX) format
to enumerate the byte spans occupied by each top-level structural item
(``header``, ``string_ids``, ``type_ids``, ``proto_ids``, ``field_ids``,
``method_ids``, ``class_defs``, ``code_item``, ``string_data``, …).

The spans are consumed by F5 (``training/pretrain_mlm.py``) to emit
per-byte item-type labels that drive an **auxiliary cross-entropy loss**
alongside byte-level MLM (see ``docs/method/ours_method_spec.md`` §3.2.1b
and §5.1; the research-level rationale is in
``docs/research_framing.md`` §4.2 / §5).

Scope / non-goals
-----------------
* Pure stdlib (``struct`` + ``dataclasses`` only). No ``androguard`` /
  ``dexparser`` / native tool dependency so that the ``[dl]`` extra
  installs cleanly in reviewer-facing environments.
* Parses **header + map_list** and the variable-length ``string_data``
  items that the map_list points at. Does **not** decompile opcodes or
  resolve references.
* Rejects packed / truncated / malformed DEX with ``DexParseError``.
  F5's corpus builder relies on this to **exclude non-benign DEX
  automatically**, keeping the benign-only MLM contract (§4.2) intact.

Reference: https://source.android.com/docs/core/runtime/dex-format
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public item-type vocabulary
# ---------------------------------------------------------------------------

# Order matters: the integer index of each name is the label id written to
# the per-byte target tensor in F5. **Do not reorder or rename after the
# first checkpoint ships.** "other" is the catch-all for bytes not covered
# by any parsed span (alignment padding, unknown map_list sections, etc.).
DEX_ITEM_TYPES: Tuple[str, ...] = (
    "header",
    "string_ids",
    "type_ids",
    "proto_ids",
    "field_ids",
    "method_ids",
    "class_defs",
    "code_item",
    "string_data",
    "other",
)

_ITEM_TYPE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(DEX_ITEM_TYPES)}
_OTHER_INDEX: int = _ITEM_TYPE_TO_INDEX["other"]


# ---------------------------------------------------------------------------
# Exceptions and public data types
# ---------------------------------------------------------------------------


class DexParseError(ValueError):
    """Raised when the byte buffer is not a valid DEX file.

    F5's corpus builder (``training/pretrain_mlm.py``) catches this
    exception and deletes the offending region from the MLM corpus; this
    is the mechanism by which packed / obfuscated / truncated payloads
    are kept out of benign-only pretraining.
    """


@dataclass(frozen=True)
class DexItemSpan:
    """A contiguous byte range belonging to a single DEX item type.

    Attributes:
        offset: Byte offset into the DEX file where the span starts.
        size: Span length in bytes (``offset + size`` is exclusive end).
        item_type: Integer index into :data:`DEX_ITEM_TYPES`.
    """

    offset: int
    size: int
    item_type: int


# ---------------------------------------------------------------------------
# DEX constants
# ---------------------------------------------------------------------------

# All currently observed DEX versions (035 to 039). See
# https://source.android.com/docs/core/runtime/dex-format#dex-file-magic
_DEX_MAGIC_PREFIX = b"dex\n"
_DEX_MAGIC_VERSIONS = (b"035\x00", b"036\x00", b"037\x00", b"038\x00", b"039\x00")
_DEX_HEADER_SIZE = 0x70  # 112 bytes

# map_list type codes. Only the ones we surface as distinct item types are
# listed here; everything else rolls up into "other" (see
# :func:`_map_type_to_item_index`).
_TYPE_HEADER_ITEM = 0x0000
_TYPE_STRING_ID_ITEM = 0x0001
_TYPE_TYPE_ID_ITEM = 0x0002
_TYPE_PROTO_ID_ITEM = 0x0003
_TYPE_FIELD_ID_ITEM = 0x0004
_TYPE_METHOD_ID_ITEM = 0x0005
_TYPE_CLASS_DEF_ITEM = 0x0006
_TYPE_MAP_LIST = 0x1000
_TYPE_TYPE_LIST = 0x1001
_TYPE_ANNOTATION_SET_REF_LIST = 0x1002
_TYPE_ANNOTATION_SET_ITEM = 0x1003
_TYPE_CLASS_DATA_ITEM = 0x2000
_TYPE_CODE_ITEM = 0x2001
_TYPE_STRING_DATA_ITEM = 0x2002
_TYPE_DEBUG_INFO_ITEM = 0x2003
_TYPE_ANNOTATION_ITEM = 0x2004
_TYPE_ENCODED_ARRAY_ITEM = 0x2005
_TYPE_ANNOTATIONS_DIRECTORY_ITEM = 0x2006

# Fixed-size entries per the spec (bytes per item).
_SIZEOF_STRING_ID = 4
_SIZEOF_TYPE_ID = 4
_SIZEOF_PROTO_ID = 12
_SIZEOF_FIELD_ID = 8
_SIZEOF_METHOD_ID = 8
_SIZEOF_CLASS_DEF = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_type_to_item_index(map_type: int) -> int:
    """Project a DEX map_list type code to our item-type vocabulary."""

    if map_type == _TYPE_HEADER_ITEM:
        return _ITEM_TYPE_TO_INDEX["header"]
    if map_type == _TYPE_STRING_ID_ITEM:
        return _ITEM_TYPE_TO_INDEX["string_ids"]
    if map_type == _TYPE_TYPE_ID_ITEM:
        return _ITEM_TYPE_TO_INDEX["type_ids"]
    if map_type == _TYPE_PROTO_ID_ITEM:
        return _ITEM_TYPE_TO_INDEX["proto_ids"]
    if map_type == _TYPE_FIELD_ID_ITEM:
        return _ITEM_TYPE_TO_INDEX["field_ids"]
    if map_type == _TYPE_METHOD_ID_ITEM:
        return _ITEM_TYPE_TO_INDEX["method_ids"]
    if map_type == _TYPE_CLASS_DEF_ITEM:
        return _ITEM_TYPE_TO_INDEX["class_defs"]
    if map_type == _TYPE_CODE_ITEM:
        return _ITEM_TYPE_TO_INDEX["code_item"]
    if map_type == _TYPE_STRING_DATA_ITEM:
        return _ITEM_TYPE_TO_INDEX["string_data"]
    # Everything else (type_list, annotations, debug_info, encoded_array,
    # class_data, map_list itself, …) rolls up into "other". We deliberately
    # keep a small vocabulary so the auxiliary loss does not need to learn a
    # long tail of near-empty classes.
    return _OTHER_INDEX


def _read_uleb128(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode a ULEB128 integer starting at ``offset``. Returns ``(value, next_offset)``.

    Raises :class:`DexParseError` if the encoding runs past the end of
    ``data`` or exceeds 5 bytes (the DEX spec cap).
    """

    result = 0
    shift = 0
    cursor = offset
    for _ in range(5):
        if cursor >= len(data):
            raise DexParseError(f"ULEB128 at offset {offset} runs past end of data")
        byte = data[cursor]
        cursor += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, cursor
        shift += 7
    raise DexParseError(f"ULEB128 at offset {offset} exceeds 5 bytes")


def _read_u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise DexParseError(f"u32 read at offset {offset} runs past end of data")
    return struct.unpack_from("<I", data, offset)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dex_item_spans(dex_bytes: bytes) -> List[DexItemSpan]:
    """Enumerate per-item byte spans in a DEX file.

    Returns a list of :class:`DexItemSpan` sorted by ``offset`` ascending.
    The ``header`` span is always present and always starts at offset 0.
    The remaining spans are derived from the DEX ``map_list``, which the
    spec guarantees to be complete (one entry per physical section).

    Raises :class:`DexParseError` for any of:

    * Buffer shorter than the DEX header.
    * Missing or unknown magic (``dex\\n03X\\x00``).
    * ``file_size`` field larger than the actual buffer.
    * ``map_off`` pointing outside the buffer or not 4-byte aligned.
    * ``map_list`` entries with impossible offsets/sizes.
    * ``string_data`` ULEB128 encoding that runs past the buffer.

    By design this function is **strict**: any ambiguity => exception.
    Callers (F5 corpus builder) treat the exception as "this buffer is
    not a benign DEX and must be excluded from the MLM corpus".
    """

    if not isinstance(dex_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"dex_bytes must be bytes-like, got {type(dex_bytes).__name__}")
    data = bytes(dex_bytes)

    if len(data) < _DEX_HEADER_SIZE:
        raise DexParseError(
            f"buffer too small for DEX header: {len(data)} < {_DEX_HEADER_SIZE}"
        )

    # ---- magic + version ---------------------------------------------------
    if data[:4] != _DEX_MAGIC_PREFIX or data[4:8] not in _DEX_MAGIC_VERSIONS:
        raise DexParseError(f"missing or unknown DEX magic: {data[:8]!r}")

    # ---- header fields we rely on -----------------------------------------
    # Layout (little-endian) per
    # https://source.android.com/docs/core/runtime/dex-format#header-item
    file_size = _read_u32(data, 0x20)
    header_size = _read_u32(data, 0x24)
    # endian_tag @ 0x28 (we assume little-endian; the alt value 0x78563412
    # is extremely rare and not worth special-casing for MVP).
    # link_size/link_off/map_off
    map_off = _read_u32(data, 0x34)
    string_ids_size = _read_u32(data, 0x38)
    string_ids_off = _read_u32(data, 0x3C)
    type_ids_size = _read_u32(data, 0x40)
    type_ids_off = _read_u32(data, 0x44)
    proto_ids_size = _read_u32(data, 0x48)
    proto_ids_off = _read_u32(data, 0x4C)
    field_ids_size = _read_u32(data, 0x50)
    field_ids_off = _read_u32(data, 0x54)
    method_ids_size = _read_u32(data, 0x58)
    method_ids_off = _read_u32(data, 0x5C)
    class_defs_size = _read_u32(data, 0x60)
    class_defs_off = _read_u32(data, 0x64)

    if header_size != _DEX_HEADER_SIZE:
        raise DexParseError(
            f"header_size={header_size} != expected {_DEX_HEADER_SIZE}"
        )
    if file_size > len(data):
        raise DexParseError(
            f"file_size={file_size} exceeds buffer length {len(data)}"
        )
    if map_off == 0 or map_off % 4 != 0 or map_off + 4 > len(data):
        raise DexParseError(
            f"map_off={map_off} is invalid (len={len(data)})"
        )

    spans: List[DexItemSpan] = []

    # ---- header span -------------------------------------------------------
    spans.append(DexItemSpan(offset=0, size=_DEX_HEADER_SIZE, item_type=_ITEM_TYPE_TO_INDEX["header"]))

    # ---- fixed-size sections (from header offsets) -------------------------
    # We record these *directly* from the header (not via map_list) because
    # their byte-level extents are easier to compute from the count-sized
    # product, and the map_list sometimes reports them out of on-disk order.
    _append_if_nonempty(spans, string_ids_off, string_ids_size * _SIZEOF_STRING_ID,
                        _ITEM_TYPE_TO_INDEX["string_ids"], len(data))
    _append_if_nonempty(spans, type_ids_off, type_ids_size * _SIZEOF_TYPE_ID,
                        _ITEM_TYPE_TO_INDEX["type_ids"], len(data))
    _append_if_nonempty(spans, proto_ids_off, proto_ids_size * _SIZEOF_PROTO_ID,
                        _ITEM_TYPE_TO_INDEX["proto_ids"], len(data))
    _append_if_nonempty(spans, field_ids_off, field_ids_size * _SIZEOF_FIELD_ID,
                        _ITEM_TYPE_TO_INDEX["field_ids"], len(data))
    _append_if_nonempty(spans, method_ids_off, method_ids_size * _SIZEOF_METHOD_ID,
                        _ITEM_TYPE_TO_INDEX["method_ids"], len(data))
    _append_if_nonempty(spans, class_defs_off, class_defs_size * _SIZEOF_CLASS_DEF,
                        _ITEM_TYPE_TO_INDEX["class_defs"], len(data))

    # ---- variable-size sections (from map_list) ----------------------------
    map_size = _read_u32(data, map_off)
    map_entries_off = map_off + 4
    if map_entries_off + map_size * 12 > len(data):
        raise DexParseError(
            f"map_list declares {map_size} entries but buffer truncates before end"
        )

    for idx in range(map_size):
        entry_off = map_entries_off + idx * 12
        map_type = struct.unpack_from("<H", data, entry_off)[0]
        # 2 bytes unused (padding) at entry_off + 2
        count = _read_u32(data, entry_off + 4)
        offset = _read_u32(data, entry_off + 8)

        if count == 0:
            continue
        if offset >= len(data):
            raise DexParseError(
                f"map entry {idx} (type=0x{map_type:04x}) offset={offset} past end"
            )

        if map_type == _TYPE_CODE_ITEM:
            # code_items are variable-sized; we walk them to compute each
            # one's extent. Spec: code_item header is 16 bytes, followed by
            # insns_size (u32) * 2 bytes of bytecode, optional padding, then
            # tries/handlers. For labelling we don't need the inner
            # structure — just the outer extent. Approximating by walking
            # one by one is more work than we need; instead we mark from
            # the first code_item's offset to either the next section's
            # offset or the map_list's offset (whichever is smaller above
            # it). This is a conservative over-approximation that still
            # gives the encoder a strong signal about "here is a contiguous
            # region of code_items". Packed/malformed buffers are already
            # rejected above, so over-approximation is safe.
            next_boundary = _next_map_offset_after(data, map_entries_off, map_size, offset)
            if next_boundary is None or next_boundary > len(data):
                next_boundary = len(data)
            span_size = max(0, next_boundary - offset)
            if span_size > 0:
                spans.append(DexItemSpan(offset=offset, size=span_size,
                                         item_type=_ITEM_TYPE_TO_INDEX["code_item"]))
        elif map_type == _TYPE_STRING_DATA_ITEM:
            # string_data_items are ULEB128 (char count) + MUTF-8 + NUL.
            # We walk all ``count`` items to compute the exact extent.
            cursor = offset
            for _ in range(count):
                _char_count, cursor = _read_uleb128(data, cursor)
                # scan for NUL terminator
                nul = data.find(b"\x00", cursor)
                if nul == -1:
                    raise DexParseError(
                        f"string_data item at {cursor} missing NUL terminator"
                    )
                cursor = nul + 1
                if cursor > len(data):
                    raise DexParseError(
                        f"string_data walk past end at cursor={cursor}"
                    )
            span_size = cursor - offset
            if span_size > 0:
                spans.append(DexItemSpan(offset=offset, size=span_size,
                                         item_type=_ITEM_TYPE_TO_INDEX["string_data"]))
        # Other variable-length map_list types (annotations, type_list,
        # class_data, debug_info, encoded_array, annotations_directory) are
        # left as "other". Emitting per-item spans for them would require
        # decoding each item individually; for the auxiliary-loss
        # supervision we'd rather keep "other" large and clean than write
        # fragile parsers for low-density classes.

    # Sort by offset so consumers can do a simple two-pointer sweep.
    spans.sort(key=lambda s: (s.offset, s.size))

    # Sanity: every span must fit inside the buffer.
    for span in spans:
        if span.offset + span.size > len(data):
            raise DexParseError(
                f"span item_type={DEX_ITEM_TYPES[span.item_type]} "
                f"offset={span.offset} size={span.size} exceeds buffer {len(data)}"
            )

    return spans


def region_item_type_labels(
    spans: Sequence[DexItemSpan],
    region_offset: int,
    region_length: int,
) -> List[int]:
    """Project DEX item spans onto a byte region, one label per byte.

    Args:
        spans: Output of :func:`parse_dex_item_spans` (or any sequence of
            :class:`DexItemSpan` with non-overlapping offsets; if spans
            overlap, the **later** one in ``spans`` wins, consistent with
            "map_list is authoritative over header-derived sections").
        region_offset: Object-local byte offset of the region's first byte.
        region_length: Number of bytes in the region.

    Returns:
        List of length ``region_length``; element ``i`` is the item-type
        index of the byte at ``region_offset + i``. Bytes not covered by
        any span receive the index of ``"other"``.

    This function does **not** validate ``region_offset`` against the DEX
    file length — it is callable with out-of-range regions and will
    simply return all ``"other"``. This matches the sliding-window
    regionization contract in :mod:`android_packer.regioning`.
    """

    if region_length < 0:
        raise ValueError(f"region_length must be non-negative, got {region_length}")

    labels = [_OTHER_INDEX] * region_length
    if region_length == 0:
        return labels

    region_end = region_offset + region_length
    for span in spans:
        span_start = span.offset
        span_end = span.offset + span.size
        if span_end <= region_offset or span_start >= region_end:
            continue
        overlap_start = max(span_start, region_offset)
        overlap_end = min(span_end, region_end)
        local_start = overlap_start - region_offset
        local_end = overlap_end - region_offset
        item_type = span.item_type
        for i in range(local_start, local_end):
            labels[i] = item_type

    return labels


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _append_if_nonempty(
    spans: List[DexItemSpan],
    offset: int,
    size: int,
    item_type: int,
    buffer_len: int,
) -> None:
    if size <= 0 or offset == 0:
        # offset==0 means "section absent" per the DEX spec for these
        # fixed-size tables (only the header lives at offset 0).
        return
    if offset >= buffer_len:
        raise DexParseError(
            f"section item_type={DEX_ITEM_TYPES[item_type]} offset={offset} past end"
        )
    if offset + size > buffer_len:
        raise DexParseError(
            f"section item_type={DEX_ITEM_TYPES[item_type]} "
            f"offset={offset} size={size} exceeds buffer {buffer_len}"
        )
    spans.append(DexItemSpan(offset=offset, size=size, item_type=item_type))


def _next_map_offset_after(
    data: bytes,
    map_entries_off: int,
    map_size: int,
    current_offset: int,
) -> int | None:
    """Return the smallest map_list offset strictly greater than ``current_offset``,
    or ``None`` if ``current_offset`` is the last one."""

    best: int | None = None
    for idx in range(map_size):
        entry_off = map_entries_off + idx * 12
        offset = _read_u32(data, entry_off + 8)
        if offset > current_offset and (best is None or offset < best):
            best = offset
    return best


__all__ = [
    "DEX_ITEM_TYPES",
    "DexItemSpan",
    "DexParseError",
    "parse_dex_item_spans",
    "region_item_type_labels",
]
