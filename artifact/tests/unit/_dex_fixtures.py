"""Shared fixtures for DEX-level unit tests.

These helpers live in ``tests/unit/`` — **not** in the main package —
because they exist solely to support test code. They must stay
self-contained (stdlib only) so that `[dl]` and `[metrics]` extras are
not required to run the core test suite.

The primary export is :func:`build_minimal_dex`, which returns a byte
buffer that passes :func:`android_packer.features.dex_item_parser.parse_dex_item_spans`
and exposes at least the five item types required by F2b's verification
matrix: ``header``, ``string_ids``, ``type_ids``, ``method_ids``,
``class_defs``, ``code_item``, and ``string_data``.

Implementation notes
--------------------
We hand-assemble the DEX rather than pulling in ``dx`` / ``d8`` because:

1. The F2b contract is "parse header + map_list only". A real compiled
   DEX contains dozens of ancillary sections (annotations, debug_info,
   encoded_array, …) that the parser intentionally rolls up into
   ``"other"``. Keeping the fixture minimal makes assertions about item
   coverage easy to write and easy to audit.
2. Tests must not depend on an Android build toolchain.
3. Keeping the assembly in Python (~100 loc) means the fixture is
   reviewable alongside the parser — both use the same constants and
   the same ``struct`` layout, so bugs on either side surface during
   code review rather than at runtime.

The checksum / signature fields are left zero. The parser ignores them,
and so does `dalvikvm` when loading DEX from memory in lenient mode, so
this is a safe shortcut for unit tests. Do **not** use this fixture as
a benchmark for real-world DEX density.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Tuple


DEX_MAGIC_035 = b"dex\n035\x00"
DEX_HEADER_SIZE = 0x70


# Map-list type codes we emit in the fixture.
_TYPE_HEADER_ITEM = 0x0000
_TYPE_STRING_ID_ITEM = 0x0001
_TYPE_TYPE_ID_ITEM = 0x0002
_TYPE_PROTO_ID_ITEM = 0x0003
_TYPE_FIELD_ID_ITEM = 0x0004
_TYPE_METHOD_ID_ITEM = 0x0005
_TYPE_CLASS_DEF_ITEM = 0x0006
_TYPE_MAP_LIST = 0x1000
_TYPE_CODE_ITEM = 0x2001
_TYPE_STRING_DATA_ITEM = 0x2002


@dataclass
class MinimalDexLayout:
    """Offsets and sizes chosen when :func:`build_minimal_dex` ran.

    Exposed so tests can assert on exact span positions without having to
    re-derive them from the returned bytes.
    """

    total_size: int
    string_ids_off: int
    string_ids_count: int
    type_ids_off: int
    type_ids_count: int
    proto_ids_off: int
    proto_ids_count: int
    field_ids_off: int
    field_ids_count: int
    method_ids_off: int
    method_ids_count: int
    class_defs_off: int
    class_defs_count: int
    code_items_off: int
    code_items_count: int
    string_data_off: int
    string_data_count: int
    map_off: int


def build_minimal_dex(
    *,
    num_strings: int = 3,
    num_types: int = 2,
    num_protos: int = 1,
    num_fields: int = 1,
    num_methods: int = 2,
    num_classes: int = 1,
    num_code_items: int = 1,
) -> Tuple[bytes, MinimalDexLayout]:
    """Build a minimal DEX buffer that :mod:`dex_item_parser` can parse.

    Returns ``(dex_bytes, layout)``; ``layout`` is a :class:`MinimalDexLayout`
    describing where each section was placed so tests can assert on
    exact offsets.
    """

    # --- section payloads --------------------------------------------------

    # Placeholder fixed-size tables; the parser only needs their total
    # byte extent, not the contents, so we zero-fill each entry.
    string_ids_blob = b"\x00" * (num_strings * 4)
    type_ids_blob = b"\x00" * (num_types * 4)
    proto_ids_blob = b"\x00" * (num_protos * 12)
    field_ids_blob = b"\x00" * (num_fields * 8)
    method_ids_blob = b"\x00" * (num_methods * 8)
    class_defs_blob = b"\x00" * (num_classes * 32)

    # string_data: ULEB128(count) + MUTF-8 bytes + NUL terminator.
    string_data_chunks: List[bytes] = []
    for i in range(num_strings):
        name = f"s{i}".encode("ascii")
        string_data_chunks.append(bytes([len(name)]) + name + b"\x00")
    string_data_blob = b"".join(string_data_chunks)

    # code_items: 16-byte header (registers_size / ins_size / outs_size /
    # tries_size / debug_info_off / insns_size) + 2 bytes of insns for
    # ``return-void``. We keep every code_item at 16 + 4 = 20 bytes (4 = 2
    # bytes of insns + 2 bytes of trailing pad to keep 4-byte alignment
    # for the next item).
    single_code_item = (
        struct.pack("<HHHHII", 1, 0, 0, 0, 0, 1)  # registers=1, insns_size=1 code unit (2 bytes)
        + struct.pack("<H", 0x000E)  # OP_RETURN_VOID (00 0E)
        + b"\x00\x00"                 # 2 bytes of padding for alignment
    )
    assert len(single_code_item) == 20
    code_items_blob = single_code_item * num_code_items

    # --- section layout (offsets) -----------------------------------------
    # Order on disk: header, string_ids, type_ids, proto_ids, field_ids,
    # method_ids, class_defs, code_items, string_data, map_list. Each
    # fixed-size section starts on a 4-byte aligned offset; the trailing
    # variable sections also align to 4.
    cursor = DEX_HEADER_SIZE
    string_ids_off = cursor
    cursor += len(string_ids_blob)
    type_ids_off = cursor
    cursor += len(type_ids_blob)
    proto_ids_off = cursor
    cursor += len(proto_ids_blob)
    field_ids_off = cursor
    cursor += len(field_ids_blob)
    method_ids_off = cursor
    cursor += len(method_ids_blob)
    class_defs_off = cursor
    cursor += len(class_defs_blob)
    code_items_off = cursor
    cursor += len(code_items_blob)
    string_data_off = cursor
    cursor += len(string_data_blob)
    # Align map_off to 4 bytes.
    if cursor % 4 != 0:
        pad = 4 - (cursor % 4)
        cursor += pad
    else:
        pad = 0
    map_off = cursor

    # map_list: u32 size + size * (u16 type, u16 pad, u32 count, u32 offset)
    map_entries = [
        (_TYPE_HEADER_ITEM, 1, 0),
        (_TYPE_STRING_ID_ITEM, num_strings, string_ids_off),
        (_TYPE_TYPE_ID_ITEM, num_types, type_ids_off),
        (_TYPE_PROTO_ID_ITEM, num_protos, proto_ids_off),
        (_TYPE_FIELD_ID_ITEM, num_fields, field_ids_off),
        (_TYPE_METHOD_ID_ITEM, num_methods, method_ids_off),
        (_TYPE_CLASS_DEF_ITEM, num_classes, class_defs_off),
        (_TYPE_CODE_ITEM, num_code_items, code_items_off),
        (_TYPE_STRING_DATA_ITEM, num_strings, string_data_off),
        (_TYPE_MAP_LIST, 1, map_off),
    ]
    map_list_blob = struct.pack("<I", len(map_entries))
    for map_type, count, offset in map_entries:
        map_list_blob += struct.pack("<HHII", map_type, 0, count, offset)
    cursor = map_off + len(map_list_blob)
    total_size = cursor

    # --- header -----------------------------------------------------------
    header = bytearray(DEX_HEADER_SIZE)
    header[0:8] = DEX_MAGIC_035
    # checksum @ 0x08 (leave zero)
    # signature @ 0x0C, 20 bytes (leave zero)
    struct.pack_into("<I", header, 0x20, total_size)      # file_size
    struct.pack_into("<I", header, 0x24, DEX_HEADER_SIZE)  # header_size
    struct.pack_into("<I", header, 0x28, 0x12345678)       # endian_tag (LE)
    # link_size, link_off = 0
    struct.pack_into("<I", header, 0x34, map_off)          # map_off
    struct.pack_into("<I", header, 0x38, num_strings)      # string_ids_size
    struct.pack_into("<I", header, 0x3C, string_ids_off)   # string_ids_off
    struct.pack_into("<I", header, 0x40, num_types)        # type_ids_size
    struct.pack_into("<I", header, 0x44, type_ids_off)     # type_ids_off
    struct.pack_into("<I", header, 0x48, num_protos)       # proto_ids_size
    struct.pack_into("<I", header, 0x4C, proto_ids_off)    # proto_ids_off
    struct.pack_into("<I", header, 0x50, num_fields)       # field_ids_size
    struct.pack_into("<I", header, 0x54, field_ids_off)    # field_ids_off
    struct.pack_into("<I", header, 0x58, num_methods)      # method_ids_size
    struct.pack_into("<I", header, 0x5C, method_ids_off)   # method_ids_off
    struct.pack_into("<I", header, 0x60, num_classes)      # class_defs_size
    struct.pack_into("<I", header, 0x64, class_defs_off)   # class_defs_off
    # data_size / data_off: point at the region after class_defs (not
    # validated by our parser, but keep it reasonable).
    data_off = code_items_off
    data_size = total_size - data_off
    struct.pack_into("<I", header, 0x68, data_size)
    struct.pack_into("<I", header, 0x6C, data_off)

    # --- assemble ----------------------------------------------------------
    buf = bytearray(total_size)
    buf[0:DEX_HEADER_SIZE] = bytes(header)
    buf[string_ids_off:string_ids_off + len(string_ids_blob)] = string_ids_blob
    buf[type_ids_off:type_ids_off + len(type_ids_blob)] = type_ids_blob
    buf[proto_ids_off:proto_ids_off + len(proto_ids_blob)] = proto_ids_blob
    buf[field_ids_off:field_ids_off + len(field_ids_blob)] = field_ids_blob
    buf[method_ids_off:method_ids_off + len(method_ids_blob)] = method_ids_blob
    buf[class_defs_off:class_defs_off + len(class_defs_blob)] = class_defs_blob
    buf[code_items_off:code_items_off + len(code_items_blob)] = code_items_blob
    buf[string_data_off:string_data_off + len(string_data_blob)] = string_data_blob
    buf[map_off:map_off + len(map_list_blob)] = map_list_blob

    layout = MinimalDexLayout(
        total_size=total_size,
        string_ids_off=string_ids_off,
        string_ids_count=num_strings,
        type_ids_off=type_ids_off,
        type_ids_count=num_types,
        proto_ids_off=proto_ids_off,
        proto_ids_count=num_protos,
        field_ids_off=field_ids_off,
        field_ids_count=num_fields,
        method_ids_off=method_ids_off,
        method_ids_count=num_methods,
        class_defs_off=class_defs_off,
        class_defs_count=num_classes,
        code_items_off=code_items_off,
        code_items_count=num_code_items,
        string_data_off=string_data_off,
        string_data_count=num_strings,
        map_off=map_off,
    )
    return bytes(buf), layout


__all__ = [
    "DEX_HEADER_SIZE",
    "DEX_MAGIC_035",
    "MinimalDexLayout",
    "build_minimal_dex",
]
