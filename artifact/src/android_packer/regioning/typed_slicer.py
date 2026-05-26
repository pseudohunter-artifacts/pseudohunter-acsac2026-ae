"""Type-aware region slicing for the full framework.

Replaces uniform 4KB sliding windows with structure-aware slicing:
- DEX entries: sliced by DEX section boundaries (header, string_ids, ...)
- ELF entries: sliced by ELF section headers (.text, .rodata, ...)
- Asset/Unknown entries: 4KB sliding window (same as legacy)

Each region carries its structural context (entry_type, section_type)
which feeds into the type-specific expert routing downstream.

Design contract (from improved_packed_apk_framework.md §3):
- Large sections (>8KB) are further windowed at 4KB stride
- Small entries (<256 bytes) yield a single region = entire entry
- Region output carries: entry_path, entry_type, section_type, offset, bytes
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from android_packer.apkio.objects import ApkObject

# Lazy import to avoid circular deps; dex_item_parser is stdlib-only
from android_packer.features.dex_item_parser import (
    DEX_ITEM_TYPES,
    DexItemSpan,
    DexParseError,
    parse_dex_item_spans,
)

__all__ = [
    "TypedRegion",
    "iter_typed_regions",
    "ENTRY_COARSE_TYPES",
    "SECTION_TYPES",
]

# ---------------------------------------------------------------------------
# Type vocabularies
# ---------------------------------------------------------------------------

# Coarse entry types (7 types, magic-based, leakage-free)
ENTRY_COARSE_TYPES: Tuple[str, ...] = (
    "dex",       # 0: file header = "dex\n"
    "elf",       # 1: file header = \x7fELF
    "manifest",  # 2: AndroidManifest.xml (AXML)
    "arsc",      # 3: resources.arsc
    "archive",   # 4: embedded ZIP/JAR/APK/gzip
    "asset",     # 5: assets/* or res/* with known magic
    "unknown",   # 6: everything else
)

_ENTRY_TYPE_TO_ID = {t: i for i, t in enumerate(ENTRY_COARSE_TYPES)}

# Section types (unified vocabulary across all entry types)
SECTION_TYPES: Tuple[str, ...] = (
    # DEX sections (0-9)
    "dex_header",       # 0
    "dex_string_ids",   # 1
    "dex_type_ids",     # 2
    "dex_proto_ids",    # 3
    "dex_field_ids",    # 4
    "dex_method_ids",   # 5
    "dex_class_defs",   # 6
    "dex_code_item",    # 7
    "dex_string_data",  # 8
    "dex_other",        # 9
    # ELF sections (10-17)
    "elf_text",         # 10
    "elf_rodata",       # 11
    "elf_data",         # 12
    "elf_dynamic",      # 13
    "elf_dynsym",       # 14
    "elf_init_array",   # 15
    "elf_custom",       # 16
    "elf_other",        # 17
    # Generic (18-19)
    "window",           # 18: sliding window (asset/unknown)
    "whole_entry",      # 19: entry too small, used as-is
)

_SECTION_TYPE_TO_ID = {t: i for i, t in enumerate(SECTION_TYPES)}

# Mapping from DEX_ITEM_TYPES to our section vocabulary
_DEX_ITEM_TO_SECTION = {
    "header": "dex_header",
    "string_ids": "dex_string_ids",
    "type_ids": "dex_type_ids",
    "proto_ids": "dex_proto_ids",
    "field_ids": "dex_field_ids",
    "method_ids": "dex_method_ids",
    "class_defs": "dex_class_defs",
    "code_item": "dex_code_item",
    "string_data": "dex_string_data",
    "other": "dex_other",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedRegion:
    """A structurally-typed region within an APK entry.

    Carries all context needed for downstream feature extraction and
    model type routing.
    """

    # Identity
    apk_id: str
    object_id: str
    entry_path: str

    # Type context (integers → feed into nn.Embedding)
    entry_type_id: int      # index into ENTRY_COARSE_TYPES
    section_type_id: int    # index into SECTION_TYPES

    # Byte range within the entry
    offset_start: int
    offset_end: int

    # Ordering context
    entry_index: int        # position of this entry in the APK
    region_index: int       # position of this region within the entry
    n_regions_in_entry: int  # total regions for this entry (set after iteration)

    @property
    def size(self) -> int:
        return self.offset_end - self.offset_start

    @property
    def entry_type(self) -> str:
        return ENTRY_COARSE_TYPES[self.entry_type_id]

    @property
    def section_type(self) -> str:
        return SECTION_TYPES[self.section_type_id]


# ---------------------------------------------------------------------------
# Entry type detection (magic-based, leakage-free)
# ---------------------------------------------------------------------------


def detect_entry_type(path: str, data: bytes) -> int:
    """Detect coarse entry type from magic bytes + path + structural hints.

    Strategy (priority order):
    1. Magic bytes: authoritative if present and valid
    2. Path extension/location: fallback when magic is absent or ambiguous
    3. Structural validation: confirm magic matches expected structure

    Returns integer index into ENTRY_COARSE_TYPES.
    Training and inference use identical logic (no label dependency).
    """
    p = path.lower().replace("\\", "/")

    # --- Priority 1: Magic bytes (authoritative) ---
    if len(data) >= 4:
        magic4 = data[:4]

        if magic4 == b"dex\n":
            # Validate: real DEX should have reasonable file_size in header
            if len(data) >= 36:
                import struct
                try:
                    file_size = struct.unpack_from("<I", data, 32)[0]
                    if file_size <= len(data) * 2:  # reasonable
                        return _ENTRY_TYPE_TO_ID["dex"]
                except struct.error:
                    pass
            # Even if validation fails, magic says DEX
            return _ENTRY_TYPE_TO_ID["dex"]

        if magic4 == b"\x7fELF":
            return _ENTRY_TYPE_TO_ID["elf"]

        if magic4 in (b"PK\x03\x04", b"PK\x05\x06"):
            return _ENTRY_TYPE_TO_ID["archive"]

        if data[:4] == b"\x02\x00\x0c\x00":
            return _ENTRY_TYPE_TO_ID["arsc"]

        if data[:2] == b"\x1f\x8b":
            return _ENTRY_TYPE_TO_ID["archive"]

        # AXML detection (Android binary XML)
        if len(data) >= 8 and magic4 == b"\x03\x00\x08\x00":
            if p == "androidmanifest.xml":
                return _ENTRY_TYPE_TO_ID["manifest"]
            return _ENTRY_TYPE_TO_ID["asset"]

    # --- Priority 2: Path-based classification ---
    # (used when magic is absent/unrecognized — e.g. encrypted files)

    if p == "androidmanifest.xml":
        return _ENTRY_TYPE_TO_ID["manifest"]
    if p == "resources.arsc":
        return _ENTRY_TYPE_TO_ID["arsc"]

    # DEX by path (classes.dex, classesN.dex)
    basename = p.rsplit("/", 1)[-1] if "/" in p else p
    if basename.startswith("classes") and basename.endswith(".dex"):
        # Path says DEX but magic doesn't match → still route to DEX expert
        # (DEX expert will detect "invalid header" as a feature = encrypted DEX)
        return _ENTRY_TYPE_TO_ID["dex"]

    # ELF/SO by path
    if p.startswith("lib/") or basename.endswith(".so"):
        return _ENTRY_TYPE_TO_ID["elf"]

    # Archives by extension
    if basename.endswith((".apk", ".jar", ".zip", ".dex")):
        return _ENTRY_TYPE_TO_ID["archive"]

    # Assets and resources
    if p.startswith("assets/") or p.startswith("res/"):
        return _ENTRY_TYPE_TO_ID["asset"]

    # META-INF (signatures)
    if p.startswith("meta-inf/"):
        return _ENTRY_TYPE_TO_ID["asset"]

    # --- Priority 3: Content-based heuristic for unknown ---
    # Scan deeper into data for embedded signatures
    if len(data) >= 64:
        # Check if it contains DEX/ELF somewhere (could be a packed payload)
        # But don't re-route — let unknown expert handle with the
        # "embedded_dex/elf_scan" features detecting this situation
        pass

    return _ENTRY_TYPE_TO_ID["unknown"]


# ---------------------------------------------------------------------------
# DEX section slicing
# ---------------------------------------------------------------------------

_MAX_WINDOW = 8192   # sections larger than this get windowed
_WINDOW_SIZE = 4096
_WINDOW_STRIDE = 2048


def _slice_dex(
    data: bytes,
    apk_id: str,
    object_id: str,
    entry_path: str,
    entry_index: int,
) -> List[TypedRegion]:
    """Slice a DEX entry by structural sections."""
    try:
        spans = parse_dex_item_spans(data)
    except DexParseError:
        # Malformed/encrypted DEX — fall back to sliding window
        return _slice_window(
            data, apk_id, object_id, entry_path,
            _ENTRY_TYPE_TO_ID["dex"], entry_index,
        )

    regions: List[TypedRegion] = []
    region_idx = 0

    for span in spans:
        item_name = DEX_ITEM_TYPES[span.item_type]
        section_name = _DEX_ITEM_TO_SECTION.get(item_name, "dex_other")
        section_id = _SECTION_TYPE_TO_ID[section_name]

        if span.size <= _MAX_WINDOW:
            # Single region for this section
            regions.append(TypedRegion(
                apk_id=apk_id,
                object_id=object_id,
                entry_path=entry_path,
                entry_type_id=_ENTRY_TYPE_TO_ID["dex"],
                section_type_id=section_id,
                offset_start=span.offset,
                offset_end=span.offset + span.size,
                entry_index=entry_index,
                region_index=region_idx,
                n_regions_in_entry=0,  # filled later
            ))
            region_idx += 1
        else:
            # Window large sections
            for win_start in range(span.offset, span.offset + span.size, _WINDOW_STRIDE):
                win_end = min(win_start + _WINDOW_SIZE, span.offset + span.size)
                if win_end - win_start < 256:
                    continue
                regions.append(TypedRegion(
                    apk_id=apk_id,
                    object_id=object_id,
                    entry_path=entry_path,
                    entry_type_id=_ENTRY_TYPE_TO_ID["dex"],
                    section_type_id=section_id,
                    offset_start=win_start,
                    offset_end=win_end,
                    entry_index=entry_index,
                    region_index=region_idx,
                    n_regions_in_entry=0,
                ))
                region_idx += 1

    # Fill n_regions_in_entry
    n = len(regions)
    filled = []
    for r in regions:
        filled.append(TypedRegion(
            apk_id=r.apk_id, object_id=r.object_id, entry_path=r.entry_path,
            entry_type_id=r.entry_type_id, section_type_id=r.section_type_id,
            offset_start=r.offset_start, offset_end=r.offset_end,
            entry_index=r.entry_index, region_index=r.region_index,
            n_regions_in_entry=n,
        ))
    return filled


# ---------------------------------------------------------------------------
# ELF section slicing
# ---------------------------------------------------------------------------

# Well-known ELF section names → our vocabulary
_ELF_SECTION_MAP = {
    b".text": "elf_text",
    b".rodata": "elf_rodata",
    b".data": "elf_data",
    b".dynamic": "elf_dynamic",
    b".dynsym": "elf_dynsym",
    b".init_array": "elf_init_array",
}


def _parse_elf_sections(data: bytes) -> List[Tuple[str, int, int]]:
    """Parse ELF section headers. Returns [(section_type_name, offset, size)]."""
    if len(data) < 64:
        return []

    # ELF header
    ei_class = data[4]  # 1=32bit, 2=64bit
    if ei_class == 1:
        # 32-bit ELF
        if len(data) < 52:
            return []
        e_shoff = struct.unpack_from("<I", data, 32)[0]
        e_shentsize = struct.unpack_from("<H", data, 46)[0]
        e_shnum = struct.unpack_from("<H", data, 48)[0]
        e_shstrndx = struct.unpack_from("<H", data, 50)[0]
        hdr_fmt = "<IIIIIIII"  # 32-bit section header
        sh_name_off, sh_offset_off, sh_size_off = 0, 16, 20
    elif ei_class == 2:
        # 64-bit ELF
        if len(data) < 64:
            return []
        e_shoff = struct.unpack_from("<Q", data, 40)[0]
        e_shentsize = struct.unpack_from("<H", data, 58)[0]
        e_shnum = struct.unpack_from("<H", data, 60)[0]
        e_shstrndx = struct.unpack_from("<H", data, 62)[0]
        sh_name_off, sh_offset_off, sh_size_off = 0, 24, 32
    else:
        return []

    if e_shoff == 0 or e_shnum == 0 or e_shentsize == 0:
        return []
    if e_shoff + e_shnum * e_shentsize > len(data):
        return []

    # Read string table section to get section names
    if e_shstrndx >= e_shnum:
        return []
    strtab_hdr_off = e_shoff + e_shstrndx * e_shentsize
    if ei_class == 1:
        strtab_offset = struct.unpack_from("<I", data, strtab_hdr_off + 16)[0]
        strtab_size = struct.unpack_from("<I", data, strtab_hdr_off + 20)[0]
    else:
        strtab_offset = struct.unpack_from("<Q", data, strtab_hdr_off + 24)[0]
        strtab_size = struct.unpack_from("<Q", data, strtab_hdr_off + 32)[0]

    if strtab_offset + strtab_size > len(data):
        return []
    strtab = data[strtab_offset:strtab_offset + strtab_size]

    sections = []
    for i in range(e_shnum):
        sh_off = e_shoff + i * e_shentsize
        if sh_off + e_shentsize > len(data):
            break

        if ei_class == 1:
            sh_name_idx = struct.unpack_from("<I", data, sh_off)[0]
            sh_offset = struct.unpack_from("<I", data, sh_off + 16)[0]
            sh_size = struct.unpack_from("<I", data, sh_off + 20)[0]
        else:
            sh_name_idx = struct.unpack_from("<I", data, sh_off)[0]
            sh_offset = struct.unpack_from("<Q", data, sh_off + 24)[0]
            sh_size = struct.unpack_from("<Q", data, sh_off + 32)[0]

        if sh_size == 0 or sh_offset + sh_size > len(data):
            continue

        # Get section name from string table
        if sh_name_idx < len(strtab):
            end = strtab.index(b"\x00", sh_name_idx) if b"\x00" in strtab[sh_name_idx:] else len(strtab)
            name_bytes = strtab[sh_name_idx:end]
        else:
            name_bytes = b""

        # Map to our section type
        section_type = _ELF_SECTION_MAP.get(name_bytes, None)
        if section_type is None:
            # Check if it's a custom/packer-related section
            if name_bytes and not name_bytes.startswith(b"."):
                section_type = "elf_custom"
            elif name_bytes in (b".packed", b".jiagu", b".bangcle", b".ijiami"):
                section_type = "elf_custom"
            else:
                section_type = "elf_other"

        sections.append((section_type, sh_offset, sh_size))

    return sections


def _slice_elf(
    data: bytes,
    apk_id: str,
    object_id: str,
    entry_path: str,
    entry_index: int,
) -> List[TypedRegion]:
    """Slice an ELF entry by section headers."""
    sections = _parse_elf_sections(data)

    if not sections:
        # Failed to parse ELF — fall back to sliding window
        return _slice_window(
            data, apk_id, object_id, entry_path,
            _ENTRY_TYPE_TO_ID["elf"], entry_index,
        )

    regions: List[TypedRegion] = []
    region_idx = 0

    for section_type_name, offset, size in sections:
        section_id = _SECTION_TYPE_TO_ID[section_type_name]

        if size <= _MAX_WINDOW:
            regions.append(TypedRegion(
                apk_id=apk_id, object_id=object_id, entry_path=entry_path,
                entry_type_id=_ENTRY_TYPE_TO_ID["elf"],
                section_type_id=section_id,
                offset_start=offset, offset_end=offset + size,
                entry_index=entry_index, region_index=region_idx,
                n_regions_in_entry=0,
            ))
            region_idx += 1
        else:
            for win_start in range(offset, offset + size, _WINDOW_STRIDE):
                win_end = min(win_start + _WINDOW_SIZE, offset + size)
                if win_end - win_start < 256:
                    continue
                regions.append(TypedRegion(
                    apk_id=apk_id, object_id=object_id, entry_path=entry_path,
                    entry_type_id=_ENTRY_TYPE_TO_ID["elf"],
                    section_type_id=section_id,
                    offset_start=win_start, offset_end=win_end,
                    entry_index=entry_index, region_index=region_idx,
                    n_regions_in_entry=0,
                ))
                region_idx += 1

    n = len(regions)
    return [
        TypedRegion(
            apk_id=r.apk_id, object_id=r.object_id, entry_path=r.entry_path,
            entry_type_id=r.entry_type_id, section_type_id=r.section_type_id,
            offset_start=r.offset_start, offset_end=r.offset_end,
            entry_index=r.entry_index, region_index=r.region_index,
            n_regions_in_entry=n,
        )
        for r in regions
    ]


# ---------------------------------------------------------------------------
# Generic sliding window (for asset/unknown/fallback)
# ---------------------------------------------------------------------------


def _slice_window(
    data: bytes,
    apk_id: str,
    object_id: str,
    entry_path: str,
    entry_type_id: int,
    entry_index: int,
    window_size: int = _WINDOW_SIZE,
    stride: int = _WINDOW_STRIDE,
) -> List[TypedRegion]:
    """Sliding window slicing for asset/unknown entries."""
    section_id = _SECTION_TYPE_TO_ID["window"]

    if len(data) < 256:
        # Tiny entry — single region
        return [TypedRegion(
            apk_id=apk_id, object_id=object_id, entry_path=entry_path,
            entry_type_id=entry_type_id,
            section_type_id=_SECTION_TYPE_TO_ID["whole_entry"],
            offset_start=0, offset_end=len(data),
            entry_index=entry_index, region_index=0,
            n_regions_in_entry=1,
        )]

    regions: List[TypedRegion] = []
    region_idx = 0

    for win_start in range(0, len(data), stride):
        win_end = min(win_start + window_size, len(data))
        if win_end - win_start < 256:
            continue
        regions.append(TypedRegion(
            apk_id=apk_id, object_id=object_id, entry_path=entry_path,
            entry_type_id=entry_type_id,
            section_type_id=section_id,
            offset_start=win_start, offset_end=win_end,
            entry_index=entry_index, region_index=region_idx,
            n_regions_in_entry=0,
        ))
        region_idx += 1

    n = len(regions)
    return [
        TypedRegion(
            apk_id=r.apk_id, object_id=r.object_id, entry_path=r.entry_path,
            entry_type_id=r.entry_type_id, section_type_id=r.section_type_id,
            offset_start=r.offset_start, offset_end=r.offset_end,
            entry_index=r.entry_index, region_index=r.region_index,
            n_regions_in_entry=n,
        )
        for r in regions
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def iter_typed_regions(
    metadata: ApkObject,
    data: bytes,
    *,
    entry_index: int = 0,
) -> List[TypedRegion]:
    """Generate typed regions for a single APK entry.

    Dispatches to the appropriate slicer based on detected entry type:
    - DEX: section-aware slicing via dex_item_parser
    - ELF: section-aware slicing via ELF header parsing
    - Other: 4KB sliding window

    Args:
        metadata: APK object metadata (from iter_apk_objects)
        data: raw bytes of the entry
        entry_index: position of this entry in the APK (for ordering context)

    Returns:
        List of TypedRegion, each carrying structural context for downstream
        feature extraction and model routing.
    """
    entry_type_id = detect_entry_type(metadata.object_path, data)

    if entry_type_id == _ENTRY_TYPE_TO_ID["dex"]:
        return _slice_dex(
            data, metadata.apk_id, metadata.object_id,
            metadata.object_path, entry_index,
        )
    elif entry_type_id == _ENTRY_TYPE_TO_ID["elf"]:
        return _slice_elf(
            data, metadata.apk_id, metadata.object_id,
            metadata.object_path, entry_index,
        )
    else:
        return _slice_window(
            data, metadata.apk_id, metadata.object_id,
            metadata.object_path, entry_type_id, entry_index,
        )
