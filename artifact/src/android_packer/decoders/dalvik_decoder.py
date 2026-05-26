"""Linear Dalvik bytecode decoder for pseudo-code BERT.

Performs linear disassembly of raw bytes as Dalvik instructions.
Even non-DEX data gets decoded — the resulting "invalid opcode" and
"abnormal operand" tokens ARE the signal that tells BERT the data is packed.

From pseudo_code_bert_packed_apk_framework.md §7:
- Decode each 2-byte unit as a Dalvik opcode
- Normalize operands to semantic classes (REG, CONST, IDX_OK, IDX_BAD, ...)
- Track abnormality indicators (invalid opcodes, impossible branches, bad indices)

Design contract:
- Pure stdlib (no torch, no androguard)
- Deterministic output
- Works on ANY bytes (not just valid DEX)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = [
    "DalvikToken",
    "decode_dalvik_region",
    "DALVIK_TOKEN_VOCAB",
    "DALVIK_TYPED_COMPONENT_TOKENS",
    "dalvik_token_to_id",
]


# ---------------------------------------------------------------------------
# Dalvik opcode table (DEX format spec, 256 opcodes)
# Grouped into ~30 semantic classes for token normalization
# ---------------------------------------------------------------------------

# Opcode → (mnemonic, format_id, n_code_units)
# format_id encodes how to parse operands. We only need instruction width.
# Reference: https://source.android.com/docs/core/runtime/dalvik-bytecode

# Instruction widths by opcode (in 16-bit code units)
# Most instructions are 1-3 units; payload instructions can be larger
_OPCODE_WIDTHS = [0] * 256

# Format 10x: 1 unit (opcode only)
for op in [0x00, 0x0e, 0x73, 0x79, 0x7a]:
    _OPCODE_WIDTHS[op] = 1

# Format 12x, 11n, 11x, 10t: 1 unit
for op in range(0x01, 0x0e):  # move variants
    _OPCODE_WIDTHS[op] = 1
for op in [0x1d, 0x1e, 0x27, 0x28]:  # monitor, throw, goto
    _OPCODE_WIDTHS[op] = 1

# Format 22x, 21t, 21s, 21h, 21c: 2 units
for op in range(0x0f, 0x12):  # return variants
    _OPCODE_WIDTHS[op] = 1
for op in [0x12, 0x13]:  # const/4, const/16
    _OPCODE_WIDTHS[op] = 1 if op == 0x12 else 2
for op in range(0x14, 0x1d):  # const variants + more
    _OPCODE_WIDTHS[op] = 2 if op < 0x18 else 3
for op in range(0x1f, 0x27):  # check-cast, instance-of, array, new-instance
    _OPCODE_WIDTHS[op] = 2

# Branches: 2 units (format 22t)
for op in range(0x29, 0x2b):  # goto/16, goto/32
    _OPCODE_WIDTHS[op] = 2 if op == 0x29 else 3
for op in range(0x2b, 0x2d):  # packed-switch, sparse-switch
    _OPCODE_WIDTHS[op] = 3
for op in range(0x2d, 0x32):  # cmp variants
    _OPCODE_WIDTHS[op] = 2
for op in range(0x32, 0x3e):  # if-* variants
    _OPCODE_WIDTHS[op] = 2

# Array ops: 2 units
for op in range(0x3e, 0x44):  # unused
    _OPCODE_WIDTHS[op] = 1
for op in range(0x44, 0x52):  # aget, aput variants
    _OPCODE_WIDTHS[op] = 2

# Instance field ops: 2 units (format 22c)
for op in range(0x52, 0x6e):  # iget, iput, sget, sput variants
    _OPCODE_WIDTHS[op] = 2

# Invoke: 3 units (format 35c, 3rc)
for op in range(0x6e, 0x79):  # invoke-* variants
    _OPCODE_WIDTHS[op] = 3

# Unary/binary ops: 1 or 2 units
for op in range(0x7b, 0xd0):  # neg, not, int-to-*, add, sub, mul, ...
    _OPCODE_WIDTHS[op] = 2 if op >= 0x90 else 1

# Binary/lit ops: 2 units
for op in range(0xd0, 0xe3):  # add-int/lit, ...
    _OPCODE_WIDTHS[op] = 2

# Fill remaining as 1 (safe default for unknown/invalid)
for i in range(256):
    if _OPCODE_WIDTHS[i] == 0:
        _OPCODE_WIDTHS[i] = 1


# Semantic opcode classes (normalized)
_OPCODE_CLASSES = {}
_CLASS_NOP = "nop"
_CLASS_MOVE = "move"
_CLASS_RETURN = "return"
_CLASS_CONST = "const"
_CLASS_MONITOR = "monitor"
_CLASS_CHECK = "check_cast"
_CLASS_ARRAY = "array"
_CLASS_GOTO = "goto"
_CLASS_SWITCH = "switch"
_CLASS_CMP = "cmp"
_CLASS_IF = "if"
_CLASS_AGET = "aget"
_CLASS_APUT = "aput"
_CLASS_IGET = "iget"
_CLASS_IPUT = "iput"
_CLASS_SGET = "sget"
_CLASS_SPUT = "sput"
_CLASS_INVOKE = "invoke"
_CLASS_UNARY = "unary"
_CLASS_BINARY = "binary"
_CLASS_LITERAL = "literal"
_CLASS_THROW = "throw"
_CLASS_NEW = "new"
_CLASS_FILL = "fill"
_CLASS_UNUSED = "unused"

_OPCODE_CLASSES[0x00] = _CLASS_NOP
for op in range(0x01, 0x0e):
    _OPCODE_CLASSES[op] = _CLASS_MOVE
for op in range(0x0e, 0x12):
    _OPCODE_CLASSES[op] = _CLASS_RETURN
for op in range(0x12, 0x1d):
    _OPCODE_CLASSES[op] = _CLASS_CONST
_OPCODE_CLASSES[0x1d] = _CLASS_MONITOR
_OPCODE_CLASSES[0x1e] = _CLASS_MONITOR
_OPCODE_CLASSES[0x1f] = _CLASS_CHECK
_OPCODE_CLASSES[0x20] = "instance_of"
_OPCODE_CLASSES[0x21] = _CLASS_ARRAY
_OPCODE_CLASSES[0x22] = _CLASS_NEW
_OPCODE_CLASSES[0x23] = "new_array"
_OPCODE_CLASSES[0x24] = "filled_new_array"
_OPCODE_CLASSES[0x25] = "filled_new_array"
_OPCODE_CLASSES[0x26] = _CLASS_FILL
_OPCODE_CLASSES[0x27] = _CLASS_THROW
for op in range(0x28, 0x2b):
    _OPCODE_CLASSES[op] = _CLASS_GOTO
for op in range(0x2b, 0x2d):
    _OPCODE_CLASSES[op] = _CLASS_SWITCH
for op in range(0x2d, 0x32):
    _OPCODE_CLASSES[op] = _CLASS_CMP
for op in range(0x32, 0x3e):
    _OPCODE_CLASSES[op] = _CLASS_IF
for op in range(0x3e, 0x44):
    _OPCODE_CLASSES[op] = _CLASS_UNUSED
for op in range(0x44, 0x4b):
    _OPCODE_CLASSES[op] = _CLASS_AGET
for op in range(0x4b, 0x52):
    _OPCODE_CLASSES[op] = _CLASS_APUT
for op in range(0x52, 0x5a):
    _OPCODE_CLASSES[op] = _CLASS_IGET
for op in range(0x5a, 0x62):
    _OPCODE_CLASSES[op] = _CLASS_IPUT
for op in range(0x62, 0x6a):
    _OPCODE_CLASSES[op] = _CLASS_SGET
for op in range(0x6a, 0x72):
    _OPCODE_CLASSES[op] = _CLASS_SPUT
for op in range(0x6e, 0x79):
    _OPCODE_CLASSES[op] = _CLASS_INVOKE
for op in range(0x7b, 0x90):
    _OPCODE_CLASSES[op] = _CLASS_UNARY
for op in range(0x90, 0xd0):
    _OPCODE_CLASSES[op] = _CLASS_BINARY
for op in range(0xd0, 0xe3):
    _OPCODE_CLASSES[op] = _CLASS_LITERAL


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------

# Special tokens
_SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]

# Opcode class tokens (semantic groups)
_OPCODE_CLASS_TOKENS = sorted(set(_OPCODE_CLASSES.values())) + ["INVALID_OPCODE"]

# Operand tokens
_OPERAND_TOKENS = [
    "REG", "REG_PAIR", "REG_LIST",
    "CONST_SMALL", "CONST_LARGE",
    "STRING_IDX_OK", "STRING_IDX_BAD",
    "TYPE_IDX_OK", "TYPE_IDX_BAD",
    "FIELD_IDX_OK", "FIELD_IDX_BAD",
    "METHOD_IDX_OK", "METHOD_IDX_BAD",
    "BRANCH_OK", "BRANCH_BAD",
    "OFFSET_OK", "OFFSET_BAD",
]

DALVIK_TYPED_COMPONENT_TOKENS: Tuple[str, ...] = (
    "INVOKE_VIRTUAL",
    "INVOKE_SUPER",
    "INVOKE_DIRECT",
    "INVOKE_STATIC",
    "INVOKE_INTERFACE",
    "INVOKE_RANGE",
)

# Meta tokens
_META_TOKENS = [
    "INSN_END",       # end of one instruction
    "PAD_ZERO",       # 0x0000 padding (normal in DEX)
    "PAD_NONZERO",    # non-zero bytes interpreted as padding (abnormal)
    "SEQUENCE_END",   # end of decoded sequence
]

# Build complete vocabulary
DALVIK_TOKEN_VOCAB: Tuple[str, ...] = tuple(
    _SPECIAL_TOKENS + _OPCODE_CLASS_TOKENS + _OPERAND_TOKENS + _META_TOKENS
)

_TOKEN_TO_ID = {t: i for i, t in enumerate(DALVIK_TOKEN_VOCAB)}


def dalvik_token_to_id(token: str) -> int:
    """Convert token string to integer ID."""
    return _TOKEN_TO_ID.get(token, _TOKEN_TO_ID["[UNK]"])


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DalvikToken:
    """A single decoded pseudo-Dalvik token."""
    token: str          # vocabulary token string
    token_id: int       # integer ID
    offset: int         # byte offset in the region
    is_abnormal: bool   # True if this indicates anomaly


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------


def decode_dalvik_region(
    data: bytes,
    *,
    max_tokens: int = 510,  # leave room for BOS/EOS
    string_ids_count: int = 0,
    type_ids_count: int = 0,
    method_ids_count: int = 0,
    field_ids_count: int = 0,
) -> List[DalvikToken]:
    """Linearly decode raw bytes as Dalvik bytecode.

    Args:
        data: Raw bytes to decode (any content, not necessarily valid DEX)
        max_tokens: Maximum tokens to produce (excluding BOS/EOS)
        string_ids_count: Number of valid string IDs (from DEX header, 0=unknown)
        type_ids_count: Number of valid type IDs
        method_ids_count: Number of valid method IDs
        field_ids_count: Number of valid field IDs

    Returns:
        List of DalvikToken. Always starts with BOS, ends with EOS.
        Invalid opcodes, abnormal indices, impossible branches → is_abnormal=True
    """
    tokens: List[DalvikToken] = []

    # BOS
    tokens.append(DalvikToken("[BOS]", _TOKEN_TO_ID["[BOS]"], 0, False))

    # Dalvik is 16-bit aligned
    offset = 0
    n_tokens = 0

    while offset + 1 < len(data) and n_tokens < max_tokens:
        # Read opcode (low byte of first 16-bit unit)
        opcode = data[offset]

        # Check if this is padding (0x0000)
        if data[offset] == 0 and data[offset + 1] == 0:
            tokens.append(DalvikToken(
                "PAD_ZERO", _TOKEN_TO_ID["PAD_ZERO"], offset, False
            ))
            offset += 2
            n_tokens += 1
            continue

        # Get opcode class
        opcode_class = _OPCODE_CLASSES.get(opcode, None)
        if opcode_class is None:
            # Invalid opcode — strong packed indicator
            tokens.append(DalvikToken(
                "INVALID_OPCODE", _TOKEN_TO_ID["INVALID_OPCODE"], offset, True
            ))
            tokens.append(DalvikToken(
                "INSN_END", _TOKEN_TO_ID["INSN_END"], offset, True
            ))
            offset += 2
            n_tokens += 2
            continue

        # Emit opcode token
        opcode_token_id = _TOKEN_TO_ID.get(opcode_class, _TOKEN_TO_ID["[UNK]"])
        tokens.append(DalvikToken(opcode_class, opcode_token_id, offset, False))
        n_tokens += 1

        # Get instruction width
        width = _OPCODE_WIDTHS[opcode]  # in 16-bit units
        byte_width = width * 2

        # Decode operands based on opcode class
        if n_tokens < max_tokens and byte_width > 2:
            operand_bytes = data[offset + 2:offset + byte_width]
            if len(operand_bytes) < byte_width - 2:
                # Truncated instruction
                tokens.append(DalvikToken(
                    "INSN_END", _TOKEN_TO_ID["INSN_END"], offset, True
                ))
                n_tokens += 1
                offset += 2
                continue

            # Emit operand tokens
            operand_token = _classify_operands(
                opcode, opcode_class, operand_bytes, offset,
                string_ids_count, type_ids_count,
                method_ids_count, field_ids_count,
                len(data),
            )
            for ot, is_abn in operand_token:
                if n_tokens >= max_tokens:
                    break
                tid = _TOKEN_TO_ID.get(ot, _TOKEN_TO_ID["[UNK]"])
                tokens.append(DalvikToken(ot, tid, offset, is_abn))
                n_tokens += 1

        # End of instruction marker
        if n_tokens < max_tokens:
            tokens.append(DalvikToken(
                "INSN_END", _TOKEN_TO_ID["INSN_END"], offset, False
            ))
            n_tokens += 1

        offset += byte_width

    # EOS
    tokens.append(DalvikToken("[EOS]", _TOKEN_TO_ID["[EOS]"], offset, False))

    return tokens


def _classify_operands(
    opcode: int,
    opcode_class: str,
    operand_bytes: bytes,
    offset: int,
    n_strings: int,
    n_types: int,
    n_methods: int,
    n_fields: int,
    data_len: int,
) -> List[Tuple[str, bool]]:
    """Classify operand bytes into semantic token types.

    Returns list of (token_name, is_abnormal) tuples.
    """
    result = []

    if opcode_class in (_CLASS_INVOKE,):
        # invoke-* has method index in bytes 2-3 of the full instruction
        result.append((_invoke_kind_token(opcode), False))
        if len(operand_bytes) >= 4:
            method_idx = struct.unpack_from("<H", operand_bytes, 0)[0]
            if n_methods > 0 and method_idx >= n_methods:
                result.append(("METHOD_IDX_BAD", True))
            else:
                result.append(("METHOD_IDX_OK", False))
            result.append(("REG_LIST", False))

    elif opcode_class in (_CLASS_IGET, _CLASS_IPUT, _CLASS_SGET, _CLASS_SPUT):
        # Field access: field index
        if len(operand_bytes) >= 2:
            field_idx = struct.unpack_from("<H", operand_bytes, 0)[0]
            if n_fields > 0 and field_idx >= n_fields:
                result.append(("FIELD_IDX_BAD", True))
            else:
                result.append(("FIELD_IDX_OK", False))
            result.append(("REG", False))

    elif opcode_class in (_CLASS_CONST,):
        # Constant: classify size
        if len(operand_bytes) <= 2:
            result.append(("CONST_SMALL", False))
        else:
            result.append(("CONST_LARGE", False))

    elif opcode_class in (_CLASS_GOTO, _CLASS_IF):
        # Branch: check if target is within data bounds
        if len(operand_bytes) >= 2:
            branch_offset = struct.unpack_from("<h", operand_bytes, 0)[0]
            target = offset + branch_offset * 2
            if 0 <= target < data_len:
                result.append(("BRANCH_OK", False))
            else:
                result.append(("BRANCH_BAD", True))

    elif opcode_class in (_CLASS_NEW, _CLASS_CHECK, "instance_of", "new_array"):
        # Type index
        if len(operand_bytes) >= 2:
            type_idx = struct.unpack_from("<H", operand_bytes, 0)[0]
            if n_types > 0 and type_idx >= n_types:
                result.append(("TYPE_IDX_BAD", True))
            else:
                result.append(("TYPE_IDX_OK", False))

    else:
        # Generic: register operands
        result.append(("REG", False))

    return result if result else [("REG", False)]


def _invoke_kind_token(opcode: int) -> str:
    if opcode == 0x6e:
        return "INVOKE_VIRTUAL"
    if opcode == 0x6f:
        return "INVOKE_SUPER"
    if opcode == 0x70:
        return "INVOKE_DIRECT"
    if opcode == 0x71:
        return "INVOKE_STATIC"
    if opcode == 0x72:
        return "INVOKE_INTERFACE"
    if 0x74 <= opcode <= 0x78:
        return "INVOKE_RANGE"
    return "INVOKE_VIRTUAL"
