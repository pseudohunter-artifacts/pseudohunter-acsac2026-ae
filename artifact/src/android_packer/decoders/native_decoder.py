"""Linear ARM64/ARM32 native instruction decoder for pseudo-code BERT.

Performs linear disassembly of raw bytes as ARM instructions.
ARM64 uses fixed 4-byte instructions, ARM32 uses variable (2/4-byte Thumb or 4-byte ARM).

For packer detection, we primarily care about ARM64 (arm64-v8a) since most
modern Android devices use it. ARM32 is handled as a fallback.

From pseudo_code_bert_packed_apk_framework.md §8:
- Decode each 4-byte unit as an ARM64 instruction
- Classify into high-level instruction classes (load, store, branch, ...)
- Track abnormality (undefined encodings, impossible targets)

Design contract:
- Pure stdlib (no capstone/keystone dependency)
- Lightweight: only classifies instruction TYPE, not full disassembly
- Works on ANY bytes (not just valid ELF .text)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "NativeToken",
    "decode_native_region",
    "NATIVE_TOKEN_VOCAB",
    "NATIVE_TYPED_COMPONENT_TOKENS",
    "native_token_to_id",
]


# ---------------------------------------------------------------------------
# ARM64 instruction classification (top-level encoding groups)
# Reference: ARM Architecture Reference Manual ARMv8-A
# https://developer.arm.com/documentation/ddi0487/latest
#
# ARM64 instruction encoding uses bits [28:25] as the main group selector:
#   0000 = Reserved / unallocated
#   100x = Data processing (immediate)
#   101x = Branch / exception / system
#   x1x0 = Loads and stores
#   x101 = Data processing (register)
#   0111 = Data processing (SIMD/FP)
# ---------------------------------------------------------------------------


def _classify_arm64(insn: int) -> str:
    """Classify a 32-bit ARM64 instruction into a high-level class."""
    # Extract op0 field: bits [28:25]
    op0 = (insn >> 25) & 0xF

    # Unallocated
    if op0 == 0b0000:
        return "reserved"

    # Data processing - immediate
    if (op0 >> 1) == 0b100:  # 100x
        op_sub = (insn >> 23) & 0x7
        if op_sub in (0, 1):
            return "pc_rel"      # PC-relative addressing (ADR/ADRP)
        elif op_sub in (2, 3):
            return "arith_imm"   # ADD/SUB immediate
        elif op_sub == 4:
            return "logic_imm"   # AND/ORR/EOR immediate
        elif op_sub == 5:
            return "move_imm"    # MOVZ/MOVN/MOVK
        elif op_sub == 6:
            return "bitfield"    # BFM/UBFM/SBFM
        else:
            return "extract"     # EXTR

    # Branch, exception, system
    if (op0 >> 1) == 0b101:  # 101x
        op1 = (insn >> 29) & 0x7
        if op1 in (0, 4):
            return "branch_uncond"  # B, BL
        elif op1 in (1, 5):
            return "branch_cmp"     # CBZ/CBNZ, TBZ/TBNZ
        elif op1 == 2:
            return "branch_cond"    # B.cond
        elif op1 == 6:
            # System / exception
            op2 = (insn >> 22) & 0x7
            if op2 == 0:
                return "exception"   # SVC, HVC, SMC, BRK
            else:
                return "system"      # MSR, MRS, barriers
        elif op1 == 3 or op1 == 7:
            return "branch_reg"     # BR, BLR, RET
        return "branch_other"

    # Loads and stores
    if (op0 & 0b0101) == 0b0100:  # x1x0
        op1 = (insn >> 28) & 0x3
        op2 = (insn >> 22) & 0x3
        if (insn >> 27) & 1:
            # Load/store register
            opc = (insn >> 22) & 0x3
            if opc & 1:
                return "load"
            else:
                return "store"
        else:
            # Load/store pair, exclusive, ordered
            if (insn >> 22) & 1:
                return "load_pair"
            else:
                return "store_pair"

    # Data processing - register
    if (op0 & 0b0111) == 0b0101:  # x101
        op1 = (insn >> 28) & 1
        op2 = (insn >> 21) & 0xF
        if op1 == 0:
            if op2 < 8:
                return "logic_reg"   # AND/ORR/EOR register
            else:
                return "arith_reg"   # ADD/SUB register (shifted)
        else:
            if op2 == 6:
                return "cond_select" # CSEL/CSINC/...
            elif op2 >= 8:
                return "data_proc3"  # MADD/MSUB/...
            else:
                return "shift_reg"   # LSL/LSR/ASR register

    # SIMD and floating-point
    if (op0 & 0b1110) == 0b0110:  # 011x or 0111
        return "simd_fp"

    # Anything else
    return "unknown_insn"


def _classify_arm32(insn: int) -> str:
    """Classify a 32-bit ARM32 instruction (simplified)."""
    cond = (insn >> 28) & 0xF
    op1 = (insn >> 25) & 0x7

    if cond == 0xF:
        return "unconditional"  # Unconditional instructions

    if op1 == 0b000 or op1 == 0b001:
        # Data processing
        return "data_proc"
    elif op1 == 0b010 or op1 == 0b011:
        # Load/store word/byte
        if (insn >> 20) & 1:
            return "load"
        else:
            return "store"
    elif op1 == 0b100:
        # Load/store multiple
        if (insn >> 20) & 1:
            return "load_multi"
        else:
            return "store_multi"
    elif op1 == 0b101:
        # Branch
        if (insn >> 24) & 1:
            return "branch_link"
        else:
            return "branch"
    elif op1 == 0b110:
        return "coproc"  # Coprocessor
    elif op1 == 0b111:
        if (insn >> 24) & 1:
            return "svc"  # Software interrupt
        else:
            return "coproc"
    return "unknown_insn"


def _native_boundary_token(insn_class: str) -> str | None:
    if insn_class in {"exception", "svc"}:
        return "SYSCALL_LIKE"
    if insn_class in {"branch_reg", "branch_link"}:
        return "JNI_SYMBOL_LIKE"
    if insn_class == "system":
        return "SYSTEM_SYMBOL_LIKE"
    return None


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]

_INSN_CLASS_TOKENS = [
    # ARM64 classes
    "reserved", "pc_rel", "arith_imm", "logic_imm", "move_imm",
    "bitfield", "extract", "branch_uncond", "branch_cmp", "branch_cond",
    "branch_reg", "branch_other", "exception", "system",
    "load", "store", "load_pair", "store_pair",
    "logic_reg", "arith_reg", "cond_select", "data_proc3", "shift_reg",
    "simd_fp", "unknown_insn",
    # ARM32 classes (additional)
    "unconditional", "data_proc", "load_multi", "store_multi",
    "branch", "branch_link", "coproc", "svc",
    # Native-specific meta
    "INVALID_ENCODING",
]

_OPERAND_TOKENS = [
    "REG", "IMM",
    "MEM_OFFSET_OK", "MEM_OFFSET_BAD",
    "TARGET_OK", "TARGET_BAD",
    "CALL_OK", "CALL_BAD",
]

NATIVE_TYPED_COMPONENT_TOKENS: Tuple[str, ...] = (
    "JNI_SYMBOL_LIKE",
    "SYSCALL_LIKE",
    "SYSTEM_SYMBOL_LIKE",
)

_META_TOKENS = [
    "INSN_END",
    "NOP_NORMAL",       # 0xD503201F (ARM64 NOP)
    "NOP_ABNORMAL",     # non-standard NOP-like encoding
    "DATA_LITERAL",     # likely data, not instruction
    "SEQUENCE_END",
]

NATIVE_TOKEN_VOCAB: Tuple[str, ...] = tuple(
    _SPECIAL_TOKENS + _INSN_CLASS_TOKENS + _OPERAND_TOKENS + _META_TOKENS
)

_TOKEN_TO_ID = {t: i for i, t in enumerate(NATIVE_TOKEN_VOCAB)}


def native_token_to_id(token: str) -> int:
    """Convert token string to integer ID."""
    return _TOKEN_TO_ID.get(token, _TOKEN_TO_ID["[UNK]"])


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeToken:
    """A single decoded pseudo-native token."""
    token: str
    token_id: int
    offset: int
    is_abnormal: bool


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------


def decode_native_region(
    data: bytes,
    *,
    arch: str = "arm64",
    max_tokens: int = 510,
    code_base_addr: int = 0,
    code_size: int = 0,
) -> List[NativeToken]:
    """Linearly decode raw bytes as native (ARM64/ARM32) instructions.

    Args:
        data: Raw bytes to decode
        arch: "arm64" (default) or "arm32"
        max_tokens: Maximum tokens to produce
        code_base_addr: Base address for branch target validation
        code_size: Total code size for branch target validation (0=no validation)

    Returns:
        List of NativeToken with instruction class + abnormality info.
    """
    tokens: List[NativeToken] = []
    tokens.append(NativeToken("[BOS]", _TOKEN_TO_ID["[BOS]"], 0, False))

    # ARM64: fixed 4-byte instructions
    # ARM32: also 4-byte (we skip Thumb mode for simplicity)
    insn_size = 4
    classify_fn = _classify_arm64 if arch == "arm64" else _classify_arm32

    offset = 0
    n_tokens = 0

    while offset + insn_size <= len(data) and n_tokens < max_tokens:
        # Read instruction (little-endian)
        insn = struct.unpack_from("<I", data, offset)[0]

        # Check for NOP
        if arch == "arm64" and insn == 0xD503201F:
            tokens.append(NativeToken(
                "NOP_NORMAL", _TOKEN_TO_ID["NOP_NORMAL"], offset, False
            ))
            n_tokens += 1
            offset += insn_size
            continue

        # Check for zero (likely data/padding, not instruction)
        if insn == 0:
            tokens.append(NativeToken(
                "DATA_LITERAL", _TOKEN_TO_ID["DATA_LITERAL"], offset, False
            ))
            n_tokens += 1
            offset += insn_size
            continue

        # Classify instruction
        insn_class = classify_fn(insn)
        is_abnormal = (insn_class in ("reserved", "unknown_insn", "INVALID_ENCODING"))

        # Emit instruction class token
        token_name = insn_class
        tid = _TOKEN_TO_ID.get(token_name, _TOKEN_TO_ID["unknown_insn"])
        tokens.append(NativeToken(token_name, tid, offset, is_abnormal))
        n_tokens += 1

        if n_tokens < max_tokens:
            symbol_token = _native_boundary_token(insn_class)
            if symbol_token is not None:
                tokens.append(NativeToken(
                    symbol_token,
                    _TOKEN_TO_ID.get(symbol_token, _TOKEN_TO_ID["[UNK]"]),
                    offset,
                    False,
                ))
                n_tokens += 1

        # For branches: check if target is reasonable
        if insn_class in ("branch_uncond", "branch_link", "branch_cmp") and n_tokens < max_tokens:
            # Extract branch offset (bits [25:0] << 2 for B/BL)
            if insn_class == "branch_uncond":
                imm26 = insn & 0x3FFFFFF
                # Sign extend
                if imm26 & (1 << 25):
                    imm26 -= (1 << 26)
                target_offset = imm26 * 4
                target = offset + target_offset

                if code_size > 0:
                    if 0 <= target < code_size:
                        tokens.append(NativeToken(
                            "TARGET_OK", _TOKEN_TO_ID["TARGET_OK"], offset, False
                        ))
                    else:
                        tokens.append(NativeToken(
                            "TARGET_BAD", _TOKEN_TO_ID["TARGET_BAD"], offset, True
                        ))
                    n_tokens += 1

        # Emit instruction end
        if n_tokens < max_tokens:
            tokens.append(NativeToken(
                "INSN_END", _TOKEN_TO_ID["INSN_END"], offset, False
            ))
            n_tokens += 1

        offset += insn_size

    tokens.append(NativeToken("[EOS]", _TOKEN_TO_ID["[EOS]"], offset, False))
    return tokens
