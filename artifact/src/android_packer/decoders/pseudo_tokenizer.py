"""Unified pseudo-code tokenizer combining Dalvik, Native, and Byte vocabularies.

Provides a single token ID space for the shared BERT encoder.
Token type IDs distinguish which stream a token belongs to.

Vocabulary layout:
  [0-4]:     Special tokens (PAD, BOS, EOS, MASK, UNK)
  [5-59]:    Dalvik pseudo-opcode tokens (55 tokens from dalvik_decoder)
  [60-110]:  Native pseudo-instruction tokens (51 tokens from native_decoder)
  [111-371]: Byte tokens (261 from existing ByteTokenizer: 0x00-0xFF + specials)

Token type IDs:
  0 = Dalvik stream
  1 = Native stream
  2 = Byte stream

This unified vocabulary allows a single shared BERT to process all three
streams with token_type_embedding distinguishing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from android_packer.decoders.byte_pattern_decoder import (
    BYTE_PATTERN_TOKEN_VOCAB,
    decode_byte_pattern_region,
)

from android_packer.decoders.dalvik_decoder import (
    DALVIK_TOKEN_VOCAB,
    DALVIK_TYPED_COMPONENT_TOKENS,
    DalvikToken,
    dalvik_token_to_id,
    decode_dalvik_region,
)
from android_packer.decoders.native_decoder import (
    NATIVE_TOKEN_VOCAB,
    NATIVE_TYPED_COMPONENT_TOKENS,
    NativeToken,
    decode_native_region,
    native_token_to_id,
)

__all__ = [
    "PseudoCodeTokenizer",
    "UNIFIED_VOCAB_SIZE",
    "UNIFIED_VOCAB_SIZE_LEGACY_RAW",
    "UNIFIED_VOCAB_SIZE_TYPED_V1",
    "TOKEN_TYPE_DALVIK",
    "TOKEN_TYPE_NATIVE",
    "TOKEN_TYPE_BYTE",
    "BYTE_REPRESENTATION_LEGACY_RAW",
    "BYTE_REPRESENTATION_TYPED_V1",
    "BYTE_REPRESENTATIONS",
    "vocab_size_for_byte_representation",
]

# Token type IDs
TOKEN_TYPE_DALVIK = 0
TOKEN_TYPE_NATIVE = 1
TOKEN_TYPE_BYTE = 2

BYTE_REPRESENTATION_LEGACY_RAW = "legacy_raw"
BYTE_REPRESENTATION_TYPED_V1 = "typed_v1"
BYTE_REPRESENTATIONS = (
    BYTE_REPRESENTATION_LEGACY_RAW,
    BYTE_REPRESENTATION_TYPED_V1,
)

# Shared special tokens (same IDs across all streams)
_SPECIAL_PAD = 0
_SPECIAL_BOS = 1
_SPECIAL_EOS = 2
_SPECIAL_MASK = 3
_SPECIAL_UNK = 4
_N_SPECIAL = 5

# Vocabulary segments
_DALVIK_OFFSET = _N_SPECIAL
_DALVIK_SIZE = len(DALVIK_TOKEN_VOCAB) - _N_SPECIAL  # exclude specials (shared)
_DALVIK_TYPED_SIZE = _DALVIK_SIZE + len(DALVIK_TYPED_COMPONENT_TOKENS)

_NATIVE_OFFSET = _DALVIK_OFFSET + _DALVIK_SIZE
_NATIVE_SIZE = len(NATIVE_TOKEN_VOCAB) - _N_SPECIAL
_NATIVE_TYPED_OFFSET = _DALVIK_OFFSET + _DALVIK_TYPED_SIZE
_NATIVE_TYPED_SIZE = _NATIVE_SIZE + len(NATIVE_TYPED_COMPONENT_TOKENS)

_BYTE_OFFSET = _NATIVE_OFFSET + _NATIVE_SIZE
_BYTE_TYPED_OFFSET = _NATIVE_TYPED_OFFSET + _NATIVE_TYPED_SIZE
_BYTE_RAW_SIZE = 256  # raw byte values 0x00-0xFF
_BYTE_PATTERN_SIZE = len(BYTE_PATTERN_TOKEN_VOCAB) - _N_SPECIAL

UNIFIED_VOCAB_SIZE_LEGACY_RAW = _N_SPECIAL + _DALVIK_SIZE + _NATIVE_SIZE + _BYTE_RAW_SIZE
UNIFIED_VOCAB_SIZE_TYPED_V1 = (
    _N_SPECIAL + _DALVIK_TYPED_SIZE + _NATIVE_TYPED_SIZE + _BYTE_PATTERN_SIZE
)

# Backward-compatible default for existing checkpoints/results.
UNIFIED_VOCAB_SIZE = UNIFIED_VOCAB_SIZE_LEGACY_RAW


def vocab_size_for_byte_representation(byte_representation: str) -> int:
    """Return the unified vocab size for a byte representation version."""
    if byte_representation == BYTE_REPRESENTATION_LEGACY_RAW:
        return UNIFIED_VOCAB_SIZE_LEGACY_RAW
    if byte_representation == BYTE_REPRESENTATION_TYPED_V1:
        return UNIFIED_VOCAB_SIZE_TYPED_V1
    raise ValueError(
        f"unknown byte_representation {byte_representation!r}; "
        f"expected one of {BYTE_REPRESENTATIONS}"
    )


@dataclass(frozen=True)
class EncodedSequence:
    """Result of encoding a region through one of the three decoders."""
    token_ids: List[int]       # unified vocab IDs
    token_type_ids: List[int]  # all same value (0/1/2)
    attention_mask: List[int]  # 1 for real tokens, 0 for padding
    n_abnormal: int            # count of abnormal tokens (for quick stats)
    length: int                # actual length before padding


class PseudoCodeTokenizer:
    """Unified tokenizer for the three-path pseudo-code BERT.

    Encodes region bytes into three parallel token sequences:
    1. Dalvik pseudo-opcodes (for DEX-like regions)
    2. Native pseudo-instructions (for ELF-like regions)
    3. Byte-path tokens (legacy raw bytes or typed byte patterns)
    """

    def __init__(
        self,
        max_length: int = 512,
        *,
        byte_representation: str = BYTE_REPRESENTATION_LEGACY_RAW,
    ):
        self.max_length = max_length
        if byte_representation not in BYTE_REPRESENTATIONS:
            raise ValueError(
                f"unknown byte_representation {byte_representation!r}; "
                f"expected one of {BYTE_REPRESENTATIONS}"
            )
        self.byte_representation = byte_representation
        self.vocab_size = vocab_size_for_byte_representation(byte_representation)

        # Build Dalvik ID mapping (skip specials, offset by _DALVIK_OFFSET)
        self._dalvik_map = {}
        for i, tok in enumerate(DALVIK_TOKEN_VOCAB):
            if i < _N_SPECIAL:
                self._dalvik_map[tok] = i  # specials stay at 0-4
            else:
                self._dalvik_map[tok] = _DALVIK_OFFSET + (i - _N_SPECIAL)
        if byte_representation == BYTE_REPRESENTATION_TYPED_V1:
            for i, tok in enumerate(DALVIK_TYPED_COMPONENT_TOKENS):
                self._dalvik_map[tok] = _DALVIK_OFFSET + _DALVIK_SIZE + i

        # Build Native ID mapping
        self._native_map = {}
        native_offset = (
            _NATIVE_OFFSET
            if byte_representation == BYTE_REPRESENTATION_LEGACY_RAW
            else _NATIVE_TYPED_OFFSET
        )
        for i, tok in enumerate(NATIVE_TOKEN_VOCAB):
            if i < _N_SPECIAL:
                self._native_map[tok] = i
            else:
                self._native_map[tok] = native_offset + (i - _N_SPECIAL)
        if byte_representation == BYTE_REPRESENTATION_TYPED_V1:
            for i, tok in enumerate(NATIVE_TYPED_COMPONENT_TOKENS):
                self._native_map[tok] = _NATIVE_TYPED_OFFSET + _NATIVE_SIZE + i

        self._byte_pattern_map = {}
        byte_pattern_offset = (
            _BYTE_OFFSET
            if byte_representation == BYTE_REPRESENTATION_LEGACY_RAW
            else _BYTE_TYPED_OFFSET
        )
        for i, tok in enumerate(BYTE_PATTERN_TOKEN_VOCAB):
            if i < _N_SPECIAL:
                self._byte_pattern_map[tok] = i
            else:
                self._byte_pattern_map[tok] = byte_pattern_offset + (i - _N_SPECIAL)

    def encode_dalvik(
        self,
        data: bytes,
        *,
        string_ids_count: int = 0,
        type_ids_count: int = 0,
        method_ids_count: int = 0,
        field_ids_count: int = 0,
    ) -> EncodedSequence:
        """Encode region bytes as Dalvik pseudo-code tokens."""
        tokens = decode_dalvik_region(
            data,
            max_tokens=self.max_length - 2,  # room for BOS/EOS
            string_ids_count=string_ids_count,
            type_ids_count=type_ids_count,
            method_ids_count=method_ids_count,
            field_ids_count=field_ids_count,
        )

        # Convert to unified IDs
        ids = []
        for t in tokens:
            unified_id = self._dalvik_map.get(t.token, _SPECIAL_UNK)
            ids.append(unified_id)

        n_abnormal = sum(1 for t in tokens if t.is_abnormal)
        return self._pad_and_wrap(ids, TOKEN_TYPE_DALVIK, n_abnormal)

    def encode_native(
        self,
        data: bytes,
        *,
        arch: str = "arm64",
        code_size: int = 0,
    ) -> EncodedSequence:
        """Encode region bytes as native pseudo-instruction tokens."""
        tokens = decode_native_region(
            data,
            arch=arch,
            max_tokens=self.max_length - 2,
            code_size=code_size,
        )

        ids = []
        for t in tokens:
            unified_id = self._native_map.get(t.token, _SPECIAL_UNK)
            ids.append(unified_id)

        n_abnormal = sum(1 for t in tokens if t.is_abnormal)
        return self._pad_and_wrap(ids, TOKEN_TYPE_NATIVE, n_abnormal)

    def encode_bytes(self, data: bytes, *, entry_type: str = "unknown") -> EncodedSequence:
        """Encode region bytes as byte-path tokens."""
        if self.byte_representation == BYTE_REPRESENTATION_TYPED_V1:
            tokens = decode_byte_pattern_region(
                data,
                entry_type=entry_type,
                max_tokens=self.max_length - 2,
            )
            ids = [
                self._byte_pattern_map.get(token.token, _SPECIAL_UNK)
                for token in tokens
            ]
            n_abnormal = sum(1 for token in tokens if token.is_abnormal)
            return self._pad_and_wrap(ids, TOKEN_TYPE_BYTE, n_abnormal)

        # BOS + up to (max_length-2) bytes + EOS
        budget = self.max_length - 2
        payload = data[:budget]

        ids = [_SPECIAL_BOS]
        for b in payload:
            ids.append(_BYTE_OFFSET + b)  # byte 0x00 → _BYTE_OFFSET, ..., 0xFF → _BYTE_OFFSET+255
        ids.append(_SPECIAL_EOS)

        return self._pad_and_wrap(ids, TOKEN_TYPE_BYTE, 0)

    def _pad_and_wrap(
        self,
        ids: List[int],
        token_type: int,
        n_abnormal: int,
    ) -> EncodedSequence:
        """Pad/truncate to max_length and create attention mask."""
        length = len(ids)

        if length > self.max_length:
            ids = ids[:self.max_length - 1] + [_SPECIAL_EOS]
            length = self.max_length

        attention_mask = [1] * length + [0] * (self.max_length - length)
        ids = ids + [_SPECIAL_PAD] * (self.max_length - length)
        token_type_ids = [token_type] * self.max_length

        return EncodedSequence(
            token_ids=ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            n_abnormal=n_abnormal,
            length=length,
        )

    def encode_region(
        self,
        data: bytes,
        *,
        entry_type: str = "unknown",
        dex_header_counts: Tuple[int, int, int, int] = (0, 0, 0, 0),
        arch: str = "arm64",
    ) -> Tuple[EncodedSequence, EncodedSequence, EncodedSequence]:
        """Encode a region through all three paths simultaneously.

        Args:
            data: Raw bytes of the region
            entry_type: Coarse entry type (from typed_slicer)
            dex_header_counts: (string_ids, type_ids, method_ids, field_ids)
            arch: Native architecture for ARM decoding

        Returns:
            Tuple of (dalvik_encoded, native_encoded, byte_encoded)
        """
        s_count, t_count, m_count, f_count = dex_header_counts

        dalvik_enc = self.encode_dalvik(
            data,
            string_ids_count=s_count,
            type_ids_count=t_count,
            method_ids_count=m_count,
            field_ids_count=f_count,
        )

        native_enc = self.encode_native(data, arch=arch)
        byte_enc = self.encode_bytes(data, entry_type=entry_type)

        return dalvik_enc, native_enc, byte_enc
