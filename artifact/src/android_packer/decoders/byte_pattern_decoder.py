"""Typed byte-pattern decoder for pseudo-code BERT.

The byte path is a fallback view: it should expose generic APK-object byte
patterns without making raw high entropy a payload shortcut. This decoder emits
coarse pattern tokens such as magic/header, entropy class, compression-like
shape, run-length shape, repetition, alignment, and entry type context.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

__all__ = [
    "BytePatternToken",
    "BYTE_PATTERN_TOKEN_VOCAB",
    "byte_pattern_token_to_id",
    "decode_byte_pattern_region",
]


_SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]

_ENTRY_TOKENS = [
    "ENTRY_DEX",
    "ENTRY_ELF",
    "ENTRY_ARCHIVE",
    "ENTRY_ASSET",
    "ENTRY_ARSC",
    "ENTRY_MANIFEST",
    "ENTRY_RESOURCE",
    "ENTRY_UNKNOWN",
]

_MAGIC_TOKENS = [
    "MAGIC_DEX",
    "MAGIC_ELF",
    "MAGIC_ZIP",
    "MAGIC_GZIP",
    "MAGIC_PNG",
    "MAGIC_SQLITE",
    "MAGIC_XML_TEXT",
    "MAGIC_NONE",
]

_ENTROPY_TOKENS = [
    "ENTROPY_LOW",
    "ENTROPY_MEDIUM",
    "ENTROPY_HIGH",
]

_SHAPE_TOKENS = [
    "ASCII_TEXT",
    "ASCII_MIXED",
    "BINARY_DENSE",
    "BASE64_LIKE",
    "COMPRESS_ZLIB_LIKE",
    "COMPRESS_ZIP_LIKE",
    "ENCRYPTION_LIKE_HIGH_ENTROPY",
    "RUN_ZERO_LONG",
    "RUN_FF_LONG",
    "RUN_BYTE_LONG",
    "RUN_SHORT",
    "REPEAT_2GRAM",
    "REPEAT_4GRAM",
    "ALIGN_ZERO_PAD",
    "ALIGN_FF_PAD",
    "UNKNOWN_PATTERN",
]

_META_TOKENS = ["PATTERN_END", "SEQUENCE_END"]

BYTE_PATTERN_TOKEN_VOCAB: Tuple[str, ...] = tuple(
    _SPECIAL_TOKENS
    + _ENTRY_TOKENS
    + _MAGIC_TOKENS
    + _ENTROPY_TOKENS
    + _SHAPE_TOKENS
    + _META_TOKENS
)

_TOKEN_TO_ID = {token: idx for idx, token in enumerate(BYTE_PATTERN_TOKEN_VOCAB)}


@dataclass(frozen=True)
class BytePatternToken:
    """A single typed byte-pattern token."""

    token: str
    token_id: int
    offset: int
    is_abnormal: bool


def byte_pattern_token_to_id(token: str) -> int:
    """Convert token string to integer ID."""
    return _TOKEN_TO_ID.get(token, _TOKEN_TO_ID["[UNK]"])


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def _entry_token(entry_type: str) -> str:
    normalized = (entry_type or "unknown").lower()
    if normalized == "dex":
        return "ENTRY_DEX"
    if normalized == "elf":
        return "ENTRY_ELF"
    if normalized == "archive":
        return "ENTRY_ARCHIVE"
    if normalized == "asset":
        return "ENTRY_ASSET"
    if normalized == "arsc":
        return "ENTRY_ARSC"
    if normalized == "manifest":
        return "ENTRY_MANIFEST"
    if normalized == "resource":
        return "ENTRY_RESOURCE"
    return "ENTRY_UNKNOWN"


def _magic_tokens(data: bytes) -> List[str]:
    if data.startswith(b"dex\n"):
        return ["MAGIC_DEX"]
    if data.startswith(b"\x7fELF"):
        return ["MAGIC_ELF"]
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
        return ["MAGIC_ZIP", "COMPRESS_ZIP_LIKE"]
    if data.startswith(b"\x1f\x8b"):
        return ["MAGIC_GZIP"]
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ["MAGIC_PNG"]
    if data.startswith(b"SQLite format 3\x00"):
        return ["MAGIC_SQLITE"]
    stripped = data[:64].lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<manifest"):
        return ["MAGIC_XML_TEXT"]
    return ["MAGIC_NONE"]


def _entropy_token(entropy: float) -> str:
    if entropy < 3.5:
        return "ENTROPY_LOW"
    if entropy < 6.5:
        return "ENTROPY_MEDIUM"
    return "ENTROPY_HIGH"


def _longest_run(data: bytes) -> Tuple[int, int]:
    if not data:
        return 0, 0
    best_byte = data[0]
    best_len = 1
    current_byte = data[0]
    current_len = 1
    for byte in data[1:]:
        if byte == current_byte:
            current_len += 1
        else:
            if current_len > best_len:
                best_byte = current_byte
                best_len = current_len
            current_byte = byte
            current_len = 1
    if current_len > best_len:
        best_byte = current_byte
        best_len = current_len
    return best_byte, best_len


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(1 for b in data if b in (9, 10, 13) or 32 <= b <= 126)
    return printable / len(data)


def _has_repeated_ngram(data: bytes, n: int) -> bool:
    if len(data) < n * 4:
        return False
    grams = [data[i:i + n] for i in range(0, len(data) - n + 1, n)]
    if not grams:
        return False
    most_common = Counter(grams).most_common(1)[0][1]
    return most_common >= max(4, len(grams) // 8)


def _base64_like(data: bytes) -> bool:
    if len(data) < 32:
        return False
    allowed = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n")
    hits = sum(1 for b in data if b in allowed)
    return hits / len(data) > 0.92


def _shape_tokens(data: bytes, entropy: float) -> List[str]:
    tokens: List[str] = []
    if not data:
        return ["UNKNOWN_PATTERN"]

    printable_ratio = _printable_ratio(data)
    if printable_ratio > 0.85:
        tokens.append("ASCII_TEXT")
    elif printable_ratio > 0.35:
        tokens.append("ASCII_MIXED")
    else:
        tokens.append("BINARY_DENSE")

    if _base64_like(data):
        tokens.append("BASE64_LIKE")
    if len(data) >= 2 and data[:2] in {b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"}:
        tokens.append("COMPRESS_ZLIB_LIKE")
    if data.startswith(b"PK"):
        tokens.append("COMPRESS_ZIP_LIKE")
    if entropy >= 7.2 and printable_ratio < 0.35:
        tokens.append("ENCRYPTION_LIKE_HIGH_ENTROPY")

    run_byte, run_len = _longest_run(data)
    if run_len >= 16:
        if run_byte == 0:
            tokens.append("RUN_ZERO_LONG")
        elif run_byte == 0xFF:
            tokens.append("RUN_FF_LONG")
        else:
            tokens.append("RUN_BYTE_LONG")
    else:
        tokens.append("RUN_SHORT")

    if _has_repeated_ngram(data, 2):
        tokens.append("REPEAT_2GRAM")
    if _has_repeated_ngram(data, 4):
        tokens.append("REPEAT_4GRAM")
    if len(data) >= 16 and data[-16:].count(0) >= 12:
        tokens.append("ALIGN_ZERO_PAD")
    if len(data) >= 16 and data[-16:].count(0xFF) >= 12:
        tokens.append("ALIGN_FF_PAD")

    return tokens or ["UNKNOWN_PATTERN"]


def _append_tokens(
    output: List[BytePatternToken],
    tokens: Iterable[str],
    offset: int,
    *,
    abnormal: bool = False,
    budget: int,
) -> None:
    for token in tokens:
        if len(output) - 1 >= budget:
            break
        output.append(
            BytePatternToken(token, _TOKEN_TO_ID.get(token, _TOKEN_TO_ID["[UNK]"]), offset, abnormal)
        )


def decode_byte_pattern_region(
    data: bytes,
    *,
    entry_type: str = "unknown",
    max_tokens: int = 510,
) -> List[BytePatternToken]:
    """Decode raw bytes into typed pattern tokens.

    Entropy tokens are descriptive only and are not marked abnormal. This keeps
    high entropy from becoming a hard-coded payload shortcut.
    """
    tokens: List[BytePatternToken] = [
        BytePatternToken("[BOS]", _TOKEN_TO_ID["[BOS]"], 0, False)
    ]
    budget = max(0, max_tokens)

    _append_tokens(tokens, [_entry_token(entry_type)], 0, budget=budget)
    _append_tokens(tokens, _magic_tokens(data), 0, budget=budget)
    entropy = _entropy(data)
    _append_tokens(tokens, [_entropy_token(entropy)], 0, budget=budget)
    _append_tokens(tokens, _shape_tokens(data, entropy), 0, budget=budget)
    _append_tokens(tokens, ["PATTERN_END"], len(data), budget=budget)

    tokens.append(BytePatternToken("[EOS]", _TOKEN_TO_ID["[EOS]"], len(data), False))
    return tokens
