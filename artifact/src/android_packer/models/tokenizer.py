"""Byte-level tokenizer for DexBERT-Loc.

This tokenizer converts raw byte streams (e.g. 4 KiB regions extracted by
``android_packer.regioning``) into fixed-length integer sequences suitable for
the byte-level RoBERTa encoder defined in the Ours method spec.

Design requirements (see ``docs/method/ours_method_spec.md`` §3.2.2):

* Closed 261-token vocabulary: 5 special tokens + 256 raw byte tokens.
* Pure standard library — no dependency on torch, tokenizers, or transformers.
  This keeps the module importable in the zero-dependency core environment and
  defers heavy imports to downstream encoder / training modules.
* ``encode`` / ``decode`` are lossless round trips for arbitrary byte payloads
  (subject to the declared truncation policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


__all__ = ["ByteTokenizer", "ByteTokenizerEncoding"]


@dataclass(frozen=True)
class ByteTokenizerEncoding:
    """Result of :meth:`ByteTokenizer.encode_with_mask`.

    Attributes
    ----------
    input_ids:
        Token ids padded (or truncated) to the requested ``max_length``.
    attention_mask:
        ``1`` for real tokens (including ``BOS``/``EOS``), ``0`` for padding.
    length:
        Number of non-pad tokens, i.e. ``sum(attention_mask)``.
    truncated:
        ``True`` when the raw byte stream was longer than the budget allowed
        and trailing bytes were dropped.
    """

    input_ids: List[int]
    attention_mask: List[int]
    length: int
    truncated: bool


class ByteTokenizer:
    """Deterministic byte-level tokenizer with a closed 261-token vocabulary.

    Token id layout::

        0       PAD
        1       BOS
        2       EOS
        3       MASK
        4       UNK
        5..260  raw byte values 0x00..0xFF (id = 5 + byte)

    The tokenizer is intentionally minimal: it performs no normalisation,
    no sub-word merging, and does not allocate any learned state. It only
    maps ``bytes <-> list[int]`` in both directions.
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    MASK_ID: int = 3
    UNK_ID: int = 4
    BYTE_OFFSET: int = 5
    VOCAB_SIZE: int = 5 + 256  # 261

    # Ranges exposed for downstream consumers (e.g. MLM collator avoiding
    # masking of special tokens).
    SPECIAL_IDS: "frozenset[int]" = frozenset({PAD_ID, BOS_ID, EOS_ID, MASK_ID, UNK_ID})

    def __init__(self, max_length: Optional[int] = None) -> None:
        """Create a tokenizer.

        Parameters
        ----------
        max_length:
            Optional default budget for :meth:`encode` / :meth:`encode_with_mask`
            when no explicit ``max_length`` is passed. ``None`` means "no
            truncation" (the caller must ensure inputs fit).
        """

        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be a positive integer or None")
        self._default_max_length = max_length

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        data: bytes,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode ``data`` into a list of token ids.

        When ``max_length`` is supplied (or a default is set on the tokenizer)
        the output is truncated from the right. ``BOS``/``EOS`` markers, when
        enabled, are preserved across truncation: the stream keeps its leading
        ``BOS`` and closes with ``EOS`` as long as the budget permits at least
        those two tokens.

        No padding is added here — use :meth:`encode_with_mask` if you need a
        fixed-length tensor-ready representation.
        """

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"ByteTokenizer.encode expected bytes-like input, got {type(data).__name__}"
            )

        raw = bytes(data)
        budget = max_length if max_length is not None else self._default_max_length

        if add_special_tokens:
            # Reserve two slots for BOS/EOS when a budget is present.
            if budget is None:
                body = [self.BYTE_OFFSET + b for b in raw]
            else:
                if budget < 2:
                    # Degenerate budget: emit as many special markers as fit.
                    return [self.BOS_ID][:budget]
                body_budget = budget - 2
                body = [self.BYTE_OFFSET + b for b in raw[:body_budget]]
            return [self.BOS_ID, *body, self.EOS_ID]

        if budget is None:
            return [self.BYTE_OFFSET + b for b in raw]
        return [self.BYTE_OFFSET + b for b in raw[:budget]]

    def encode_with_mask(
        self,
        data: bytes,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> ByteTokenizerEncoding:
        """Encode ``data`` and right-pad to ``max_length``.

        Returns a :class:`ByteTokenizerEncoding` carrying ``input_ids``,
        ``attention_mask`` and metadata (``length``, ``truncated``). This is
        the preferred entrypoint for training / inference pipelines that need
        fixed-length tensors.
        """

        budget = max_length if max_length is not None else self._default_max_length
        if budget is None:
            raise ValueError(
                "encode_with_mask requires an explicit max_length (or a default "
                "max_length configured on the tokenizer)"
            )

        raw = bytes(data)
        overhead = 2 if add_special_tokens else 0
        body_budget = max(budget - overhead, 0)
        truncated = len(raw) > body_budget

        ids = self.encode(
            raw, add_special_tokens=add_special_tokens, max_length=budget
        )
        actual_length = len(ids)
        if actual_length < budget:
            pad_count = budget - actual_length
            ids = ids + [self.PAD_ID] * pad_count
            mask = [1] * actual_length + [0] * pad_count
        else:
            mask = [1] * budget

        return ByteTokenizerEncoding(
            input_ids=ids,
            attention_mask=mask,
            length=actual_length,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> bytes:
        """Decode ``ids`` back into raw bytes.

        By default all special tokens (PAD/BOS/EOS/MASK/UNK) are dropped so
        the returned payload matches the original byte stream. Set
        ``skip_special_tokens=False`` to replace special tokens with ``0x00``
        (mostly useful for debugging encoded sequences).
        """

        if isinstance(ids, (bytes, bytearray, memoryview, str)):
            raise TypeError(
                "ByteTokenizer.decode expects a sequence of ints, not bytes/str"
            )

        out = bytearray()
        for tok in ids:
            if not isinstance(tok, int):
                raise TypeError(
                    f"token id must be int, got {type(tok).__name__}"
                )
            if tok < 0 or tok >= self.VOCAB_SIZE:
                raise ValueError(
                    f"token id {tok} is out of range [0, {self.VOCAB_SIZE})"
                )
            if tok < self.BYTE_OFFSET:
                if skip_special_tokens:
                    continue
                out.append(0x00)
                continue
            out.append(tok - self.BYTE_OFFSET)
        return bytes(out)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    def encode_batch(
        self,
        batch: Iterable[bytes],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> List[ByteTokenizerEncoding]:
        """Convenience wrapper around :meth:`encode_with_mask` for batches."""

        return [
            self.encode_with_mask(
                item,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
            for item in batch
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def default_max_length(self) -> Optional[int]:
        return self._default_max_length

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ByteTokenizer(vocab_size={self.VOCAB_SIZE}, "
            f"default_max_length={self._default_max_length})"
        )
