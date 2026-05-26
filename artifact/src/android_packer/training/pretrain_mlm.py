"""Byte-level MLM pre-training with DEX grammar-aware auxiliary loss.

Batch **F-MIL-c** per ``docs/method/ours_method_spec.md`` §5.1 / §12.4 and
``docs/research_framing.md`` §4.2 sellpoint 2.

What lives here
---------------
1. :class:`MLMCorpusBuilder` — takes an iterable of raw DEX buffers, runs
   them through :func:`android_packer.features.dex_item_parser.parse_dex_item_spans`,
   drops any buffer that fails to parse (packed / truncated / non-benign)
   and surfaces the **benign-only** constraint as a hard error when the
   exclusion ratio exceeds a configurable threshold (default 5 %).
   This mirrors the §8 contract "`L_mlm` is never fit on packed data."
2. :class:`MLMCollator` — 15 % token-level masking (80/10/10 split per
   Devlin et al. 2018) with per-token DEX item-type labels derived from
   the parser spans so we can fold in the auxiliary cross-entropy::

       L = L_mlm + λ_item · L_item_type

   PAD / BOS / EOS / MASK positions are *never* masked by MLM and carry
   ``DEX_ITEM_TYPE_PAD_ID`` on the item-type side so ``ignore_index``
   keeps them out of both losses.
3. :class:`PretrainMLMConfig` / :func:`compute_pretrain_loss` — entry
   points for the trainer (``android-packer-pretrain-mlm`` CLI).

Lazy-torch contract: torch is only pulled in by
:func:`compute_pretrain_loss` and the CLI.  Dataset building, collating
and stats tracking are **pure stdlib** so unit tests can exercise the
corpus pipeline without installing ``[dl]``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from android_packer.features.dex_item_parser import (
    DEX_ITEM_TYPES,
    DexParseError,
    parse_dex_item_spans,
    region_item_type_labels,
)
from android_packer.models.item_type_head import DEX_ITEM_TYPE_PAD_ID
from android_packer.models.tokenizer import ByteTokenizer


__all__ = [
    "BenignCorpusError",
    "MLMBatch",
    "MLMCollator",
    "MLMCorpusBuilder",
    "MLMCorpusStats",
    "MLMExample",
    "PretrainMLMConfig",
    "build_mlm_corpus",
    "build_mlm_example",
    "compute_pretrain_loss",
]


class BenignCorpusError(RuntimeError):
    """Raised when the benign-only MLM corpus constraint is violated.

    Two distinct trigger conditions:

    * Too many buffers failed to parse as benign DEX (see
      :attr:`PretrainMLMConfig.benign_exclusion_max_ratio`).  The trainer
      refuses to proceed because silently pre-training on 20 % packed
      DEX would destroy sellpoint 2.
    * The corpus is empty after filtering, i.e. *every* input buffer
      looked packed.  This is almost certainly a data-wiring bug.
    """


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PretrainMLMConfig:
    """Hyper-parameters for the byte-MLM + grammar-aux pre-training stage.

    Parameters
    ----------
    max_seq_length:
        Token budget per sequence after BOS/EOS.  4 KiB + 2 = 4098 is the
        default window for DEX regions in §3.2.3 of the spec.
    mlm_mask_prob:
        Fraction of non-special tokens chosen for MLM labels (Devlin
        et al. 0.15).
    mlm_replace_mask_prob / mlm_replace_random_prob:
        Of the *selected* tokens: 80 % get replaced by ``MASK``, 10 % by
        a random byte, 10 % stay unchanged (left implicit: 1 - mask -
        random).
    item_type_aux_weight:
        ``λ_item`` — weight on ``L_item_type``.  Default 0.2 matches
        ``ours_method_spec.md`` §12.4.  Setting this to ``0.0`` disables
        the grammar-aware auxiliary loss for ablation runs (§12.5).
    benign_exclusion_max_ratio:
        Hard cap on ``dropped / total``.  Default 5 % enforces the
        "benign-only" invariant from §8.  Set to ``1.0`` in unit tests
        that want to stress-test the corpus builder.
    seed:
        Deterministic masking for reproducible experiments (§11 of the
        spec requires seeded runs).
    """

    max_seq_length: int = 4098
    mlm_mask_prob: float = 0.15
    mlm_replace_mask_prob: float = 0.8
    mlm_replace_random_prob: float = 0.1
    item_type_aux_weight: float = 0.2
    benign_exclusion_max_ratio: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_seq_length <= 2:
            raise ValueError(
                f"max_seq_length must be > 2 to hold BOS/EOS + ≥ 1 byte, "
                f"got {self.max_seq_length}"
            )
        if not (0.0 < self.mlm_mask_prob < 1.0):
            raise ValueError(
                f"mlm_mask_prob must be in (0, 1), got {self.mlm_mask_prob}"
            )
        if not (0.0 <= self.mlm_replace_mask_prob <= 1.0):
            raise ValueError(
                "mlm_replace_mask_prob must be in [0, 1], got "
                f"{self.mlm_replace_mask_prob}"
            )
        if not (0.0 <= self.mlm_replace_random_prob <= 1.0):
            raise ValueError(
                "mlm_replace_random_prob must be in [0, 1], got "
                f"{self.mlm_replace_random_prob}"
            )
        if self.mlm_replace_mask_prob + self.mlm_replace_random_prob > 1.0 + 1e-9:
            raise ValueError(
                "mlm_replace_mask_prob + mlm_replace_random_prob must be <= 1"
            )
        if self.item_type_aux_weight < 0.0:
            raise ValueError(
                f"item_type_aux_weight must be >= 0, got {self.item_type_aux_weight}"
            )
        if not (0.0 <= self.benign_exclusion_max_ratio <= 1.0):
            raise ValueError(
                "benign_exclusion_max_ratio must be in [0, 1], got "
                f"{self.benign_exclusion_max_ratio}"
            )


@dataclass(frozen=True)
class MLMExample:
    """A single pre-training example (before masking).

    Attributes
    ----------
    input_ids:
        Padded token ids (length == ``max_seq_length``).
    attention_mask:
        1 for real tokens, 0 for PAD.
    item_type_labels:
        Per-token DEX item-type id (0..``len(DEX_ITEM_TYPES)-1``) for
        real body bytes, and :data:`DEX_ITEM_TYPE_PAD_ID` for BOS / EOS /
        PAD / positions past the raw-byte region.
    source_length:
        Number of body bytes actually encoded (== ``min(len(dex),
        max_seq_length - 2)``).
    """

    input_ids: List[int]
    attention_mask: List[int]
    item_type_labels: List[int]
    source_length: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MLMCorpusStats:
    """Bookkeeping for the corpus builder.

    Reported by :class:`MLMCorpusBuilder.finalise` and logged by the CLI
    so reviewers can audit the benign-only claim end-to-end.
    """

    total_seen: int
    kept: int
    dropped_parse_error: int
    dropped_empty: int

    @property
    def exclusion_ratio(self) -> float:
        if self.total_seen == 0:
            return 0.0
        return (self.total_seen - self.kept) / self.total_seen


@dataclass(frozen=True)
class MLMBatch:
    """Output of :meth:`MLMCollator.collate` — ready for torch stacking.

    All fields are python ``List[List[int]]`` (pure stdlib).  The trainer
    converts them to tensors at the last mile.  ``labels`` uses ``-100``
    (== :data:`DEX_ITEM_TYPE_PAD_ID`) to mark positions that should *not*
    contribute to ``L_mlm``.
    """

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]
    item_type_labels: List[List[int]]


# ---------------------------------------------------------------------------
# Corpus builder — benign-only invariant
# ---------------------------------------------------------------------------


def build_mlm_example(
    dex_bytes: bytes,
    *,
    tokenizer: ByteTokenizer,
    max_seq_length: int,
) -> MLMExample:
    """Turn one raw DEX buffer into a padded :class:`MLMExample`.

    Raises
    ------
    DexParseError:
        Propagated from :func:`parse_dex_item_spans`.  The buffer is not
        a benign DEX and must be dropped.
    ValueError:
        If the encoded example would be empty (no body bytes survive the
        budget) — this catches degenerate inputs before they pollute the
        corpus.
    """

    spans = parse_dex_item_spans(dex_bytes)  # raises DexParseError on packed
    encoding = tokenizer.encode_with_mask(
        bytes(dex_bytes),
        max_length=max_seq_length,
        add_special_tokens=True,
    )
    body_len = encoding.length - 2  # minus BOS/EOS (both always present here)
    if body_len <= 0:
        raise ValueError(
            "MLM example has zero body bytes after tokenisation; check "
            "max_seq_length vs buffer size"
        )

    body_labels = region_item_type_labels(
        spans, region_offset=0, region_length=body_len
    )

    # item_type_labels layout: [PAD for BOS] + body_labels + [PAD for EOS] +
    # [PAD for padding tokens]
    item_type_labels: List[int] = [DEX_ITEM_TYPE_PAD_ID] * max_seq_length
    item_type_labels[0] = DEX_ITEM_TYPE_PAD_ID  # BOS (redundant, explicit)
    for i, lbl in enumerate(body_labels):
        item_type_labels[1 + i] = lbl
    # Position 1 + body_len is EOS; already PAD.  Everything after is PAD too.

    return MLMExample(
        input_ids=list(encoding.input_ids),
        attention_mask=list(encoding.attention_mask),
        item_type_labels=item_type_labels,
        source_length=body_len,
    )


class MLMCorpusBuilder:
    """Stream DEX buffers -> :class:`MLMExample` with benign-only filtering.

    Usage::

        builder = MLMCorpusBuilder(config)
        for dex_bytes in iter_benign_dex():
            builder.add(dex_bytes)
        examples, stats = builder.finalise()

    :meth:`finalise` enforces the exclusion-ratio cap: it raises
    :class:`BenignCorpusError` when more than
    :attr:`PretrainMLMConfig.benign_exclusion_max_ratio` of buffers
    failed to parse, on the assumption that a real benign corpus would
    not trip the DEX parser.
    """

    def __init__(
        self,
        config: PretrainMLMConfig,
        *,
        tokenizer: Optional[ByteTokenizer] = None,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer or ByteTokenizer(
            max_length=config.max_seq_length
        )
        self._examples: List[MLMExample] = []
        self._total = 0
        self._dropped_parse = 0
        self._dropped_empty = 0

    @property
    def config(self) -> PretrainMLMConfig:
        return self._config

    def add(self, dex_bytes: bytes) -> bool:
        """Try to ingest ``dex_bytes``. Returns True iff kept."""

        self._total += 1
        try:
            example = build_mlm_example(
                dex_bytes,
                tokenizer=self._tokenizer,
                max_seq_length=self._config.max_seq_length,
            )
        except DexParseError:
            self._dropped_parse += 1
            return False
        except ValueError:
            self._dropped_empty += 1
            return False
        self._examples.append(example)
        return True

    def extend(self, buffers: Iterable[bytes]) -> None:
        for buf in buffers:
            self.add(buf)

    def finalise(self) -> Tuple[List[MLMExample], MLMCorpusStats]:
        stats = MLMCorpusStats(
            total_seen=self._total,
            kept=len(self._examples),
            dropped_parse_error=self._dropped_parse,
            dropped_empty=self._dropped_empty,
        )
        if stats.total_seen == 0:
            raise BenignCorpusError(
                "no DEX buffers were provided; refusing to build empty corpus"
            )
        if stats.kept == 0:
            raise BenignCorpusError(
                "benign-only corpus is empty: every buffer failed "
                f"parse_dex_item_spans ({stats.dropped_parse_error} parse "
                f"errors, {stats.dropped_empty} empty). Likely wired up to "
                "packed data by mistake — see ours_method_spec §8."
            )
        if stats.exclusion_ratio > self._config.benign_exclusion_max_ratio:
            raise BenignCorpusError(
                f"benign-only corpus exclusion ratio "
                f"{stats.exclusion_ratio:.1%} exceeds cap "
                f"{self._config.benign_exclusion_max_ratio:.1%}. "
                f"This usually means packed / obfuscated DEX is leaking "
                f"into the pre-training pool, which violates "
                "sellpoint-2 benign-only invariance "
                "(ours_method_spec §8 / research_framing §4.2). "
                f"Stats: {stats}"
            )
        return list(self._examples), stats


def build_mlm_corpus(
    buffers: Iterable[bytes],
    config: PretrainMLMConfig,
    *,
    tokenizer: Optional[ByteTokenizer] = None,
) -> Tuple[List[MLMExample], MLMCorpusStats]:
    """One-shot helper: build + finalise in a single call."""

    builder = MLMCorpusBuilder(config, tokenizer=tokenizer)
    builder.extend(buffers)
    return builder.finalise()


# ---------------------------------------------------------------------------
# MLM collator — 80/10/10 masking + aux labels
# ---------------------------------------------------------------------------


class MLMCollator:
    """Token-level masking collator, pure stdlib.

    The collator is *stateful* in its RNG only (seeded from
    :attr:`PretrainMLMConfig.seed`); all masking decisions are recorded
    per-call in the returned :class:`MLMBatch` so downstream trainers
    can inspect them.

    Contract
    --------
    * ``input_ids[i][j] == PAD_ID`` or ``j == 0 / j == EOS-pos`` ⇒
      position is never masked and its ``labels[i][j] == -100``.
    * Other positions are selected with probability
      ``mlm_mask_prob``.  Of those: 80 % ``MASK``, 10 % random valid byte,
      10 % unchanged-but-supervised.
    * ``item_type_labels`` is passed through *unchanged* from the
      example; we rely on the examples to carry ``DEX_ITEM_TYPE_PAD_ID``
      at BOS / EOS / PAD positions so the aux CE's ``ignore_index``
      handles them automatically.
    """

    def __init__(
        self,
        config: PretrainMLMConfig,
        *,
        tokenizer: Optional[ByteTokenizer] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer or ByteTokenizer()
        self._rng = rng or random.Random(config.seed)

    def collate(self, examples: Sequence[MLMExample]) -> MLMBatch:
        if not examples:
            raise ValueError("collate requires at least one example")

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        item_type_batch: List[List[int]] = []

        tok = self._tokenizer
        cfg = self._config

        for ex in examples:
            src = list(ex.input_ids)
            mask = list(ex.attention_mask)
            labels = [DEX_ITEM_TYPE_PAD_ID] * len(src)

            for j, tok_id in enumerate(src):
                if mask[j] == 0:
                    continue  # PAD
                if tok_id in tok.SPECIAL_IDS:
                    continue  # BOS / EOS / (MASK: never in input) / UNK
                if self._rng.random() >= cfg.mlm_mask_prob:
                    continue  # not selected for MLM

                labels[j] = tok_id  # original id supervises L_mlm
                r = self._rng.random()
                if r < cfg.mlm_replace_mask_prob:
                    src[j] = tok.MASK_ID
                elif r < cfg.mlm_replace_mask_prob + cfg.mlm_replace_random_prob:
                    src[j] = tok.BYTE_OFFSET + self._rng.randrange(256)
                # else: keep original token (still supervised)

            input_ids_batch.append(src)
            attention_mask_batch.append(mask)
            labels_batch.append(labels)
            item_type_batch.append(list(ex.item_type_labels))

        return MLMBatch(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            labels=labels_batch,
            item_type_labels=item_type_batch,
        )


# ---------------------------------------------------------------------------
# Lazy-torch loss assembly
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_pretrain_loss requires torch. Install via "
            "``pip install -e \".[dl]\"`` (see AGENTS.md §2)."
        ) from exc
    return torch, F


def compute_pretrain_loss(
    mlm_logits: Any,
    item_type_logits: Any,
    mlm_labels: Any,
    item_type_labels: Any,
    *,
    item_type_aux_weight: float,
) -> Any:
    """Combined byte-MLM + grammar-aware aux loss (F-MIL-c core).

    ``L = L_mlm + λ_item · L_item_type``

    * ``mlm_logits``: ``[B, L, vocab]``
    * ``item_type_logits``: ``[B, L, n_item_types]``
    * ``mlm_labels``: ``[B, L]``, ``-100`` on ignored positions.
    * ``item_type_labels``: ``[B, L]``, ``-100`` on ignored positions.

    Setting ``item_type_aux_weight == 0`` makes the aux branch
    *non-participating* in the autograd graph, which downstream ablation
    runs rely on (§12.5 row "no grammar aux").
    """

    _, F = _require_torch()

    if item_type_aux_weight < 0.0:
        raise ValueError(
            f"item_type_aux_weight must be >= 0, got {item_type_aux_weight}"
        )

    vocab_size = mlm_logits.shape[-1]
    mlm_loss = F.cross_entropy(
        mlm_logits.reshape(-1, vocab_size),
        mlm_labels.reshape(-1),
        ignore_index=DEX_ITEM_TYPE_PAD_ID,
    )

    if item_type_aux_weight == 0.0:
        # Skip aux branch entirely — keeps it out of autograd for ablation.
        return mlm_loss

    n_item_types = item_type_logits.shape[-1]
    item_loss = F.cross_entropy(
        item_type_logits.reshape(-1, n_item_types),
        item_type_labels.reshape(-1),
        ignore_index=DEX_ITEM_TYPE_PAD_ID,
    )
    return mlm_loss + item_type_aux_weight * item_loss
