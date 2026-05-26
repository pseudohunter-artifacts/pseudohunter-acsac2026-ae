"""Structured Pseudo-code Masked Language Modeling (spMLM) pretraining.

Stage 1 from pseudo_code_bert_packed_apk_framework.md §11/§15:
Trains the shared PseudoCodeBERT on benign APK regions to learn:
- Normal Dalvik instruction patterns
- Normal native instruction patterns
- Normal byte distributions

Uses structured masking (§11.3):
- Don't simultaneously mask opcode + operand in same instruction
- Higher masking probability for abnormal tokens
- Dynamic masking (re-mask each epoch)

Loss: L_spMLM = L_mlm_reconstruction + lambda_type * L_token_type_prediction
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

from android_packer.decoders.pseudo_tokenizer import (
    UNIFIED_VOCAB_SIZE,
    TOKEN_TYPE_DALVIK,
    TOKEN_TYPE_NATIVE,
    TOKEN_TYPE_BYTE,
)

__all__ = [
    "SpMLMConfig",
    "SpMLMBatch",
    "create_spmlm_batch",
    "corrupt_token_sequence",
    "compute_spmlm_loss",
    "train_spmlm",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Special token IDs (shared across all streams)
_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_MASK_ID = 3
_UNK_ID = 4


@dataclass
class SpMLMConfig:
    """Configuration for spMLM pretraining."""

    # Masking
    mask_prob: float = 0.15          # fraction of tokens to mask
    mask_token_prob: float = 0.80    # of masked: replace with [MASK]
    random_token_prob: float = 0.10  # of masked: replace with random
    # remaining 10%: keep unchanged

    # Structured masking
    never_mask_special: bool = True   # never mask PAD/BOS/EOS
    boost_abnormal_mask: float = 2.0  # abnormal tokens masked 2x more often

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # MLM head
    vocab_size: int = UNIFIED_VOCAB_SIZE

    # B4 minimal V3 auxiliary objective. Corruptions are generic typed-token
    # perturbations, not DPT- or task-case-specific features.
    normality_loss_weight: float = 0.0
    corruption_prob: float = 0.35

    # Device
    device: str = "auto"
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_prob < 1.0:
            raise ValueError(f"mask_prob must be in (0, 1), got {self.mask_prob}")
        if not 0.0 <= self.mask_token_prob <= 1.0:
            raise ValueError(
                f"mask_token_prob must be in [0, 1], got {self.mask_token_prob}"
            )
        if not 0.0 <= self.random_token_prob <= 1.0:
            raise ValueError(
                f"random_token_prob must be in [0, 1], got {self.random_token_prob}"
            )
        if self.mask_token_prob + self.random_token_prob > 1.0:
            raise ValueError("mask_token_prob + random_token_prob must be <= 1")
        if self.normality_loss_weight < 0.0:
            raise ValueError(
                "normality_loss_weight must be non-negative, "
                f"got {self.normality_loss_weight}"
            )
        if not 0.0 <= self.corruption_prob <= 1.0:
            raise ValueError(
                f"corruption_prob must be in [0, 1], got {self.corruption_prob}"
            )
        if self.vocab_size <= _UNK_ID + 1:
            raise ValueError(f"vocab_size is too small: {self.vocab_size}")


# ---------------------------------------------------------------------------
# Batch creation with structured masking
# ---------------------------------------------------------------------------


@dataclass
class SpMLMBatch:
    """A batch for spMLM training."""
    input_ids: np.ndarray        # [B, L] — masked input
    clean_input_ids: np.ndarray
    token_type_ids: np.ndarray   # [B, L]
    attention_mask: np.ndarray   # [B, L]
    labels: np.ndarray           # [B, L] — original tokens at masked positions, -100 elsewhere


def _real_token_positions(ids: List[int], attention_mask: List[int]) -> List[int]:
    return [
        idx
        for idx, (token_id, is_real) in enumerate(zip(ids, attention_mask))
        if is_real and token_id not in (_PAD_ID, _BOS_ID, _EOS_ID)
    ]


def _random_token_like(original: int, config: SpMLMConfig, rng: random.Random) -> int:
    replacement = rng.randint(_UNK_ID + 1, config.vocab_size - 1)
    if replacement == original:
        replacement = _UNK_ID + 1 + (
            (replacement - _UNK_ID) % (config.vocab_size - _UNK_ID - 1)
        )
    return replacement


def corrupt_token_sequence(
    token_ids: List[int],
    attention_mask: List[int],
    config: SpMLMConfig,
    rng: random.Random,
) -> List[int]:
    """Apply generic typed-token corruption for B4 normality pretraining."""
    corrupted = list(token_ids)
    positions = _real_token_positions(corrupted, attention_mask)
    if not positions:
        return corrupted

    operation = rng.choice(
        (
            "shuffle",
            "chunk_reorder",
            "token_remap",
            "fake_magic",
            "reference_remap",
            "encrypted_insert",
        )
    )

    if operation == "shuffle":
        values = [corrupted[pos] for pos in positions]
        rng.shuffle(values)
        for pos, value in zip(positions, values):
            corrupted[pos] = value
        return corrupted

    if operation == "chunk_reorder" and len(positions) >= 4:
        chunk_count = min(4, max(2, len(positions) // 4))
        chunk_size = max(1, len(positions) // chunk_count)
        chunks = [
            positions[start:start + chunk_size]
            for start in range(0, len(positions), chunk_size)
        ]
        values_by_chunk = [[corrupted[pos] for pos in chunk] for chunk in chunks]
        order = list(range(len(chunks)))
        rng.shuffle(order)
        reordered = [value for idx in order for value in values_by_chunk[idx]]
        for pos, value in zip(positions, reordered):
            corrupted[pos] = value
        return corrupted

    if operation == "fake_magic":
        for pos in positions[:min(3, len(positions))]:
            corrupted[pos] = _random_token_like(corrupted[pos], config, rng)
        return corrupted

    if operation == "reference_remap":
        n_change = max(1, len(positions) // 5)
        for pos in rng.sample(positions, min(n_change, len(positions))):
            corrupted[pos] = _random_token_like(corrupted[pos], config, rng)
        return corrupted

    if operation == "encrypted_insert":
        n_change = max(1, len(positions) // 3)
        high_start = max(_UNK_ID + 1, config.vocab_size - 16)
        for pos in rng.sample(positions, min(n_change, len(positions))):
            corrupted[pos] = rng.randint(high_start, config.vocab_size - 1)
        return corrupted

    n_change = max(1, len(positions) // 4)
    for pos in rng.sample(positions, min(n_change, len(positions))):
        old = corrupted[pos]
        corrupted[pos] = _UNK_ID + 1 + (
            (old + rng.randint(1, 17)) % (config.vocab_size - _UNK_ID - 1)
        )
    return corrupted


def create_spmlm_batch(
    sequences: List[Dict[str, List[int]]],
    config: SpMLMConfig,
    rng: random.Random,
) -> SpMLMBatch:
    """Create a masked batch from pre-encoded sequences.

    Each sequence dict has: token_ids, token_type_ids, attention_mask, abnormal_mask
    """
    B = len(sequences)
    L = len(sequences[0]["token_ids"])

    input_ids = np.zeros((B, L), dtype=np.int64)
    clean_input_ids = np.zeros((B, L), dtype=np.int64)
    token_type_ids = np.zeros((B, L), dtype=np.int64)
    attention_mask = np.zeros((B, L), dtype=np.int64)
    labels = np.full((B, L), -100, dtype=np.int64)

    for i, seq in enumerate(sequences):
        ids = list(seq["token_ids"])
        types = seq["token_type_ids"]
        mask = seq["attention_mask"]
        abnormal = seq.get("abnormal_mask", [0] * L)

        token_type_ids[i] = types
        attention_mask[i] = mask
        clean_input_ids[i] = ids

        # Structured masking
        for j in range(L):
            if mask[j] == 0:
                continue  # padding
            if config.never_mask_special and ids[j] in (_PAD_ID, _BOS_ID, _EOS_ID):
                continue

            # Masking probability (boosted for abnormal tokens)
            prob = config.mask_prob
            if abnormal[j]:
                prob *= config.boost_abnormal_mask
            prob = min(prob, 0.5)

            if rng.random() < prob:
                labels[i, j] = ids[j]  # record original for loss

                r = rng.random()
                if r < config.mask_token_prob:
                    ids[j] = _MASK_ID
                elif r < config.mask_token_prob + config.random_token_prob:
                    ids[j] = rng.randint(5, config.vocab_size - 1)
                # else: keep unchanged (model still must predict)

        input_ids[i] = ids

    return SpMLMBatch(
        input_ids=input_ids,
        clean_input_ids=clean_input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_spmlm_loss(
    model: "nn.Module",
    batch: SpMLMBatch,
    device: "torch.device",
    config: Optional[SpMLMConfig] = None,
    rng: Optional[random.Random] = None,
) -> "torch.Tensor":
    """Compute spMLM loss for a batch.

    Uses the model's forward_with_mlm to get per-token hidden states,
    then projects to vocab for cross-entropy at masked positions.
    """
    import torch
    import torch.nn.functional as F

    cfg = config or SpMLMConfig()
    input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device=device)
    token_type_ids = torch.tensor(batch.token_type_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(batch.attention_mask, dtype=torch.float32, device=device)
    labels = torch.tensor(batch.labels, dtype=torch.long, device=device)

    # Forward pass — get per-token hidden states
    _, hidden_states = model.forward_with_mlm(input_ids, token_type_ids, attention_mask)
    # hidden_states: [B, L, hidden_dim]

    # MLM prediction head (simple linear projection to vocab)
    # Note: the MLM head should be part of the model or passed separately
    # For now, use the token embedding weights (tied weights)
    vocab_logits = F.linear(hidden_states, model.bert.token_embed.weight)  # [B, L, vocab_size]

    # Compute loss only at masked positions (labels != -100)
    loss = F.cross_entropy(
        vocab_logits.view(-1, vocab_logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
    )

    if cfg.normality_loss_weight > 0.0 and cfg.corruption_prob > 0.0:
        if not hasattr(model, "forward_pretrain_normality"):
            raise ValueError("normality pretraining requires forward_pretrain_normality")
        rng = rng or random.Random(1234)
        n_clean = batch.clean_input_ids.shape[0]
        n_corrupt = max(1, int(round(n_clean * cfg.corruption_prob)))
        selected_indices = sorted(rng.sample(range(n_clean), min(n_corrupt, n_clean)))
        selected = set(selected_indices)
        corrupted_rows = [
            corrupt_token_sequence(row.tolist(), mask.tolist(), cfg, rng)
            for idx, (row, mask) in enumerate(
                zip(batch.clean_input_ids, batch.attention_mask)
            )
            if idx in selected
        ]
        corrupted = np.array(
            corrupted_rows,
            dtype=np.int64,
        )
        clean = batch.clean_input_ids
        clean_types = batch.token_type_ids
        clean_mask = batch.attention_mask
        corrupted_types = batch.token_type_ids[selected_indices]
        corrupted_mask = batch.attention_mask[selected_indices]
        normality_ids = np.concatenate([clean, corrupted], axis=0)
        normality_types = np.concatenate([clean_types, corrupted_types], axis=0)
        normality_mask = np.concatenate([clean_mask, corrupted_mask], axis=0)
        normality_targets = np.concatenate(
            [
                np.ones(clean.shape[0], dtype=np.float32),
                np.zeros(corrupted.shape[0], dtype=np.float32),
            ],
            axis=0,
        )

        norm_ids = torch.tensor(normality_ids, dtype=torch.long, device=device)
        norm_types = torch.tensor(normality_types, dtype=torch.long, device=device)
        norm_mask = torch.tensor(normality_mask, dtype=torch.float32, device=device)
        norm_targets = torch.tensor(normality_targets, dtype=torch.float32, device=device)

        cls, _ = model.forward_with_mlm(norm_ids, norm_types, norm_mask)
        normality_logits = model.forward_pretrain_normality(cls)
        normality_loss = F.binary_cross_entropy_with_logits(
            normality_logits,
            norm_targets,
        )
        loss = loss + cfg.normality_loss_weight * normality_loss

    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_spmlm(
    model: "nn.Module",
    corpus: List[Dict[str, List[int]]],
    config: Optional[SpMLMConfig] = None,
) -> "nn.Module":
    """Train pseudo-code BERT with spMLM objective.

    Args:
        model: FusionEncoder (uses model.bert for spMLM)
        corpus: List of pre-encoded sequences (token_ids, token_type_ids, attention_mask, abnormal_mask)
        config: Training configuration

    Returns:
        Trained model
    """
    import torch

    cfg = config or SpMLMConfig()
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else cfg.device if cfg.device != "auto" else "cpu"
    )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    rng = random.Random(42)
    n_examples = len(corpus)
    n_batches = max(1, n_examples // cfg.batch_size)
    total_steps = cfg.epochs * n_batches

    print(f"[spMLM] Training: {n_examples} examples, {cfg.epochs} epochs, "
          f"{n_batches} batches/epoch, {total_steps} total steps", flush=True)
    print(f"[spMLM] Device: {device}, FP16: {cfg.use_fp16}", flush=True)

    # Optional: FP16 scaler
    scaler = None
    if cfg.use_fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        indices = list(range(n_examples))
        rng.shuffle(indices)

        for batch_idx in range(n_batches):
            batch_start = batch_idx * cfg.batch_size
            batch_indices = indices[batch_start:batch_start + cfg.batch_size]
            if not batch_indices:
                continue

            batch_seqs = [corpus[i] for i in batch_indices]
            batch = create_spmlm_batch(batch_seqs, cfg, rng)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = compute_spmlm_loss(model, batch, device, cfg, rng)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_spmlm_loss(model, batch, device, cfg, rng)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{cfg.epochs}: loss={avg_loss:.4f}", flush=True)

    model.eval()
    return model
