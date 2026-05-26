"""Stage 3: Differential fine-tuning trainer (improved_packed_apk_framework.md §8).

Training loss:
    L = L_bag + λ₁·L_rank_diff + λ₂·L_attention_align + λ₃·L_normality_reg

Where:
    L_bag: BCE on bag_logit (APK packed/benign)
    L_rank_diff: margin ranking loss (diff-positive entries > diff-negative)
    L_attention_align: KL(attention || softmax(diff_targets / τ))
    L_normality_reg: keep clean APK entry normality high

Contract:
- Lazy torch (no import at module scope)
- Uses FullFrameworkModel from entry_aggregator.py
- Input: list of APK bags (features + diff labels)
- Output: trained model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DifferentialTrainerConfig",
    "APKBag",
    "train_differential",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DifferentialTrainerConfig:
    """Configuration for Stage 3 differential training."""

    # Loss weights
    lambda_rank: float = 0.3
    lambda_align: float = 0.3
    lambda_normality: float = 0.1

    # Alignment
    align_temperature: float = 0.5
    rank_margin: float = 0.2

    # Training
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_regions_per_bag: int = 512  # subsample large APKs

    # Device
    device: str = "auto"

    # Verbosity
    verbose: bool = True


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class APKBag:
    """A single APK represented as a bag of regions for training.

    Fields:
        scalar_features: [N_regions, 318] numpy array
        entry_type_ids: [N_regions] int array
        section_type_ids: [N_regions] int array
        entry_boundaries: list of (start, end) tuples grouping regions by entry
        apk_label: 0 (benign) or 1 (packed)
        diff_targets: [N_entries] float array (per-entry diff scores, 0-1)
                      None for APKs without paired data
        apk_id: identifier string
    """

    scalar_features: np.ndarray
    entry_type_ids: np.ndarray
    section_type_ids: np.ndarray
    entry_boundaries: List[Tuple[int, int]]
    apk_label: int
    diff_targets: Optional[np.ndarray] = None  # [N_entries] or None
    apk_id: str = ""


# ---------------------------------------------------------------------------
# Lazy torch
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn, optim
    except ImportError as exc:
        raise ImportError("Stage 3 trainer requires torch.") from exc
    return torch, F, nn, optim


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _bag_loss(bag_logit, label, torch_module):
    """BCE loss on bag prediction."""
    torch = torch_module
    target = torch.tensor(float(label), device=bag_logit.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(bag_logit, target)


def _rank_diff_loss(entry_logits, diff_targets, margin, torch_module):
    """Margin ranking loss: diff-positive entries should score higher.

    For all pairs (positive_entry, negative_entry):
        loss += max(0, margin - logit_pos + logit_neg)
    """
    torch = torch_module
    if diff_targets is None or len(entry_logits) < 2:
        return torch.tensor(0.0, device=entry_logits.device)

    pos_mask = diff_targets > 0.5  # diff-positive entries
    neg_mask = diff_targets < 0.2  # unchanged entries

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=entry_logits.device)

    pos_logits = entry_logits[pos_mask]
    neg_logits = entry_logits[neg_mask]

    # All-pairs ranking loss (efficient: mean of pos vs mean of neg)
    pos_mean = pos_logits.mean()
    neg_mean = neg_logits.mean()
    loss = torch.clamp(margin - pos_mean + neg_mean, min=0.0)
    return loss


def _attention_align_loss(attention, diff_targets, temperature, torch_module):
    """KL divergence between attention and diff-derived soft targets.

    attention: [N_entries] model attention (sums to 1)
    diff_targets: [N_entries] diff scores (0-1)
    """
    torch = torch_module
    F = torch.nn.functional

    if diff_targets is None or len(attention) < 2:
        return torch.tensor(0.0, device=attention.device)

    # Convert diff targets to a probability distribution
    target_logits = diff_targets / temperature
    target_dist = F.softmax(target_logits, dim=0)

    # KL(target || attention) — use log_softmax of attention
    # Avoid log(0) by clamping attention
    log_attn = torch.log(attention.clamp(min=1e-8))
    kl = F.kl_div(log_attn, target_dist, reduction="batchmean")
    return kl


def _normality_reg_loss(entry_normality, apk_label, torch_module):
    """Normality regularization: benign APKs should have high normality.

    For benign APKs: push normality toward 1.0
    For packed APKs: no constraint (let the model learn)
    """
    torch = torch_module
    if apk_label == 0:
        # Benign: all entries should be "normal"
        return (1.0 - entry_normality).mean()
    return torch.tensor(0.0, device=entry_normality.device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str):
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _subsample_bag(bag: APKBag, max_regions: int, rng: np.random.RandomState) -> APKBag:
    """Subsample regions if bag is too large."""
    n_regions = bag.scalar_features.shape[0]
    if n_regions <= max_regions:
        return bag

    # Keep all entries but subsample regions within large entries
    # Simple approach: random sample of regions, recompute boundaries
    indices = rng.choice(n_regions, size=max_regions, replace=False)
    indices.sort()

    # Recompute entry boundaries
    new_boundaries = []
    idx_set = set(indices)
    new_idx_map = {old: new for new, old in enumerate(indices)}

    for start, end in bag.entry_boundaries:
        entry_indices = [new_idx_map[i] for i in range(start, end) if i in idx_set]
        if entry_indices:
            new_boundaries.append((entry_indices[0], entry_indices[-1] + 1))

    return APKBag(
        scalar_features=bag.scalar_features[indices],
        entry_type_ids=bag.entry_type_ids[indices],
        section_type_ids=bag.section_type_ids[indices],
        entry_boundaries=new_boundaries,
        apk_label=bag.apk_label,
        diff_targets=bag.diff_targets,
        apk_id=bag.apk_id,
    )


def train_differential(
    bags: List[APKBag],
    config: Optional[DifferentialTrainerConfig] = None,
    model=None,  # Optional pre-built model; if None, builds from scratch
) -> Any:
    """Train the full framework model with Stage 3 differential losses.

    Args:
        bags: List of APKBag instances (packed + benign, with optional diff_targets)
        config: Training configuration
        model: Optional pre-built FullFrameworkModel. If None, builds new one.

    Returns:
        Trained nn.Module (FullFrameworkModel)
    """
    cfg = config or DifferentialTrainerConfig()
    torch, F, nn, optim = _require_torch()

    device = _resolve_device(cfg.device)

    # Build model if not provided
    if model is None:
        from android_packer.models.entry_aggregator import (
            FullFrameworkConfig,
            build_full_framework_model,
        )
        model_cfg = FullFrameworkConfig()
        model = build_full_framework_model(model_cfg)

    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    rng = np.random.RandomState(42)
    n_bags = len(bags)

    if cfg.verbose:
        n_with_diff = sum(1 for b in bags if b.diff_targets is not None)
        n_packed = sum(1 for b in bags if b.apk_label == 1)
        print(f"[Stage 3] Training: {n_bags} bags ({n_packed} packed, "
              f"{n_bags - n_packed} benign), {n_with_diff} with diff targets", flush=True)
        print(f"[Stage 3] Config: epochs={cfg.epochs}, lr={cfg.learning_rate}, "
              f"λ_rank={cfg.lambda_rank}, λ_align={cfg.lambda_align}, "
              f"λ_norm={cfg.lambda_normality}", flush=True)

    for epoch in range(cfg.epochs):
        epoch_losses = {"total": 0.0, "bag": 0.0, "rank": 0.0, "align": 0.0, "norm": 0.0}
        order = rng.permutation(n_bags)

        for batch_start in range(0, n_bags, cfg.batch_size):
            batch_indices = order[batch_start:batch_start + cfg.batch_size]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for bag_idx in batch_indices:
                bag = bags[bag_idx]
                bag = _subsample_bag(bag, cfg.max_regions_per_bag, rng)

                if len(bag.entry_boundaries) == 0:
                    continue

                # Convert to tensors
                scalars = torch.tensor(
                    bag.scalar_features, dtype=torch.float32, device=device
                )
                et_ids = torch.tensor(
                    bag.entry_type_ids, dtype=torch.long, device=device
                )
                st_ids = torch.tensor(
                    bag.section_type_ids, dtype=torch.long, device=device
                )

                # Forward pass
                (bag_logit, entry_attention, entry_logits,
                 region_susp, region_norm, region_attn) = model(
                    scalars, et_ids, st_ids, bag.entry_boundaries
                )

                # Compute losses
                l_bag = _bag_loss(bag_logit, bag.apk_label, torch)

                # Diff-based losses (only for bags with diff targets)
                diff_t = None
                if bag.diff_targets is not None:
                    # Ensure diff_targets matches n_entries from boundaries
                    n_entries = len(bag.entry_boundaries)
                    dt = bag.diff_targets
                    if len(dt) > n_entries:
                        dt = dt[:n_entries]
                    elif len(dt) < n_entries:
                        dt = np.pad(dt, (0, n_entries - len(dt)))
                    diff_t = torch.tensor(
                        dt, dtype=torch.float32, device=device
                    )

                l_rank = _rank_diff_loss(
                    entry_logits, diff_t, cfg.rank_margin, torch
                )
                l_align = _attention_align_loss(
                    entry_attention, diff_t, cfg.align_temperature, torch
                )

                # Normality from regions → entries (mean per entry)
                entry_norms = []
                for start, end in bag.entry_boundaries:
                    if start < end:
                        entry_norms.append(region_norm[start:end].mean())
                    else:
                        entry_norms.append(torch.tensor(0.5, device=device))
                entry_normality = torch.stack(entry_norms)

                l_norm = _normality_reg_loss(entry_normality, bag.apk_label, torch)

                # Total loss for this bag
                bag_total = (l_bag
                             + cfg.lambda_rank * l_rank
                             + cfg.lambda_align * l_align
                             + cfg.lambda_normality * l_norm)

                batch_loss = batch_loss + bag_total / len(batch_indices)

                # Track
                epoch_losses["bag"] += l_bag.item()
                epoch_losses["rank"] += l_rank.item()
                epoch_losses["align"] += l_align.item()
                epoch_losses["norm"] += l_norm.item()
                epoch_losses["total"] += bag_total.item()

            # Backward + step
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Epoch summary
        if cfg.verbose:
            n = max(n_bags, 1)
            print(
                f"  Epoch {epoch+1:3d}/{cfg.epochs}: "
                f"total={epoch_losses['total']/n:.4f} "
                f"bag={epoch_losses['bag']/n:.4f} "
                f"rank={epoch_losses['rank']/n:.4f} "
                f"align={epoch_losses['align']/n:.4f} "
                f"norm={epoch_losses['norm']/n:.4f}",
                flush=True,
            )

    model.eval()
    return model
