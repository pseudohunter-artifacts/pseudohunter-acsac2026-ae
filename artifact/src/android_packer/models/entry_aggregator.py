"""Entry aggregation + APK-level normality-conditioned MIL (Module D.4-D.5).

Architecture from improved_packed_apk_framework.md §6-§7:

§6 Entry Aggregation:
    Regions belonging to the same entry are pooled into a single entry embedding.
    Uses attention pooling + max pooling concatenation.

§7 APK-level Aggregation:
    Entries are aggregated via normality-conditioned MIL:
        α_i = softmax(a_i_mech + β * a_i_norm)
        bag_logit = Σ α_i * z_i

Design contract:
- Lazy-torch (no import at module scope)
- Returns (bag_logit, entry_attention, entry_suspicion, region_suspicion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = [
    "EntryAggregatorConfig",
    "APKMILConfig",
    "build_entry_aggregator",
    "build_apk_mil",
    "FullFrameworkModel",
    "FullFrameworkConfig",
    "build_full_framework_model",
]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryAggregatorConfig:
    """Config for region→entry aggregation."""
    region_dim: int = 256
    entry_dim: int = 256
    attn_hidden: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class APKMILConfig:
    """Config for entry→APK normality-conditioned MIL."""
    entry_dim: int = 256
    attn_hidden: int = 128
    dropout: float = 0.1
    use_normality: bool = True


@dataclass(frozen=True)
class FullFrameworkConfig:
    """Top-level config for the complete model."""
    # Import here to avoid circular at class-definition time
    scalar_dim: int = 318
    entry_type_embed_dim: int = 32
    section_type_embed_dim: int = 32
    hidden_dim: int = 256
    expert_dim: int = 256
    n_trunk_layers: int = 2
    dropout: float = 0.1
    attn_hidden: int = 128
    use_normality: bool = True


# ---------------------------------------------------------------------------
# Lazy torch
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "Full framework model requires torch."
        ) from exc
    return torch, nn


# ---------------------------------------------------------------------------
# Entry Aggregator (§6)
# ---------------------------------------------------------------------------


def build_entry_aggregator(
    config: Optional[EntryAggregatorConfig] = None,
) -> "nn.Module":
    """Build region→entry aggregation module."""
    cfg = config or EntryAggregatorConfig()
    torch, nn = _require_torch()

    class _EntryAggregator(nn.Module):
        def __init__(self, cfg: EntryAggregatorConfig):
            super().__init__()
            self.cfg = cfg

            # Gated attention for region pooling within an entry
            self.tanh_branch = nn.Sequential(
                nn.Linear(cfg.region_dim, cfg.attn_hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout),
            )
            self.gate_branch = nn.Sequential(
                nn.Linear(cfg.region_dim, cfg.attn_hidden),
                nn.Sigmoid(),
            )
            self.attn_weight = nn.Linear(cfg.attn_hidden, 1)

            # Project concat(attn_pool, max_pool) → entry_dim
            self.proj = nn.Linear(cfg.region_dim * 2, cfg.entry_dim)

        def forward(
            self,
            region_embeddings: "torch.Tensor",  # [total_regions, region_dim]
            region_suspicions: "torch.Tensor",  # [total_regions]
            entry_boundaries: List[Tuple[int, int]],  # [(start, end), ...]
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                entry_embeddings: [n_entries, entry_dim]
                entry_suspicions: [n_entries] (max suspicion of regions)
                region_attention: [total_regions] (attention within each entry)
            """
            entry_embs = []
            entry_susps = []
            region_attn_all = torch.zeros(
                region_embeddings.shape[0], device=region_embeddings.device
            )

            for start, end in entry_boundaries:
                if start >= end:
                    # Empty entry — shouldn't happen but handle gracefully
                    entry_embs.append(torch.zeros(
                        self.cfg.entry_dim, device=region_embeddings.device
                    ))
                    entry_susps.append(torch.tensor(
                        0.0, device=region_embeddings.device
                    ))
                    continue

                regions = region_embeddings[start:end]  # [K, region_dim]
                susps = region_suspicions[start:end]     # [K]

                if regions.shape[0] == 1:
                    # Single region entry
                    attn_pooled = regions[0]
                    max_pooled = regions[0]
                    region_attn_all[start] = 1.0
                else:
                    # Gated attention pooling
                    gated = self.tanh_branch(regions) * self.gate_branch(regions)
                    scores = self.attn_weight(gated).squeeze(-1)  # [K]
                    attn = torch.softmax(scores, dim=0)           # [K]
                    attn_pooled = (attn.unsqueeze(-1) * regions).sum(dim=0)
                    max_pooled = regions.max(dim=0).values
                    region_attn_all[start:end] = attn

                # Project to entry dim
                entry_emb = self.proj(torch.cat([attn_pooled, max_pooled]))
                entry_embs.append(entry_emb)
                entry_susps.append(susps.max())

            entry_embeddings = torch.stack(entry_embs)       # [n_entries, entry_dim]
            entry_suspicions = torch.stack(entry_susps)      # [n_entries]
            return entry_embeddings, entry_suspicions, region_attn_all

    return _EntryAggregator(cfg)


# ---------------------------------------------------------------------------
# APK-level MIL (§7 — normality-conditioned)
# ---------------------------------------------------------------------------


def build_apk_mil(
    config: Optional[APKMILConfig] = None,
) -> "nn.Module":
    """Build normality-conditioned APK-level MIL aggregator."""
    cfg = config or APKMILConfig()
    torch, nn = _require_torch()

    class _APKMIL(nn.Module):
        def __init__(self, cfg: APKMILConfig):
            super().__init__()
            self.cfg = cfg

            # Entry-level logit head
            self.entry_logit_head = nn.Linear(cfg.entry_dim, 1)

            # Mechanism-based attention scoring
            self.tanh_branch = nn.Sequential(
                nn.Linear(cfg.entry_dim, cfg.attn_hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout),
            )
            self.gate_branch = nn.Sequential(
                nn.Linear(cfg.entry_dim, cfg.attn_hidden),
                nn.Sigmoid(),
            )
            self.attn_weight = nn.Linear(cfg.attn_hidden, 1)

            # Normality weight (learnable)
            if cfg.use_normality:
                self.beta = nn.Parameter(torch.tensor(0.5))
            else:
                self.beta = None

        def forward(
            self,
            entry_embeddings: "torch.Tensor",    # [N, entry_dim]
            entry_normality: "torch.Tensor",     # [N] (normality scores from encoder)
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                bag_logit: scalar
                attention: [N] (entry-level attention = localization scores)
                entry_logits: [N] (per-entry logits)
            """
            N = entry_embeddings.shape[0]

            if N == 0:
                device = entry_embeddings.device
                return (
                    torch.zeros((), device=device),
                    torch.zeros((0,), device=device),
                    torch.zeros((0,), device=device),
                )

            # Per-entry logits
            entry_logits = self.entry_logit_head(entry_embeddings).squeeze(-1)  # [N]

            # Mechanism-based attention scores
            gated = self.tanh_branch(entry_embeddings) * self.gate_branch(entry_embeddings)
            attn_scores_mech = self.attn_weight(gated).squeeze(-1)  # [N]

            # Normality-conditioned attention
            if self.beta is not None:
                # Low normality = high anomaly → higher attention
                attn_scores_norm = 1.0 - entry_normality  # anomaly score
                attn_scores = attn_scores_mech + self.beta * attn_scores_norm
            else:
                attn_scores = attn_scores_mech

            attention = torch.softmax(attn_scores, dim=0)  # [N]

            # Bag logit
            bag_logit = (attention * entry_logits).sum()

            return bag_logit, attention, entry_logits

    return _APKMIL(cfg)


# ---------------------------------------------------------------------------
# Complete Model (assembles all components)
# ---------------------------------------------------------------------------


def build_full_framework_model(
    config: Optional[FullFrameworkConfig] = None,
) -> "nn.Module":
    """Build the complete full-framework model.

    Assembles: FullEncoder + EntryAggregator + APKMIL
    into a single nn.Module with a unified forward pass.
    """
    cfg = config or FullFrameworkConfig()
    torch, nn = _require_torch()

    from android_packer.models.full_encoder import FullEncoderConfig, build_full_encoder

    class _FullFrameworkModel(nn.Module):
        def __init__(self, cfg: FullFrameworkConfig):
            super().__init__()

            # Region encoder
            encoder_cfg = FullEncoderConfig(
                scalar_dim=cfg.scalar_dim,
                entry_type_embed_dim=cfg.entry_type_embed_dim,
                section_type_embed_dim=cfg.section_type_embed_dim,
                hidden_dim=cfg.hidden_dim,
                n_trunk_layers=cfg.n_trunk_layers,
                dropout=cfg.dropout,
                expert_dim=cfg.expert_dim,
            )
            self.encoder = build_full_encoder(encoder_cfg)

            # Entry aggregator
            agg_cfg = EntryAggregatorConfig(
                region_dim=cfg.expert_dim,
                entry_dim=cfg.expert_dim,
                attn_hidden=cfg.attn_hidden,
                dropout=cfg.dropout,
            )
            self.entry_aggregator = build_entry_aggregator(agg_cfg)

            # APK MIL
            mil_cfg = APKMILConfig(
                entry_dim=cfg.expert_dim,
                attn_hidden=cfg.attn_hidden,
                dropout=cfg.dropout,
                use_normality=cfg.use_normality,
            )
            self.apk_mil = build_apk_mil(mil_cfg)

        def forward(
            self,
            scalar_features: "torch.Tensor",   # [total_regions, scalar_dim]
            entry_type_ids: "torch.Tensor",     # [total_regions] long
            section_type_ids: "torch.Tensor",   # [total_regions] long
            entry_boundaries: List[Tuple[int, int]],  # [(start, end), ...] per entry
        ) -> Tuple[
            "torch.Tensor",  # bag_logit (scalar)
            "torch.Tensor",  # entry_attention [n_entries]
            "torch.Tensor",  # entry_logits [n_entries]
            "torch.Tensor",  # region_suspicion [total_regions]
            "torch.Tensor",  # region_normality [total_regions]
            "torch.Tensor",  # region_attention [total_regions]
        ]:
            """Full forward pass: regions → entries → APK bag prediction.

            Args:
                scalar_features: all regions stacked [total_regions, scalar_dim]
                entry_type_ids: per-region entry type [total_regions]
                section_type_ids: per-region section type [total_regions]
                entry_boundaries: list of (start_idx, end_idx) grouping regions by entry

            Returns tuple of:
                bag_logit: scalar APK-level logit
                entry_attention: [n_entries] attention weights (= localization)
                entry_logits: [n_entries] per-entry logits
                region_suspicion: [total_regions] region-level suspicion scores
                region_normality: [total_regions] region-level normality scores
                region_attention: [total_regions] region attention within entries
            """
            # Step 1: Encode all regions
            embeddings, suspicion, normality, logits = self.encoder(
                scalar_features, entry_type_ids, section_type_ids
            )

            # Step 2: Aggregate regions into entries
            entry_embeddings, entry_suspicions, region_attention = (
                self.entry_aggregator(embeddings, suspicion, entry_boundaries)
            )

            # Step 3: Compute entry-level normality (mean of region normality per entry)
            entry_normality_list = []
            for start, end in entry_boundaries:
                if start < end:
                    entry_normality_list.append(normality[start:end].mean())
                else:
                    entry_normality_list.append(
                        torch.tensor(0.5, device=normality.device)
                    )
            entry_normality = torch.stack(entry_normality_list)

            # Step 4: APK-level MIL
            bag_logit, entry_attention, entry_logits = self.apk_mil(
                entry_embeddings, entry_normality
            )

            return (
                bag_logit,
                entry_attention,
                entry_logits,
                suspicion,
                normality,
                region_attention,
            )

    return _FullFrameworkModel(cfg)
