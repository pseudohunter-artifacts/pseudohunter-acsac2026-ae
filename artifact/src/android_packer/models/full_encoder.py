"""Full framework region encoder with type-specific experts (Module D.1-D.3).

Architecture from improved_packed_apk_framework.md §5:

    scalar_features [N, 318]
    entry_type_ids  [N] (int)
    section_type_ids [N] (int)
        ↓
    Entry Type Embedding [N, embed_dim]
    Section Type Embedding [N, embed_dim]
        ↓ concat with scalars
    Shared Trunk: Linear→GELU→Dropout→Linear→GELU→Dropout → [N, hidden_dim]
        ↓
    Type Expert Routing: 4 experts (dex/elf/asset/unknown) → [N, hidden_dim]
        ↓
    Output Heads:
      - suspicion_head → [N, 1] (sigmoid → instance suspicion score)
      - normality_head → [N, 1] (sigmoid → instance normality score)
      - instance_logit_head → [N, 1] (raw logit for MIL)

Design contract:
- Pure lazy-torch (no torch at module scope)
- Configs are frozen dataclasses (no torch dependency)
- Forward returns (embeddings, suspicion, normality, logits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    import torch
    from torch import nn

from android_packer.features.full_feature_extractor import SCALAR_FEATURE_DIM
from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES, SECTION_TYPES

__all__ = [
    "FullEncoderConfig",
    "build_full_encoder",
]


# ---------------------------------------------------------------------------
# Config (pure stdlib, no torch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullEncoderConfig:
    """Configuration for the full framework region encoder."""

    # Feature dimensions
    scalar_dim: int = SCALAR_FEATURE_DIM  # 318

    # Embedding dimensions
    entry_type_embed_dim: int = 32
    section_type_embed_dim: int = 32
    n_entry_types: int = len(ENTRY_COARSE_TYPES)   # 7
    n_section_types: int = len(SECTION_TYPES)       # 20

    # Shared trunk
    hidden_dim: int = 256
    n_trunk_layers: int = 2
    dropout: float = 0.1

    # Type experts
    n_experts: int = 4  # dex, elf, asset, unknown
    expert_dim: int = 256

    @property
    def trunk_input_dim(self) -> int:
        return self.scalar_dim + self.entry_type_embed_dim + self.section_type_embed_dim


# ---------------------------------------------------------------------------
# Lazy torch helpers
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "Full encoder requires torch. Install via pip install -e \".[dl]\""
        ) from exc
    return torch, nn


# ---------------------------------------------------------------------------
# Expert routing: maps entry_type_id to expert index
# ---------------------------------------------------------------------------

# entry_type_id: 0=dex, 1=elf, 2=manifest, 3=arsc, 4=archive, 5=asset, 6=unknown
# expert_id:     0=dex, 1=elf, 2=asset (covers manifest/arsc/archive/asset), 3=unknown
_ENTRY_TYPE_TO_EXPERT = [0, 1, 2, 2, 2, 2, 3]  # len=7


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_full_encoder(
    config: Optional[FullEncoderConfig] = None,
) -> "nn.Module":
    """Build the full framework region encoder.

    Returns an nn.Module with forward signature:
        forward(scalar_features, entry_type_ids, section_type_ids)
        → (embeddings [N, expert_dim],
           suspicion [N],
           normality [N],
           logits [N])
    """
    cfg = config or FullEncoderConfig()
    torch, nn = _require_torch()

    class _FullEncoder(nn.Module):
        def __init__(self, cfg: FullEncoderConfig):
            super().__init__()
            self.cfg = cfg

            # Embedding layers for categorical features
            self.entry_type_embed = nn.Embedding(
                cfg.n_entry_types, cfg.entry_type_embed_dim
            )
            self.section_type_embed = nn.Embedding(
                cfg.n_section_types, cfg.section_type_embed_dim
            )

            # Shared trunk
            trunk_layers = []
            in_dim = cfg.trunk_input_dim
            for i in range(cfg.n_trunk_layers):
                out_dim = cfg.hidden_dim
                trunk_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ])
                in_dim = out_dim
            self.shared_trunk = nn.Sequential(*trunk_layers)

            # Type-specific experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.expert_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                )
                for _ in range(cfg.n_experts)
            ])

            # Output heads
            self.suspicion_head = nn.Linear(cfg.expert_dim, 1)
            self.normality_head = nn.Linear(cfg.expert_dim, 1)
            self.logit_head = nn.Linear(cfg.expert_dim, 1)

            # Register expert routing as buffer
            self.register_buffer(
                "_expert_map",
                torch.tensor(_ENTRY_TYPE_TO_EXPERT, dtype=torch.long),
            )

        def forward(
            self,
            scalar_features: "torch.Tensor",   # [N, scalar_dim]
            entry_type_ids: "torch.Tensor",     # [N] long
            section_type_ids: "torch.Tensor",   # [N] long
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                embeddings: [N, expert_dim] — region embeddings
                suspicion: [N] — instance suspicion scores (0-1)
                normality: [N] — instance normality scores (0-1)
                logits: [N] — raw instance logits for MIL
            """
            N = scalar_features.shape[0]

            # Embed categoricals
            et_emb = self.entry_type_embed(entry_type_ids)    # [N, 32]
            st_emb = self.section_type_embed(section_type_ids)  # [N, 32]

            # Concatenate all inputs
            x = torch.cat([scalar_features, et_emb, st_emb], dim=-1)  # [N, trunk_input_dim]

            # Shared trunk
            h_shared = self.shared_trunk(x)  # [N, hidden_dim]

            # Expert routing via scatter
            expert_ids = self._expert_map[entry_type_ids]  # [N]
            h_expert = torch.zeros(N, self.cfg.expert_dim, device=h_shared.device)

            for expert_idx in range(self.cfg.n_experts):
                mask = (expert_ids == expert_idx)
                if mask.any():
                    h_expert[mask] = self.experts[expert_idx](h_shared[mask])

            # Output heads
            suspicion = torch.sigmoid(self.suspicion_head(h_expert)).squeeze(-1)  # [N]
            normality = torch.sigmoid(self.normality_head(h_expert)).squeeze(-1)  # [N]
            logits = self.logit_head(h_expert).squeeze(-1)  # [N]

            return h_expert, suspicion, normality, logits

    return _FullEncoder(cfg)
