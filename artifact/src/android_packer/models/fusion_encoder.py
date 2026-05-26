"""Three-path fusion encoder: Dalvik BERT + Native BERT + Byte BERT + stat features.

From pseudo_code_bert_packed_apk_framework.md §10.2:
    h_region = Fusion(h_dalvik, h_native, h_byte, stat_features)

Uses a shared BERT with 3 forward passes (one per token type).
Fuses the CLS embeddings with statistical features via MLP.

Integration with existing framework:
- Replaces full_encoder.py's MLP as the region encoder
- Output shape [N, output_dim] plugs directly into entry_aggregator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    import torch
    from torch import nn

from android_packer.decoders.pseudo_tokenizer import (
    BYTE_REPRESENTATION_LEGACY_RAW,
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_DALVIK,
    TOKEN_TYPE_NATIVE,
    vocab_size_for_byte_representation,
)
from android_packer.features.full_feature_extractor import SCALAR_FEATURE_DIM
from android_packer.models.pseudo_code_bert import PseudoCodeBERTConfig
from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES

_DEX_ENTRY_ID = ENTRY_COARSE_TYPES.index("dex")
_ELF_ENTRY_ID = ENTRY_COARSE_TYPES.index("elf")
_ARCHIVE_ENTRY_ID = ENTRY_COARSE_TYPES.index("archive")
_ASSET_ENTRY_ID = ENTRY_COARSE_TYPES.index("asset")
_ARSC_ENTRY_ID = ENTRY_COARSE_TYPES.index("arsc")
_MANIFEST_ENTRY_ID = ENTRY_COARSE_TYPES.index("manifest")

__all__ = [
    "FusionEncoderConfig",
    "build_fusion_encoder",
]


@dataclass(frozen=True)
class FusionEncoderConfig:
    """Config for the three-path fusion encoder."""

    # BERT config
    bert_hidden_dim: int = 256
    bert_n_layers: int = 4
    bert_n_heads: int = 8
    bert_intermediate_dim: int = 512
    bert_max_length: int = 256  # shorter for memory efficiency
    bert_dropout: float = 0.1

    # Statistical features
    stat_dim: int = SCALAR_FEATURE_DIM  # 318

    # Fusion
    fusion_hidden_dim: int = 512
    output_dim: int = 256
    fusion_dropout: float = 0.1

    # Gated fusion (adaptive BERT vs stat weighting)
    use_gated_fusion: bool = False  # False = legacy concat, True = learned gating
    gate_hidden_dim: int = 128
    use_bert_features: bool = True
    use_stat_features: bool = True
    path_dropout_prob: float = 0.0
    use_region_type_routing: bool = False
    batch_bert_streams: bool = True
    routing_dex_byte_weight: float = 0.25
    routing_elf_byte_weight: float = 0.25
    routing_byte_entry_weight: float = 1.0
    routing_unknown_weight: float = 0.25
    byte_representation: str = BYTE_REPRESENTATION_LEGACY_RAW

    # Output heads (same as full_encoder.py)
    use_suspicion_head: bool = True
    use_normality_head: bool = True
    use_pretrain_normality_head: bool = True


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("FusionEncoder requires torch.") from exc
    return torch, nn


def build_fusion_encoder(
    config: Optional[FusionEncoderConfig] = None,
) -> "nn.Module":
    """Build the three-path fusion encoder.

    Forward signature:
        forward(dalvik_ids, dalvik_types, dalvik_mask,
                native_ids, native_types, native_mask,
                byte_ids, byte_types, byte_mask,
                stat_features)
        → (region_embeddings [N, output_dim],
           suspicion [N],
           normality [N])
    """
    cfg = config or FusionEncoderConfig()
    torch, nn = _require_torch()

    from android_packer.models.pseudo_code_bert import build_pseudo_code_bert

    class _FusionEncoder(nn.Module):
        def __init__(self, cfg: FusionEncoderConfig):
            super().__init__()
            self.cfg = cfg

            # Shared BERT (one model, three token types)
            bert_cfg = PseudoCodeBERTConfig(
                vocab_size=vocab_size_for_byte_representation(cfg.byte_representation),
                hidden_dim=cfg.bert_hidden_dim,
                n_layers=cfg.bert_n_layers,
                n_heads=cfg.bert_n_heads,
                intermediate_dim=cfg.bert_intermediate_dim,
                max_length=cfg.bert_max_length,
                dropout=cfg.bert_dropout,
                n_token_types=3,
            )
            self.bert = build_pseudo_code_bert(bert_cfg)

            if not cfg.use_bert_features and not cfg.use_stat_features:
                raise ValueError("FusionEncoder requires at least one feature branch.")

            if not cfg.use_bert_features:
                # Stat-only ablation branch.
                self.stat_proj = nn.Sequential(
                    nn.Linear(cfg.stat_dim, cfg.fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                    nn.Linear(cfg.fusion_hidden_dim, cfg.output_dim),
                    nn.GELU(),
                )
            elif not cfg.use_stat_features:
                # BERT-only path aggregation for path ablations.
                self.bert_aggregation = nn.Sequential(
                    nn.Linear(cfg.bert_hidden_dim * 3, cfg.fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                    nn.Linear(cfg.fusion_hidden_dim, cfg.output_dim),
                    nn.GELU(),
                )
            elif cfg.use_gated_fusion:
                # --- Gated Fusion: ABMIL-style adaptive weighting ---
                # Project BERT 3×256 → 256 and stat 318 → 256 (same dim for fair gating)
                self.bert_aggregation = nn.Sequential(
                    nn.Linear(cfg.bert_hidden_dim * 3, cfg.output_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                )
                self.stat_proj = nn.Sequential(
                    nn.Linear(cfg.stat_dim, cfg.output_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                )

                # Gating network (ABMIL pattern: tanh * sigmoid)
                gate_input_dim = cfg.output_dim * 2  # [h_bert, h_stat] concat
                self.gate_tanh = nn.Sequential(
                    nn.Linear(gate_input_dim, cfg.gate_hidden_dim),
                    nn.Tanh(),
                )
                self.gate_sigmoid = nn.Sequential(
                    nn.Linear(gate_input_dim, cfg.gate_hidden_dim),
                    nn.Sigmoid(),
                )
                self.gate_logits = nn.Linear(cfg.gate_hidden_dim, 2)

                # Final projection after gated fusion
                self.fusion_head = nn.Sequential(
                    nn.Linear(cfg.output_dim, cfg.output_dim),
                    nn.GELU(),
                )
            else:
                # --- Legacy: simple concatenation ---
                self.stat_proj = nn.Sequential(
                    nn.Linear(cfg.stat_dim, 128),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                )
                fusion_input_dim = cfg.bert_hidden_dim * 3 + 128  # 896
                self.fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, cfg.fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.fusion_dropout),
                    nn.Linear(cfg.fusion_hidden_dim, cfg.output_dim),
                    nn.GELU(),
                )

            # Output heads
            if cfg.use_suspicion_head:
                self.suspicion_head = nn.Linear(cfg.output_dim, 1)
            else:
                self.suspicion_head = None

            if cfg.use_normality_head:
                self.normality_head = nn.Linear(cfg.output_dim, 1)
            else:
                self.normality_head = None

            if cfg.use_pretrain_normality_head:
                self.pretrain_normality_head = nn.Linear(cfg.bert_hidden_dim, 1)
            else:
                self.pretrain_normality_head = None

        def forward(
            self,
            dalvik_ids: "torch.Tensor",     # [N, L]
            dalvik_types: "torch.Tensor",   # [N, L]
            dalvik_mask: "torch.Tensor",    # [N, L]
            native_ids: "torch.Tensor",     # [N, L]
            native_types: "torch.Tensor",   # [N, L]
            native_mask: "torch.Tensor",    # [N, L]
            byte_ids: "torch.Tensor",       # [N, L]
            byte_types: "torch.Tensor",     # [N, L]
            byte_mask: "torch.Tensor",      # [N, L]
            stat_features: "torch.Tensor",  # [N, stat_dim]
            active_paths: Optional[Tuple[str, ...]] = None,
            entry_type_ids: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                embeddings: [N, output_dim]
                suspicion: [N] (0-1)
                normality: [N] (0-1)
            """
            if active_paths is None:
                active = {"dalvik", "arm64", "byte"}
            else:
                active = set(active_paths)
                unknown = active - {"dalvik", "arm64", "byte"}
                if unknown:
                    raise ValueError(f"Unknown active path(s): {sorted(unknown)}")

            if self.training and self.cfg.path_dropout_prob > 0 and len(active) > 1:
                kept = {
                    path for path in active
                    if torch.rand((), device=stat_features.device).item()
                    >= self.cfg.path_dropout_prob
                }
                if kept:
                    active = kept

            if not self.cfg.use_bert_features:
                embeddings = self.stat_proj(stat_features)
            else:
                if self.cfg.batch_bert_streams:
                    # Batch enabled pseudo-code streams into one BERT call.
                    # This is mathematically equivalent to separate shared-BERT
                    # forwards, but keeps modern GPUs fed with a larger batch.
                    path_inputs = []
                    if "dalvik" in active:
                        path_inputs.append(("dalvik", dalvik_ids, dalvik_types, dalvik_mask))
                    if "arm64" in active:
                        path_inputs.append(("arm64", native_ids, native_types, native_mask))
                    if "byte" in active:
                        path_inputs.append(("byte", byte_ids, byte_types, byte_mask))

                    if not path_inputs:
                        raise ValueError("active_paths must enable at least one BERT path")
                    batched_ids = torch.cat([items[1] for items in path_inputs], dim=0)
                    batched_types = torch.cat([items[2] for items in path_inputs], dim=0)
                    batched_mask = torch.cat([items[3] for items in path_inputs], dim=0)
                    batched_h = self.bert(batched_ids, batched_types, batched_mask)
                    chunks = batched_h.split(dalvik_ids.shape[0], dim=0)
                    stream_outputs = {
                        name: chunk for (name, *_), chunk in zip(path_inputs, chunks)
                    }
                    template = batched_h[:dalvik_ids.shape[0]]
                    h_dalvik = stream_outputs.get("dalvik")
                    h_native = stream_outputs.get("arm64")
                    h_byte = stream_outputs.get("byte")
                else:
                    h_dalvik = h_native = h_byte = None
                    template = None
                    if "dalvik" in active:
                        h_dalvik = self.bert(dalvik_ids, dalvik_types, dalvik_mask)
                        template = h_dalvik
                    if "arm64" in active:
                        h_native = self.bert(native_ids, native_types, native_mask)
                        template = h_native
                    if "byte" in active:
                        h_byte = self.bert(byte_ids, byte_types, byte_mask)
                        template = h_byte
                    if template is None:
                        raise ValueError("active_paths must enable at least one BERT path")
                if h_dalvik is None:
                    h_dalvik = torch.zeros_like(template)
                if h_native is None:
                    h_native = torch.zeros_like(template)
                if h_byte is None:
                    h_byte = torch.zeros_like(template)

                if self.cfg.use_region_type_routing:
                    if entry_type_ids is None:
                        raise ValueError("entry_type_ids are required when region-type routing is enabled")
                    weights = torch.ones(
                        (entry_type_ids.shape[0], 3),
                        dtype=template.dtype,
                        device=template.device,
                    ) * self.cfg.routing_unknown_weight
                    dex_mask = entry_type_ids == _DEX_ENTRY_ID
                    elf_mask = entry_type_ids == _ELF_ENTRY_ID
                    byte_mask_entries = (
                        (entry_type_ids == _ARCHIVE_ENTRY_ID)
                        | (entry_type_ids == _ASSET_ENTRY_ID)
                        | (entry_type_ids == _ARSC_ENTRY_ID)
                        | (entry_type_ids == _MANIFEST_ENTRY_ID)
                    )
                    weights[dex_mask] = torch.tensor(
                        [1.0, 0.0, self.cfg.routing_dex_byte_weight],
                        dtype=template.dtype,
                        device=template.device,
                    )
                    weights[elf_mask] = torch.tensor(
                        [0.0, 1.0, self.cfg.routing_elf_byte_weight],
                        dtype=template.dtype,
                        device=template.device,
                    )
                    weights[byte_mask_entries] = torch.tensor(
                        [0.0, 0.0, self.cfg.routing_byte_entry_weight],
                        dtype=template.dtype,
                        device=template.device,
                    )

                    h_dalvik = h_dalvik * weights[:, 0:1]
                    h_native = h_native * weights[:, 1:2]
                    h_byte = h_byte * weights[:, 2:3]

                if not self.cfg.use_stat_features:
                    # --- BERT-only path aggregation ---
                    embeddings = self.bert_aggregation(
                        torch.cat([h_dalvik, h_native, h_byte], dim=-1)
                    )
                elif self.cfg.use_gated_fusion:
                    # --- Gated Fusion ---
                    # Aggregate BERT streams: [N, 768] → [N, 256]
                    h_bert = self.bert_aggregation(
                        torch.cat([h_dalvik, h_native, h_byte], dim=-1)
                    )
                    # Project stat features: [N, 318] → [N, 256]
                    h_stat = self.stat_proj(stat_features)

                    # Compute gate weights via ABMIL gating
                    gate_input = torch.cat([h_bert, h_stat], dim=-1)  # [N, 512]
                    gated = self.gate_tanh(gate_input) * self.gate_sigmoid(gate_input)  # [N, gate_hidden]
                    gate_logits = self.gate_logits(gated)  # [N, 2]
                    gates = torch.softmax(gate_logits, dim=-1)  # [N, 2]

                    # Weighted fusion: gates[:,0] for BERT, gates[:,1] for stat
                    h_fused = gates[:, 0:1] * h_bert + gates[:, 1:2] * h_stat  # [N, 256]
                    embeddings = self.fusion_head(h_fused)  # [N, 256]

                    # Store gate values for interpretability (detached)
                    self._last_gate_bert = gates[:, 0].detach().mean().item()
                    self._last_gate_stat = gates[:, 1].detach().mean().item()
                else:
                    # --- Legacy concat fusion ---
                    h_stat = self.stat_proj(stat_features)  # [N, 128]
                    h_fused = torch.cat([h_dalvik, h_native, h_byte, h_stat], dim=-1)  # [N, 896]
                    embeddings = self.fusion(h_fused)  # [N, output_dim]

            # Output heads
            if self.suspicion_head is not None:
                suspicion = torch.sigmoid(self.suspicion_head(embeddings)).squeeze(-1)
            else:
                suspicion = torch.zeros(embeddings.shape[0], device=embeddings.device)

            if self.normality_head is not None:
                normality = torch.sigmoid(self.normality_head(embeddings)).squeeze(-1)
            else:
                normality = torch.ones(embeddings.shape[0], device=embeddings.device) * 0.5

            return embeddings, suspicion, normality

        def forward_single_stream(
            self,
            token_ids: "torch.Tensor",
            token_type_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Forward pass for a single stream (used during pretraining).

            Returns: cls_embedding [B, hidden_dim]
            """
            return self.bert(token_ids, token_type_ids, attention_mask)

        def forward_with_mlm(
            self,
            token_ids: "torch.Tensor",
            token_type_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass returning CLS + per-token hidden (for MLM loss).

            Returns: (cls [B, hidden], hidden_states [B, L, hidden])
            """
            return self.bert.forward_with_hidden(token_ids, token_type_ids, attention_mask)

        def forward_pretrain_normality(
            self,
            cls_embedding: "torch.Tensor",
        ) -> "torch.Tensor":
            """Return normality logits from the single-stream BERT CLS state."""
            if self.pretrain_normality_head is None:
                raise ValueError("pretrain normality head is disabled")
            return self.pretrain_normality_head(cls_embedding).squeeze(-1)

    return _FusionEncoder(cfg)
