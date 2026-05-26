"""Pseudo-code BERT: shared Transformer encoder for three-path decoding.

Architecture from pseudo_code_bert_packed_apk_framework.md §10:
- Single BERT with token_type_embedding to distinguish Dalvik/Native/Byte streams
- 4 layers, 256 hidden dim, 8 heads (fits RTX 5060 8GB easily)
- Input: token_ids [B, L] + token_type_ids [B, L] + attention_mask [B, L]
- Output: CLS embedding [B, hidden_dim] for region-level representation

Lazy-torch contract: no torch at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    import torch
    from torch import nn

from android_packer.decoders.pseudo_tokenizer import UNIFIED_VOCAB_SIZE

__all__ = [
    "PseudoCodeBERTConfig",
    "build_pseudo_code_bert",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudoCodeBERTConfig:
    """Configuration for the shared pseudo-code BERT encoder."""

    vocab_size: int = UNIFIED_VOCAB_SIZE  # 358
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    intermediate_dim: int = 512  # FFN intermediate
    max_length: int = 512
    dropout: float = 0.1
    n_token_types: int = 3  # Dalvik=0, Native=1, Byte=2

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.hidden_dim  # token embed
        embed += self.max_length * self.hidden_dim  # position embed
        embed += self.n_token_types * self.hidden_dim  # type embed
        # Per layer: 4 * hidden^2 (attn) + 2 * hidden * intermediate (FFN) + norms
        per_layer = (4 * self.hidden_dim ** 2 +
                     2 * self.hidden_dim * self.intermediate_dim +
                     4 * self.hidden_dim)
        return embed + self.n_layers * per_layer


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("PseudoCodeBERT requires torch.") from exc
    return torch, nn


def build_pseudo_code_bert(
    config: Optional[PseudoCodeBERTConfig] = None,
) -> "nn.Module":
    """Build the shared pseudo-code BERT encoder.

    Returns nn.Module with forward signature:
        forward(token_ids, token_type_ids, attention_mask)
        → cls_embedding [B, hidden_dim]
    """
    cfg = config or PseudoCodeBERTConfig()
    torch, nn = _require_torch()
    import math

    class _MultiHeadAttention(nn.Module):
        def __init__(self, hidden_dim, n_heads, dropout):
            super().__init__()
            assert hidden_dim % n_heads == 0
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            self.scale = math.sqrt(self.head_dim)
            self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, d]
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, L, L]
            if mask is not None:
                # mask: [B, L] → [B, 1, 1, L]
                # Use -65000 instead of -1e9 to stay within fp16 range
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -65000.0)
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, L, D)
            return self.out_proj(out)

    class _TransformerBlock(nn.Module):
        def __init__(self, hidden_dim, n_heads, intermediate_dim, dropout):
            super().__init__()
            self.attn = _MultiHeadAttention(hidden_dim, n_heads, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x, mask=None):
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
            return x

    class _PseudoCodeBERT(nn.Module):
        def __init__(self, cfg: PseudoCodeBERTConfig):
            super().__init__()
            self.cfg = cfg

            # Embeddings
            self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.position_embed = nn.Embedding(cfg.max_length, cfg.hidden_dim)
            self.token_type_embed = nn.Embedding(cfg.n_token_types, cfg.hidden_dim)
            self.embed_norm = nn.LayerNorm(cfg.hidden_dim)
            self.embed_dropout = nn.Dropout(cfg.dropout)

            # Transformer layers
            self.layers = nn.ModuleList([
                _TransformerBlock(
                    cfg.hidden_dim, cfg.n_heads,
                    cfg.intermediate_dim, cfg.dropout
                )
                for _ in range(cfg.n_layers)
            ])

            # CLS pooling head
            self.cls_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.Tanh(),
            )

            # Initialize weights
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        def forward(
            self,
            token_ids: "torch.Tensor",       # [B, L] long
            token_type_ids: "torch.Tensor",   # [B, L] long
            attention_mask: "torch.Tensor",   # [B, L] float/long
        ) -> "torch.Tensor":
            """
            Returns:
                cls_embedding: [B, hidden_dim] — CLS token representation
            """
            B, L = token_ids.shape

            # Position IDs
            position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)

            # Embedding sum
            x = (self.token_embed(token_ids)
                 + self.position_embed(position_ids)
                 + self.token_type_embed(token_type_ids))
            x = self.embed_norm(x)
            x = self.embed_dropout(x)

            # Transformer layers
            for layer in self.layers:
                x = layer(x, attention_mask)

            # CLS pooling (first token = BOS position)
            cls_output = self.cls_head(x[:, 0, :])  # [B, hidden_dim]
            return cls_output

        def forward_with_hidden(
            self,
            token_ids: "torch.Tensor",
            token_type_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass returning both CLS and per-token hidden states.

            Returns:
                cls_embedding: [B, hidden_dim]
                hidden_states: [B, L, hidden_dim] (for MLM prediction)
            """
            B, L = token_ids.shape
            position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)

            x = (self.token_embed(token_ids)
                 + self.position_embed(position_ids)
                 + self.token_type_embed(token_type_ids))
            x = self.embed_norm(x)
            x = self.embed_dropout(x)

            for layer in self.layers:
                x = layer(x, attention_mask)

            cls_output = self.cls_head(x[:, 0, :])
            return cls_output, x

    return _PseudoCodeBERT(cfg)
