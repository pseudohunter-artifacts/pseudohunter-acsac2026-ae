"""Byte-level CNN region scorer for APK payload localization.

This module intentionally follows the lazy-torch import contract used by
``payload_hunter_lite``: importing :mod:`android_packer.models` must not
require torch, while calling the factory does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - type-only import
    import torch
    from torch import nn


__all__ = [
    "ByteCnnRegionScorerConfig",
    "build_byte_cnn_region_scorer",
]


@dataclass(frozen=True)
class ByteCnnRegionScorerConfig:
    """Hyper-parameters for the byte-CNN region scorer.

    The input is a padded byte sequence with values in ``[0, 255]`` and a
    dedicated padding token. Default ``max_length=4096`` matches the v4
    Track-A region window so the model sees one complete region without
    changing the shared regioning protocol.
    """

    max_length: int = 4096
    embedding_dim: int = 32
    conv_channels: int = 64
    kernel_sizes: Tuple[int, ...] = (3, 5, 7, 15)
    hidden_dim: int = 128
    dropout: float = 0.1
    pad_token_id: int = 256
    activation: str = "gelu"


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised without [dl]
        raise ImportError(
            "Byte-CNN model components require torch. Install optional "
            "dependencies via ``pip install -e .[dl]``."
        ) from exc
    return torch, nn


def _resolve_activation(name: str) -> Any:
    _, nn = _require_torch()
    name = name.lower()
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    raise ValueError(
        f"unsupported activation {name!r}; expected one of 'gelu' / 'relu'"
    )


def build_byte_cnn_region_scorer(
    config: Optional[ByteCnnRegionScorerConfig] = None,
) -> "nn.Module":
    """Build a byte-CNN scorer returning one logit per region.

    Parameters
    ----------
    config:
        Model hyper-parameters. The returned module accepts a ``LongTensor``
        of shape ``[batch, max_length]`` and emits logits of shape ``[batch]``.
    """

    cfg = config or ByteCnnRegionScorerConfig()
    if cfg.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {cfg.max_length}")
    if cfg.embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {cfg.embedding_dim}")
    if cfg.conv_channels <= 0:
        raise ValueError(f"conv_channels must be positive, got {cfg.conv_channels}")
    if cfg.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
    if cfg.pad_token_id < 256:
        raise ValueError(
            f"pad_token_id must be >= 256 so raw byte values remain valid, got {cfg.pad_token_id}"
        )
    if not cfg.kernel_sizes:
        raise ValueError("kernel_sizes must contain at least one kernel")
    for kernel in cfg.kernel_sizes:
        if kernel <= 0:
            raise ValueError(f"kernel sizes must be positive, got {kernel}")
        if kernel % 2 == 0:
            raise ValueError(
                f"kernel sizes must be odd to preserve sequence length, got {kernel}"
            )
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")

    torch, nn = _require_torch()
    activation_cls = _resolve_activation(cfg.activation)

    class _ByteCnnRegionScorer(nn.Module):
        def __init__(self, scorer_cfg: ByteCnnRegionScorerConfig) -> None:
            super().__init__()
            self.cfg = scorer_cfg
            self.embedding = nn.Embedding(
                scorer_cfg.pad_token_id + 1,
                scorer_cfg.embedding_dim,
                padding_idx=scorer_cfg.pad_token_id,
            )
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        scorer_cfg.embedding_dim,
                        scorer_cfg.conv_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                    for kernel in scorer_cfg.kernel_sizes
                ]
            )
            self.activation = activation_cls()
            pooled_dim = scorer_cfg.conv_channels * len(scorer_cfg.kernel_sizes)
            self.head = nn.Sequential(
                nn.Dropout(scorer_cfg.dropout),
                nn.Linear(pooled_dim, scorer_cfg.hidden_dim),
                activation_cls(),
                nn.Dropout(scorer_cfg.dropout),
                nn.Linear(scorer_cfg.hidden_dim, 1),
            )

        def forward(self, token_ids: "torch.Tensor") -> "torch.Tensor":
            if token_ids.dim() != 2:
                raise ValueError(
                    "byte-CNN token_ids must be 2-D [batch, length], got "
                    f"shape {tuple(token_ids.shape)}"
                )
            if token_ids.shape[1] != self.cfg.max_length:
                raise ValueError(
                    "byte-CNN token_ids length must match max_length "
                    f"{self.cfg.max_length}, got {token_ids.shape[1]}"
                )
            token_ids = token_ids.clamp(min=0, max=self.cfg.pad_token_id)
            x = self.embedding(token_ids).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                h = self.activation(conv(x))
                pooled.append(torch.amax(h, dim=-1))
            features = torch.cat(pooled, dim=-1)
            return self.head(features).squeeze(-1)

    return _ByteCnnRegionScorer(cfg)
