"""DEX item-type auxiliary prediction head (grammar-aware aux loss).

Implements the **auxiliary supervision signal** of
``docs/research_framing.md`` §4.2 sellpoint 2 and ``docs/method/ours_method_spec.md``
§5.1 (batch **F-MIL-c**).  At pre-training time each DEX byte token
carries a per-token label drawn from
:data:`android_packer.features.dex_item_parser.DEX_ITEM_TYPES`
(header / string_ids / type_ids / proto_ids / field_ids / method_ids /
class_defs / code_item / string_data / other).  This head reads the
per-token hidden state of the byte encoder and predicts the item type;
the resulting cross-entropy is summed into the pre-training loss as

    L = L_mlm + λ_item · L_item_type

where ``ignore_index`` drops PAD tokens.

The signal is **zero-cost** — the parser is pure stdlib and only ever
runs on benign DEX (packed DEX would fail to parse, which is the
desired filter; see research_framing §5.2) — so the auxiliary label is
*not* a synthetic-only artefact.  This is a major robustness argument
for reviewers.

Torch is lazy-imported (§3.1 spec contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

from android_packer.features.dex_item_parser import DEX_ITEM_TYPES

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn


__all__ = [
    "DEX_ITEM_TYPE_PAD_ID",
    "ItemTypeHeadConfig",
    "build_item_type_head",
]


#: PAD id for per-token item-type labels.  Must differ from every valid
#: item-type integer id (valid range is ``0..len(DEX_ITEM_TYPES)-1``).
#: Downstream loss code must pass ``ignore_index=DEX_ITEM_TYPE_PAD_ID``.
DEX_ITEM_TYPE_PAD_ID: int = -100


@dataclass(frozen=True)
class ItemTypeHeadConfig:
    """Hyper-parameters for :func:`build_item_type_head`.

    The head is intentionally tiny: the main capacity lives in the byte
    encoder, and this head only needs to decode a ≈ 10-way per-token
    classification from the hidden state.
    """

    hidden_size: int = 256
    n_item_types: int = len(DEX_ITEM_TYPES)
    dropout: float = 0.1


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ItemTypeHead requires torch. Install via "
            "``pip install -e \".[dl]\"``."
        ) from exc
    return torch, nn


def build_item_type_head(
    config: Optional[ItemTypeHeadConfig] = None,
) -> "nn.Module":
    """Build the per-token DEX item-type classification head.

    forward(hidden_states: Tensor[B, L, H]) -> logits[B, L, n_item_types]
    """

    cfg = config or ItemTypeHeadConfig()
    if cfg.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {cfg.hidden_size}")
    if cfg.n_item_types <= 1:
        raise ValueError(f"n_item_types must be > 1, got {cfg.n_item_types}")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")

    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(cfg.hidden_size, cfg.hidden_size),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden_size, cfg.n_item_types),
    )
