"""Ours = Typed-Instance MIL model (Stage A main method).

This module wires together the three pieces that define the **new**
"Ours" method after the 2026-05-06 novelty uplift (see
``docs/research_framing.md`` §3.2 / §4.2 and
``docs/method/ours_method_spec.md`` §12):

1. :mod:`android_packer.models.typed_encoder.build_typed_encoder`
   — per-instance typed projection (6 types: ``encrypted_dex /
   extracted_method_body / metadata_table / compressed_payload / shim /
   native_stub``).  Zero-cost labels from the injected-packer adapter.
2. :mod:`android_packer.models.mil_head.build_mil_pooling` — one of
   ``topk`` / ``noisy_or`` / ``attention`` (gated ABMIL).  Provides
   bag-level logit **and** instance-level localisation scores.
3. Optional :mod:`android_packer.models.item_type_head.build_item_type_head`
   — grammar-aware auxiliary prediction head used only during byte
   pre-training; kept here for config-level symmetry (trainer builds it
   explicitly).

Public API:
    >>> from android_packer.models.ours import OursConfig, build_ours
    >>> cfg = OursConfig(typed=TypedEncoderConfig(input_dim=15))
    >>> model = build_ours(cfg)            # requires torch (lazy)
    >>> bag_logit, attn, instance_logits = model(features, types)

Relationship to :mod:`android_packer.models.payload_hunter_lite`
----------------------------------------------------------------
``PayloadHunter-Lite`` remains as a first-class **ablation baseline** —
it captures the "just a learned region scorer + attention aggregator"
design point that the MIL formulation is supposed to beat.  It is *not*
deleted.  See the ablation matrix in
``docs/method/ours_method_spec.md`` §12.5.

Lazy-torch contract: ``torch`` is only imported at instantiation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

from android_packer.models.mil_head import (
    AttentionPoolingConfig,
    MILPoolingKind,
    NoisyOrPoolingConfig,
    TopKPoolingConfig,
    build_mil_pooling,
)
from android_packer.models.typed_encoder import (
    N_TYPED_INSTANCE_TYPES,
    TypedEncoderConfig,
    build_typed_encoder,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn


__all__ = [
    "OursConfig",
    "build_ours",
]


@dataclass(frozen=True)
class OursConfig:
    """Top-level config for the Ours (Typed-Instance MIL) model.

    Parameters
    ----------
    typed:
        :class:`TypedEncoderConfig` controlling the per-instance
        typed projection.
    mil_pooling:
        One of ``"topk"`` / ``"noisy_or"`` / ``"attention"``.  Default is
        ``"attention"`` (gated ABMIL) because it gives the cleanest
        instance-level explanation layer, which is the core user-facing
        value proposition of the paper.
    topk / noisy_or / attention:
        Sub-configs for the selected pooling.  Unused sub-configs are
        ignored.  If ``attention.feature_dim`` is None we auto-fill it
        from ``typed.head_hidden_dim`` so the attention sees the typed
        hidden state (typical setup), not the raw input features.
    use_feature_attention:
        If True (default), the attention pooling is fed the typed hidden
        states ``[N, H]``.  If False, attention scores on instance
        logits alone (useful ablation: "attention-on-logits vs
        attention-on-features").
    """

    typed: TypedEncoderConfig = field(default_factory=TypedEncoderConfig)
    mil_pooling: MILPoolingKind = "attention"
    topk: TopKPoolingConfig = field(default_factory=TopKPoolingConfig)
    noisy_or: NoisyOrPoolingConfig = field(default_factory=NoisyOrPoolingConfig)
    attention: AttentionPoolingConfig = field(default_factory=AttentionPoolingConfig)
    use_feature_attention: bool = True


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Ours (Typed-Instance MIL) requires torch. Install via "
            "``pip install -e \".[dl]\"`` (see AGENTS.md §2)."
        ) from exc
    return torch, nn


def build_ours(config: Optional[OursConfig] = None) -> "nn.Module":
    """Assemble the Ours (Typed-Instance MIL) model from its config.

    Returns an ``nn.Module`` whose ``forward`` signature is::

        forward(instance_features: Tensor[N, D],
                instance_types:    Tensor[N]  int64)
          -> (bag_logit:        Tensor[scalar],
              attention:        Tensor[N],
              instance_logits:  Tensor[N])

    ``instance_types`` values must lie in
    ``[0, N_TYPED_INSTANCE_TYPES)`` — see
    :func:`android_packer.models.typed_encoder.instance_type_id`.
    """

    cfg = config or OursConfig()
    if cfg.typed.n_types != N_TYPED_INSTANCE_TYPES:
        # Parametric override kept only for tests; warn loudly at build.
        if cfg.typed.n_types <= 0:
            raise ValueError(
                f"typed.n_types must be positive, got {cfg.typed.n_types}"
            )

    _, nn = _require_torch()

    typed_encoder = build_typed_encoder(cfg.typed)

    # Resolve attention feature_dim so ABMIL sees the typed hidden state
    # when we feed ``hidden`` in.  We only override None (user-explicit
    # values are respected).
    attention_cfg = cfg.attention
    if (
        cfg.mil_pooling == "attention"
        and cfg.use_feature_attention
        and attention_cfg.feature_dim is None
    ):
        attention_cfg = AttentionPoolingConfig(
            feature_dim=cfg.typed.head_hidden_dim,
            attn_hidden_dim=attention_cfg.attn_hidden_dim,
            dropout=attention_cfg.dropout,
            attention_on_logit=attention_cfg.attention_on_logit,
        )

    pooling = build_mil_pooling(
        cfg.mil_pooling,
        topk=cfg.topk,
        noisy_or=cfg.noisy_or,
        attention=attention_cfg,
    )

    class _OursModule(nn.Module):
        """Thin wrapper that glues typed encoder + MIL pooling."""

        def __init__(
            self,
            typed_module: "nn.Module",
            pooling_module: "nn.Module",
            use_feature_attention: bool,
            pooling_kind: MILPoolingKind,
        ):
            super().__init__()
            self.typed_encoder = typed_module
            self.pooling = pooling_module
            self.use_feature_attention = use_feature_attention
            self.pooling_kind = pooling_kind

        def forward(
            self,
            instance_features: "torch.Tensor",
            instance_types: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            hidden, instance_logits = self.typed_encoder(
                instance_features, instance_types
            )
            if self.pooling_kind == "attention" and self.use_feature_attention:
                bag_logit, attention = self.pooling(
                    instance_logits, instance_features=hidden
                )
            else:
                bag_logit, attention = self.pooling(instance_logits)
            return bag_logit, attention, instance_logits

    return _OursModule(
        typed_encoder,
        pooling,
        cfg.use_feature_attention,
        cfg.mil_pooling,
    )
