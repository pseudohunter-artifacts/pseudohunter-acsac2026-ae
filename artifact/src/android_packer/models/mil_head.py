"""Typed-Instance Multiple-Instance Learning (MIL) pooling heads.

This module implements the **bag-level aggregation** side of the new
"Ours" method (see ``docs/method/ours_method_spec.md`` §2.2 / §12, landing
as batch **F-MIL-a**).  It replaces the simpler attention-only aggregator
from PayloadHunter-Lite with three pooling strategies that are standard
in the weakly-supervised MIL literature and let us *localise* the
payload-bearing instance from an APK-level label alone:

* :class:`TopKPoolingConfig` / :func:`build_topk_pooling` —
  mean of the top-k instance logits.  Simple, calibrated, differentiable,
  a strong default baseline (Maron & Lozano-Pérez 1998; Campanella et
  al., Nat. Med. 2019).
* :class:`NoisyOrPoolingConfig` / :func:`build_noisy_or_pooling` —
  probabilistic OR: ``1 - prod(1 - sigmoid(z_i))``.  Interpretable as
  "bag positive iff ≥ 1 instance is positive", matches our generative
  assumption ("packed" ⇔ ≥ 1 hidden-payload instance present).
* :class:`AttentionPoolingConfig` / :func:`build_attention_pooling` —
  gated-attention pooling (Ilse et al., ICML 2018 ``abmil``): learns a
  soft selector ``α_i`` over instances whose normalised weights serve as
  **instance-level localisation scores** — i.e. the explanation layer.

Design invariants (same as ``payload_hunter_lite``):
* Pure stdlib at import time (see §3.1 lazy-torch contract).  ``torch``
  is never imported at module scope; it is pulled in by
  :func:`_require_torch` only at instantiation.
* Configs are ``@dataclass(frozen=True)`` and require **no** torch.
* Public API exports the configs + factory functions; the concrete
  ``nn.Module`` subclasses are assembled lazily inside ``__new__``.

All three pooling heads share the same I/O contract so the trainer can
swap them via a single ``pooling`` field in :class:`OursConfig`:

.. code-block:: text

    input  : instance_logits          [N]         (bag of N instances)
             instance_features=None   [N, D] or None  (optional; attention uses it)
             instance_types=None      [N] int    (optional; reserved for typed gating)
    output : (bag_logit  [scalar tensor],
              attention  [N]  -> instance-level localisation scores)

``attention`` is always returned (even for top-k / noisy-or, in which
case it is a uniform / noisy-or-contribution vector) so downstream code
has a single interface for explanation dumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - type-only import
    import torch
    from torch import nn


__all__ = [
    "TopKPoolingConfig",
    "NoisyOrPoolingConfig",
    "AttentionPoolingConfig",
    "MILPoolingKind",
    "build_topk_pooling",
    "build_noisy_or_pooling",
    "build_attention_pooling",
    "build_mil_pooling",
]


MILPoolingKind = Literal["topk", "noisy_or", "attention"]


# ---------------------------------------------------------------------------
# Configs (pure stdlib, no torch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopKPoolingConfig:
    """Hyper-parameters for :func:`build_topk_pooling`.

    ``k`` is the number of top-scoring instances averaged to form the bag
    logit.  ``k_ratio`` (if > 0) overrides ``k`` with ``max(1, ceil(k_ratio * N))``
    at runtime, which is useful when bag sizes vary wildly across APKs
    (DEX-heavy bags may have N >> 100 instances, asset-only bags may
    have < 10).
    """

    k: int = 4
    k_ratio: float = 0.0  # if > 0, overrides k per-bag


@dataclass(frozen=True)
class NoisyOrPoolingConfig:
    """Hyper-parameters for :func:`build_noisy_or_pooling`.

    The noisy-or pooling computes ``p_bag = 1 - prod(1 - sigmoid(z_i))``.
    We return ``logit(p_bag)`` so BCE-with-logits loss stays numerically
    stable; ``eps`` clamps the product for float32 safety.
    """

    eps: float = 1e-6


@dataclass(frozen=True)
class AttentionPoolingConfig:
    """Hyper-parameters for :func:`build_attention_pooling`.

    Uses the **gated attention** formulation of Ilse et al. 2018: the
    attention score is ``w^T (tanh(V h) * sigmoid(U h))`` over instance
    features ``h``.  If ``feature_dim`` is None we fall back to scoring
    directly on the ``[logit]`` concatenation (degenerates to
    logit-attention, kept for ablations).

    ``attention_on_logit`` controls whether the scalar instance logit is
    concatenated to the instance feature before attention scoring (our
    default, matches Lite's design and improves stability).
    """

    feature_dim: Optional[int] = None
    attn_hidden_dim: int = 128
    dropout: float = 0.1
    attention_on_logit: bool = True


# ---------------------------------------------------------------------------
# Lazy torch helpers
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    """Import torch + nn on demand, raising a helpful error if missing."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - only hit without [dl]
        raise ImportError(
            "MIL pooling heads require torch. Install the optional "
            "dependencies via ``pip install -e \".[dl]\"`` (see "
            "AGENTS.md §2)."
        ) from exc
    return torch, nn


# ---------------------------------------------------------------------------
# Top-k pooling
# ---------------------------------------------------------------------------


def build_topk_pooling(
    config: Optional[TopKPoolingConfig] = None,
) -> "nn.Module":
    """Build a top-k pooling head (see §12.1 of the Ours spec)."""

    cfg = config or TopKPoolingConfig()
    if cfg.k <= 0 and cfg.k_ratio <= 0.0:
        raise ValueError("either k or k_ratio must be > 0")
    if cfg.k_ratio < 0.0 or cfg.k_ratio > 1.0:
        raise ValueError(f"k_ratio must be in [0, 1], got {cfg.k_ratio}")
    return _TopKPoolingImpl(cfg)


class _TopKPoolingImpl:
    """Internal factory; use :func:`build_topk_pooling`."""

    def __new__(cls, config: TopKPoolingConfig):  # type: ignore[override]
        torch, nn = _require_torch()

        class _Module(nn.Module):
            def __init__(self, cfg: TopKPoolingConfig):
                super().__init__()
                self.cfg = cfg

            def forward(
                self,
                instance_logits: "torch.Tensor",
                instance_features: Optional["torch.Tensor"] = None,  # noqa: ARG002
                instance_types: Optional["torch.Tensor"] = None,  # noqa: ARG002
            ) -> Tuple["torch.Tensor", "torch.Tensor"]:
                if instance_logits.dim() != 1:
                    raise ValueError(
                        "instance_logits must be 1-D [N], got "
                        f"{tuple(instance_logits.shape)}"
                    )
                n = instance_logits.shape[0]
                if n == 0:
                    # Degenerate bag: no instances. Return 0 logit so BCE
                    # treats it as ambiguous; empty attention.
                    zero = instance_logits.new_zeros(())
                    return zero, instance_logits.new_zeros((0,))

                if self.cfg.k_ratio > 0.0:
                    k = max(1, int(round(self.cfg.k_ratio * n)))
                else:
                    k = self.cfg.k
                k = min(k, n)

                topk_vals, topk_idx = torch.topk(instance_logits, k=k, dim=0)
                bag_logit = topk_vals.mean()

                # Attention = 1/k on the selected indices, 0 elsewhere.
                attn = instance_logits.new_zeros((n,))
                attn.scatter_(0, topk_idx, instance_logits.new_full((k,), 1.0 / k))
                return bag_logit, attn

        return _Module(config)


# ---------------------------------------------------------------------------
# Noisy-OR pooling
# ---------------------------------------------------------------------------


def build_noisy_or_pooling(
    config: Optional[NoisyOrPoolingConfig] = None,
) -> "nn.Module":
    """Build a noisy-or pooling head (see §12.2 of the Ours spec)."""

    cfg = config or NoisyOrPoolingConfig()
    if cfg.eps <= 0.0 or cfg.eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {cfg.eps}")
    return _NoisyOrPoolingImpl(cfg)


class _NoisyOrPoolingImpl:
    def __new__(cls, config: NoisyOrPoolingConfig):  # type: ignore[override]
        torch, nn = _require_torch()

        class _Module(nn.Module):
            def __init__(self, cfg: NoisyOrPoolingConfig):
                super().__init__()
                self.cfg = cfg

            def forward(
                self,
                instance_logits: "torch.Tensor",
                instance_features: Optional["torch.Tensor"] = None,  # noqa: ARG002
                instance_types: Optional["torch.Tensor"] = None,  # noqa: ARG002
            ) -> Tuple["torch.Tensor", "torch.Tensor"]:
                if instance_logits.dim() != 1:
                    raise ValueError(
                        "instance_logits must be 1-D [N], got "
                        f"{tuple(instance_logits.shape)}"
                    )
                n = instance_logits.shape[0]
                if n == 0:
                    zero = instance_logits.new_zeros(())
                    return zero, instance_logits.new_zeros((0,))

                probs = torch.sigmoid(instance_logits)
                clamped = probs.clamp(min=self.cfg.eps, max=1.0 - self.cfg.eps)
                log_not = torch.log1p(-clamped)
                # p_bag = 1 - prod(1 - p_i); use log-space for stability.
                log_prod_not = log_not.sum()
                p_bag = 1.0 - torch.exp(log_prod_not)
                p_bag = p_bag.clamp(min=self.cfg.eps, max=1.0 - self.cfg.eps)
                bag_logit = torch.log(p_bag) - torch.log1p(-p_bag)

                # Instance-level contribution to the OR: higher ``probs``
                # → higher share of the "at least one positive" mass.
                # Normalise to a proper simplex for the attention slot.
                contrib = probs
                contrib_sum = contrib.sum()
                if float(contrib_sum.detach()) <= 0.0:
                    attn = instance_logits.new_full((n,), 1.0 / n)
                else:
                    attn = contrib / contrib_sum
                return bag_logit, attn

        return _Module(config)


# ---------------------------------------------------------------------------
# Attention pooling (gated ABMIL)
# ---------------------------------------------------------------------------


def build_attention_pooling(
    config: Optional[AttentionPoolingConfig] = None,
) -> "nn.Module":
    """Build a gated-attention MIL pooling head (ABMIL, Ilse 2018).

    See ``docs/method/ours_method_spec.md`` §12.3.
    """

    cfg = config or AttentionPoolingConfig()
    if cfg.attn_hidden_dim <= 0:
        raise ValueError(
            f"attn_hidden_dim must be positive, got {cfg.attn_hidden_dim}"
        )
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")
    if cfg.feature_dim is not None and cfg.feature_dim <= 0:
        raise ValueError(
            f"feature_dim must be positive or None, got {cfg.feature_dim}"
        )
    return _AttentionPoolingImpl(cfg)


class _AttentionPoolingImpl:
    def __new__(cls, config: AttentionPoolingConfig):  # type: ignore[override]
        torch, nn = _require_torch()

        # Compute the effective input dim for the attention gate.
        base_dim = config.feature_dim or 0
        if config.attention_on_logit:
            base_dim += 1
        if base_dim <= 0:
            # No features and no logit concat → degenerate; scoring on
            # a scalar constant is useless. Force logit concat on.
            base_dim = 1
            # (We do not mutate the frozen dataclass; the concrete module
            # below just always concatenates the logit.)

        class _Module(nn.Module):
            def __init__(self, cfg: AttentionPoolingConfig, in_dim: int):
                super().__init__()
                self.cfg = cfg
                self.in_dim = in_dim
                # Gated attention: α_i = softmax( w^T (tanh(V h_i) ⊙ sigmoid(U h_i)) )
                self.tanh_branch = nn.Sequential(
                    nn.Linear(in_dim, cfg.attn_hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(cfg.dropout),
                )
                self.gate_branch = nn.Sequential(
                    nn.Linear(in_dim, cfg.attn_hidden_dim),
                    nn.Sigmoid(),
                )
                self.attn_weight = nn.Linear(cfg.attn_hidden_dim, 1)

            def _assemble(
                self,
                instance_logits: "torch.Tensor",
                instance_features: Optional["torch.Tensor"],
            ) -> "torch.Tensor":
                n = instance_logits.shape[0]
                if instance_features is None:
                    if not self.cfg.attention_on_logit:
                        raise ValueError(
                            "AttentionPooling requires either "
                            "instance_features or attention_on_logit=True"
                        )
                    h = instance_logits.view(n, 1)
                else:
                    if instance_features.dim() != 2:
                        raise ValueError(
                            "instance_features must be 2-D [N, D], got "
                            f"{tuple(instance_features.shape)}"
                        )
                    if instance_features.shape[0] != n:
                        raise ValueError(
                            "instance_features and instance_logits must share N"
                        )
                    if (
                        self.cfg.feature_dim is not None
                        and instance_features.shape[1] != self.cfg.feature_dim
                    ):
                        raise ValueError(
                            "instance_features last dim must match "
                            f"feature_dim={self.cfg.feature_dim}, got "
                            f"{instance_features.shape[1]}"
                        )
                    h = instance_features
                    if self.cfg.attention_on_logit:
                        h = torch.cat([h, instance_logits.view(n, 1)], dim=-1)
                return h

            def forward(
                self,
                instance_logits: "torch.Tensor",
                instance_features: Optional["torch.Tensor"] = None,
                instance_types: Optional["torch.Tensor"] = None,  # noqa: ARG002
            ) -> Tuple["torch.Tensor", "torch.Tensor"]:
                if instance_logits.dim() != 1:
                    raise ValueError(
                        "instance_logits must be 1-D [N], got "
                        f"{tuple(instance_logits.shape)}"
                    )
                n = instance_logits.shape[0]
                if n == 0:
                    zero = instance_logits.new_zeros(())
                    return zero, instance_logits.new_zeros((0,))

                h = self._assemble(instance_logits, instance_features)
                gated = self.tanh_branch(h) * self.gate_branch(h)
                attn_scores = self.attn_weight(gated).squeeze(-1)  # [N]
                attn = torch.softmax(attn_scores, dim=0)
                bag_logit = (attn * instance_logits).sum()
                return bag_logit, attn

        return _Module(config, base_dim)


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------


def build_mil_pooling(
    kind: MILPoolingKind,
    *,
    topk: Optional[TopKPoolingConfig] = None,
    noisy_or: Optional[NoisyOrPoolingConfig] = None,
    attention: Optional[AttentionPoolingConfig] = None,
) -> "nn.Module":
    """Dispatch helper: build a MIL pooling head by kind.

    Intended to be called from :mod:`android_packer.models.ours` which
    reads ``OursConfig.mil_pooling`` and forwards the matching sub-config.
    """

    if kind == "topk":
        return build_topk_pooling(topk)
    if kind == "noisy_or":
        return build_noisy_or_pooling(noisy_or)
    if kind == "attention":
        return build_attention_pooling(attention)
    raise ValueError(
        f"unknown MIL pooling kind {kind!r}; expected one of "
        "'topk' / 'noisy_or' / 'attention'"
    )
