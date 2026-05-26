"""PayloadHunter-Lite: Stage A simplified Ours method model components.

Implements the architecture defined in
:mod:`docs/method/ours_method_spec.md` §11.3:

* ``LiteRegionScorer`` — shallow MLP over handcrafted region features
  (feature_dim -> 128 -> 128 -> 1 logit).  ~20K params at the spec's
  default 34-dim input.
* ``LiteObjectAggregator`` — attention pooling over regions belonging
  to the same ZIP/DEX object; returns an object-level score plus the
  attention distribution so downstream code can dump it for case
  studies / visualisations.

Both modules follow the lazy-torch-import contract from
:mod:`docs/method/ours_method_spec.md` §3.1: torch is never imported at
module load time so ``from android_packer.models import ...`` does not
require the ``[dl]`` optional extra.  Instantiation raises
``ImportError`` with an actionable hint if torch is missing.

Design notes
------------
* ``feature_dim`` is a constructor argument rather than a hard-coded 34
  because the feature assembly (:mod:`android_packer.features.handcrafted`)
  is landing in a follow-up batch; keeping the dim parametric lets us
  land the scorer / aggregator skeletons now without pinning a feature
  vocabulary that may still shift.
* The attention head sees ``[region_feature || region_logit]`` so that
  attention weights condition on both the raw features and the
  scorer's own judgement — matches spec §11.3.3.
* Dropout defaults (0.2 for scorer, 0.1 for aggregator) follow spec
  §11.3.2 / §11.3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - type-only import
    import torch
    from torch import nn


__all__ = [
    "LiteRegionScorerConfig",
    "LiteObjectAggregatorConfig",
    "build_lite_region_scorer",
    "build_lite_object_aggregator",
]


# ---------------------------------------------------------------------------
# Configs (pure stdlib, no torch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteRegionScorerConfig:
    """Hyper-parameters for :class:`LiteRegionScorer`.

    ``feature_dim`` defaults to 34 to match
    :mod:`docs/method/ours_method_spec.md` §11.3.1, but any positive
    int is accepted; unit tests use a smaller dim for smoke testing.
    """

    feature_dim: int = 34
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.2
    activation: str = "gelu"  # one of {"gelu", "relu"}


@dataclass(frozen=True)
class LiteObjectAggregatorConfig:
    """Hyper-parameters for :class:`LiteObjectAggregator`.

    ``input_dim`` is the dimensionality of the region feature vector
    that will be concatenated with the region logit before attention
    scoring; i.e. the aggregator sees ``[feature || logit]`` of width
    ``input_dim + 1``.
    """

    input_dim: int = 34
    attn_hidden_dim: int = 64
    dropout: float = 0.1
    # When True, the aggregator also exposes the attention distribution
    # so the training loop can add an entropy regulariser (spec §11.3.4)
    # and case studies can visualise "which region was the object score
    # attending to".
    return_attention: bool = True


# ---------------------------------------------------------------------------
# Lazy torch helpers
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    """Import torch + nn on demand, raising a helpful error if missing."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without [dl]
        raise ImportError(
            "PayloadHunter-Lite model components require torch. "
            "Install the optional dependencies via "
            "``pip install -e \".[dl]\"`` (see AGENTS.md §2)."
        ) from exc
    return torch, nn


def _resolve_activation(name: str) -> Any:
    """Return the torch activation module class by name."""

    _, nn = _require_torch()
    name = name.lower()
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    raise ValueError(
        f"unsupported activation {name!r}; expected one of 'gelu' / 'relu'"
    )


# ---------------------------------------------------------------------------
# Module factories (return torch.nn.Module instances)
# ---------------------------------------------------------------------------


def build_lite_region_scorer(
    config: Optional[LiteRegionScorerConfig] = None,
) -> "nn.Module":
    """Build a ``LiteRegionScorer`` nn.Module from ``config``.

    Factory wrapper kept separate from the class so callers that want
    only the nn.Module (and do not need to subclass) do not have to
    touch the ``_Impl`` pattern.
    """

    cfg = config or LiteRegionScorerConfig()
    if cfg.feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {cfg.feature_dim}")
    if cfg.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
    if cfg.num_hidden_layers < 1:
        raise ValueError(
            f"num_hidden_layers must be >= 1, got {cfg.num_hidden_layers}"
        )
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")

    _, nn = _require_torch()
    activation_cls = _resolve_activation(cfg.activation)

    layers = []
    in_dim = cfg.feature_dim
    for _ in range(cfg.num_hidden_layers):
        layers.extend(
            [
                nn.Linear(in_dim, cfg.hidden_dim),
                activation_cls(),
                nn.Dropout(cfg.dropout),
            ]
        )
        in_dim = cfg.hidden_dim
    layers.append(nn.Linear(in_dim, 1))  # logit
    return nn.Sequential(*layers)


def build_lite_object_aggregator(
    config: Optional[LiteObjectAggregatorConfig] = None,
) -> "nn.Module":
    """Build a :class:`LiteObjectAggregator` nn.Module from ``config``."""

    cfg = config or LiteObjectAggregatorConfig()
    if cfg.input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {cfg.input_dim}")
    if cfg.attn_hidden_dim <= 0:
        raise ValueError(
            f"attn_hidden_dim must be positive, got {cfg.attn_hidden_dim}"
        )
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")

    return _LiteObjectAggregatorImpl(cfg)


# ---------------------------------------------------------------------------
# Public nn.Module classes (thin wrappers around factories)
# ---------------------------------------------------------------------------


class _LiteObjectAggregatorImpl:
    """Internal implementation. Use :func:`build_lite_object_aggregator`.

    We cannot inherit from ``torch.nn.Module`` at module scope without
    importing torch, so the implementation is defined lazily inside
    :meth:`__init_subclass__`-style factory. To keep the code simple
    this class *becomes* an nn.Module only at instantiation time via a
    tiny wrapper object assembled below.
    """

    def __new__(cls, config: LiteObjectAggregatorConfig):  # type: ignore[override]
        torch, nn = _require_torch()

        class _Module(nn.Module):
            """Attention-pooling aggregator; see spec §11.3.3."""

            def __init__(self, cfg: LiteObjectAggregatorConfig):
                super().__init__()
                self.cfg = cfg
                self.attention = nn.Sequential(
                    nn.Linear(cfg.input_dim + 1, cfg.attn_hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.attn_hidden_dim, 1),
                )

            def forward(
                self,
                region_features: "torch.Tensor",
                region_logits: "torch.Tensor",
            ) -> Any:
                """Aggregate a variable number of region scores.

                Parameters
                ----------
                region_features:
                    ``[N, input_dim]`` tensor of raw region features.
                region_logits:
                    ``[N]`` or ``[N, 1]`` tensor of per-region logits
                    coming from :class:`LiteRegionScorer`.

                Returns
                -------
                If ``cfg.return_attention`` is True:
                    ``(object_logit [scalar], attention_weights [N])``
                Otherwise:
                    ``object_logit`` scalar tensor.
                """

                if region_features.dim() != 2:
                    raise ValueError(
                        "region_features must be 2-D [N, D], got shape "
                        f"{tuple(region_features.shape)}"
                    )
                if region_features.shape[1] != self.cfg.input_dim:
                    raise ValueError(
                        "region_features last dim must match input_dim"
                        f" {self.cfg.input_dim}, got {region_features.shape[1]}"
                    )
                # Normalise region_logits to [N, 1] for concat.
                if region_logits.dim() == 1:
                    region_logits = region_logits.unsqueeze(-1)
                if region_logits.shape[0] != region_features.shape[0]:
                    raise ValueError(
                        "region_features and region_logits must share N; "
                        f"got {region_features.shape[0]} vs {region_logits.shape[0]}"
                    )

                attn_input = torch.cat([region_features, region_logits], dim=-1)
                attn_scores = self.attention(attn_input).squeeze(-1)  # [N]
                attn_weights = torch.softmax(attn_scores, dim=0)  # [N]
                object_logit = (attn_weights * region_logits.squeeze(-1)).sum()

                if self.cfg.return_attention:
                    return object_logit, attn_weights
                return object_logit

        return _Module(config)
