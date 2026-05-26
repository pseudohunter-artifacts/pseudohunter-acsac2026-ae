"""Typed per-instance encoder for the Ours (MIL) method.

In the Typed-Instance MIL formulation (see
``docs/method/ours_method_spec.md`` §12, batch **F-MIL-b**) an APK is a
**bag** whose instances carry one of six *types* that come for free from
the injected-packer labelling pipeline (`labeling/injected_packer_adapter.py`):

* payload kinds (``_PAYLOAD_KINDS``):
  - ``encrypted_dex``
  - ``extracted_method_body``
  - ``metadata_table``
  - ``compressed_payload``
* loader kinds (``_LOADER_KINDS``):
  - ``shim``
  - ``native_stub``

Different types carry radically different distributions (``shim`` is
pure Smali byte code; ``encrypted_dex`` looks like high-entropy noise;
``metadata_table`` is an offset table; ``native_stub`` is ELF), so a
**shared feature backbone + per-type projection head** gives us a
*typed instance representation* without duplicating the expensive
encoder six times.

This module exposes:

* :data:`TYPED_INSTANCE_TYPES` — canonical ordering of the six types
  (must stay in sync with the labeling adapter; see conftest below).
* :func:`instance_type_id` — stable ``str -> int`` mapping.
* :class:`TypedEncoderConfig` — hyper-parameters.
* :func:`build_typed_encoder` — builds a ``nn.Module`` that maps
  ``(feature [N, D], type_id [N]) -> (hidden [N, H], logit [N])``.

The per-type head is a tiny ``Linear -> GELU -> Dropout -> Linear``
stack.  Torch is lazy-imported (see §3.1 of the method spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn


__all__ = [
    "TYPED_INSTANCE_TYPES",
    "N_TYPED_INSTANCE_TYPES",
    "instance_type_id",
    "TypedEncoderConfig",
    "build_typed_encoder",
]


# ---------------------------------------------------------------------------
# Type vocabulary
# ---------------------------------------------------------------------------

#: Canonical instance-type ordering used by the MIL head.
#:
#: Order is *load-bearing*: downstream code (e.g. per-type ablation
#: configs, confusion-matrix reporters, inference dumps) keys on the
#: integer id assigned here.  **Do not reorder** without bumping a major
#: version tag in the configs.
#:
#: This list mirrors ``_PAYLOAD_KINDS`` ∪ ``_LOADER_KINDS`` from
#: ``src/android_packer/labeling/injected_packer_adapter.py`` so that
#: Path-A ground-truth labels can be read straight into the MIL trainer
#: with **zero schema churn** — a hard constraint from
#: ``docs/method/ours_method_spec.md`` §8.
#:
#: L42 fix (2026-05-07): added ``benign_other`` as the 7th type for
#: non-packer-introduced APK objects (the ~1500 native ``res/``,
#: ``kotlin/``, ``META-INF/`` instances per bag).  Before this fix,
#: benign objects were mis-routed through the ``shim`` head by the
#: path-based default-fallback in ``baselines/ours.py::_object_instance_type``.
TYPED_INSTANCE_TYPES: Tuple[str, ...] = (
    "encrypted_dex",
    "extracted_method_body",
    "metadata_table",
    "compressed_payload",
    "shim",
    "native_stub",
    "benign_other",
)

N_TYPED_INSTANCE_TYPES: int = len(TYPED_INSTANCE_TYPES)

_TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPED_INSTANCE_TYPES)}


def instance_type_id(type_name: str) -> int:
    """Return the integer id for a typed-instance name.

    Raises ``KeyError`` on an unknown type so typos surface immediately
    rather than silently being bucketed as type 0.
    """

    try:
        return _TYPE_TO_ID[type_name]
    except KeyError as exc:
        raise KeyError(
            f"unknown typed-instance name {type_name!r}; expected one of "
            f"{list(TYPED_INSTANCE_TYPES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedEncoderConfig:
    """Hyper-parameters for :func:`build_typed_encoder`.

    Parameters
    ----------
    input_dim:
        Dimensionality of the shared instance feature vector produced by
        the byte / handcrafted backbone.  Matches
        :class:`android_packer.features.handcrafted.HandcraftedFeatureConfig`
        at 15 by default, but the MIL architecture is bigger than the
        Lite baseline so we keep it parametric.
    hidden_dim:
        Width of the shared trunk.
    num_trunk_layers:
        Number of ``Linear -> GELU -> Dropout`` blocks in the shared trunk.
    head_hidden_dim:
        Width of each per-type head's hidden layer.
    dropout:
        Dropout used in both trunk and heads.
    n_types:
        Number of typed-instance classes; defaults to
        :data:`N_TYPED_INSTANCE_TYPES`.  Kept parametric only so unit
        tests can stub smaller values.
    """

    input_dim: int = 15
    hidden_dim: int = 128
    num_trunk_layers: int = 2
    head_hidden_dim: int = 64
    dropout: float = 0.1
    n_types: int = N_TYPED_INSTANCE_TYPES


# ---------------------------------------------------------------------------
# Lazy torch helper
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TypedEncoder requires torch. Install via "
            "``pip install -e \".[dl]\"`` (see AGENTS.md §2)."
        ) from exc
    return torch, nn


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_typed_encoder(
    config: Optional[TypedEncoderConfig] = None,
) -> "nn.Module":
    """Build a typed per-instance encoder.

    The returned module's ``forward`` signature is::

        forward(instance_features: Tensor[N, D],
                instance_types:    Tensor[N]  int64)
          -> (hidden_states: Tensor[N, H],
              instance_logit: Tensor[N])

    The per-type head picks the correct row of the type-keyed weight
    bank by scatter-reading; unknown type ids cause a ``ValueError``.
    """

    cfg = config or TypedEncoderConfig()
    if cfg.input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {cfg.input_dim}")
    if cfg.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
    if cfg.head_hidden_dim <= 0:
        raise ValueError(
            f"head_hidden_dim must be positive, got {cfg.head_hidden_dim}"
        )
    if cfg.num_trunk_layers < 1:
        raise ValueError(
            f"num_trunk_layers must be >= 1, got {cfg.num_trunk_layers}"
        )
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")
    if cfg.n_types < 1:
        raise ValueError(f"n_types must be >= 1, got {cfg.n_types}")

    return _TypedEncoderImpl(cfg)


class _TypedEncoderImpl:
    def __new__(cls, config: TypedEncoderConfig):  # type: ignore[override]
        _, nn = _require_torch()

        class _Module(nn.Module):
            def __init__(self, cfg: TypedEncoderConfig):
                super().__init__()
                self.cfg = cfg

                # Shared trunk.
                trunk: List[Any] = []
                in_dim = cfg.input_dim
                for _ in range(cfg.num_trunk_layers):
                    trunk.extend(
                        [
                            nn.Linear(in_dim, cfg.hidden_dim),
                            nn.GELU(),
                            nn.Dropout(cfg.dropout),
                        ]
                    )
                    in_dim = cfg.hidden_dim
                self.trunk = nn.Sequential(*trunk)

                # Per-type projection bank: [T, H, head_hidden_dim] + bias.
                # Implemented as a stack of Linear layers; using a raw
                # parameter tensor + bmm would be slightly faster but
                # harder to unit-test and would lose nn.Linear init.
                self.type_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.hidden_dim, cfg.head_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(cfg.dropout),
                        )
                        for _ in range(cfg.n_types)
                    ]
                )
                # Shared instance classifier (one logit per instance).
                self.instance_head = nn.Linear(cfg.head_hidden_dim, 1)

            def forward(
                self,
                instance_features: "torch.Tensor",
                instance_types: "torch.Tensor",
            ) -> Tuple["torch.Tensor", "torch.Tensor"]:
                if instance_features.dim() != 2:
                    raise ValueError(
                        "instance_features must be 2-D [N, D], got "
                        f"{tuple(instance_features.shape)}"
                    )
                if instance_features.shape[1] != self.cfg.input_dim:
                    raise ValueError(
                        "instance_features last dim must match input_dim"
                        f" {self.cfg.input_dim}, got "
                        f"{instance_features.shape[1]}"
                    )
                if instance_types.dim() != 1:
                    raise ValueError(
                        "instance_types must be 1-D [N], got "
                        f"{tuple(instance_types.shape)}"
                    )
                n = instance_features.shape[0]
                if instance_types.shape[0] != n:
                    raise ValueError(
                        "instance_features and instance_types must share N"
                    )
                if n == 0:
                    hidden = instance_features.new_zeros((0, self.cfg.head_hidden_dim))
                    logits = instance_features.new_zeros((0,))
                    return hidden, logits

                type_min = int(instance_types.min().item())
                type_max = int(instance_types.max().item())
                if type_min < 0 or type_max >= self.cfg.n_types:
                    raise ValueError(
                        "instance_types out of range "
                        f"[0, {self.cfg.n_types}); got [{type_min}, {type_max}]"
                    )

                trunk_out = self.trunk(instance_features)  # [N, hidden_dim]

                # Per-type routing: for each unique type id, take the
                # sub-batch of instances with that id, project with the
                # matching head, scatter back.  O(T) passes where T ≤ 6.
                hidden_out = trunk_out.new_zeros(
                    (n, self.cfg.head_hidden_dim)
                )
                for t in range(self.cfg.n_types):
                    mask = instance_types == t
                    if not bool(mask.any()):
                        continue
                    sub = trunk_out[mask]
                    hidden_out[mask] = self.type_proj[t](sub)

                instance_logit = self.instance_head(hidden_out).squeeze(-1)  # [N]
                return hidden_out, instance_logit

        return _Module(config)
