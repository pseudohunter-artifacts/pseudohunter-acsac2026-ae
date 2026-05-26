"""Packed / unpacked contrastive pre-training (F-MIL-d).

Sellpoint 3 of the novelty uplift (see ``docs/research_framing.md``
§4.2 and ``docs/method/ours_method_spec.md`` §12.3):

    "Represent an APK with two heads — :math:`h_{\\mathrm{app}}`
    (semantic content, invariant to packing) and :math:`h_{\\mathrm{pack}}`
    (packing residual, what packers *add* on top).  Contrastive training
    uses Track B v2 pairs as InfoNCE positive / negative examples."

Why two heads, not one
----------------------
A single joint embedding would force the model to sacrifice either
"benign ≈ packed(benign)" (app-semantic) or "benign ≉ packed(benign)"
(packer-vs-no-packer).  Two heads resolve the conflict cleanly:

* ``h_app(benign_i) ≈ h_app(packed_i)`` for the same app ``i``.
  This is an **invariance** objective → standard symmetric InfoNCE.
* ``h_pack(packed_i) - h_pack(benign_i)`` should concentrate the packer's
  signature; different packers on the same app should push
  :math:`\\Delta z` into **packer-specific clusters**.
  This is a **residual** objective — InfoNCE on the *difference* vectors
  with packer family as the class.

The 18 real-world packed / benign pairs in Track B v2 are the ground-truth
supply.  (9 benign APKs × 2 packer families = 18 pairs in the `s5` +
`s6` subtree; see ``docs/workstreams/track_b/README.md``.)

Scope of this module
--------------------
This file lands **loss + pairing** only.  The data wiring (APK →
feature tensor) lives in the MIL trainer (F-MIL-e) so we can mock it
in unit tests without touching disk.  Public entry points:

* :class:`ContrastivePairBatch` — dataclass carrying the stacked
  anchor / positive / (optional) packer-family tensors.
* :class:`ContrastiveConfig` — temperature + projection-dim +
  residual-weight knobs.
* :func:`info_nce_app_head` — symmetric InfoNCE for the app-semantic
  head.
* :func:`info_nce_pack_residual` — residual InfoNCE for the packer
  head.
* :func:`compute_contrastive_loss` — combined
  :math:`L = L_{\\mathrm{app}} + \\lambda_{\\mathrm{pack}} \\cdot L_{\\mathrm{pack}}`.

Lazy-torch contract (§3.1): torch is only pulled in by the loss
functions; configs / batching helpers are pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "ContrastiveConfig",
    "ContrastivePairBatch",
    "build_pair_batch",
    "compute_contrastive_loss",
    "info_nce_app_head",
    "info_nce_pack_residual",
]


# ---------------------------------------------------------------------------
# Config + batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastiveConfig:
    """Hyper-parameters for the two-head contrastive pre-training step.

    Parameters
    ----------
    temperature_app:
        ``τ`` for the app-semantic InfoNCE (SimCLR default 0.1).
    temperature_pack:
        ``τ`` for the packer-residual InfoNCE.  Usually slightly higher
        because packer clusters are smaller & louder.
    pack_loss_weight:
        ``λ_pack`` — weight on the residual head in
        :func:`compute_contrastive_loss`.  Default 1.0.  Set to 0 for
        the ablation row "app-head only" in §12.5.
    normalize:
        Whether to L2-normalise embeddings before dot-product.  Standard
        SimCLR-style contrastive asks for True; kept parametric for
        ablation sanity checks.
    """

    temperature_app: float = 0.1
    temperature_pack: float = 0.15
    pack_loss_weight: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.temperature_app <= 0:
            raise ValueError(
                f"temperature_app must be positive, got {self.temperature_app}"
            )
        if self.temperature_pack <= 0:
            raise ValueError(
                f"temperature_pack must be positive, got {self.temperature_pack}"
            )
        if self.pack_loss_weight < 0:
            raise ValueError(
                f"pack_loss_weight must be >= 0, got {self.pack_loss_weight}"
            )


@dataclass(frozen=True)
class ContrastivePairBatch:
    """A batch of ``(benign, packed, packer_family)`` triples.

    * ``anchors``: ``[B, D]`` benign APK embeddings.
    * ``positives``: ``[B, D]`` packed APK embeddings, ``positives[i]``
      being the packed counterpart of ``anchors[i]``.
    * ``packer_ids``: optional ``[B]`` int labels (0..num_packers-1)
      grouping pairs by packer family.  Used by
      :func:`info_nce_pack_residual`.  When all pairs in the batch
      share a packer id, the residual head has no negatives and falls
      back gracefully (see implementation note).

    The class holds **torch tensors** by design (the loss functions
    consume them directly); :func:`build_pair_batch` is a thin
    constructor that also validates shape alignment.
    """

    anchors: Any
    positives: Any
    packer_ids: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.anchors is None or self.positives is None:
            raise ValueError("anchors and positives must be provided")


def build_pair_batch(
    anchors: Any,
    positives: Any,
    packer_ids: Optional[Any] = None,
) -> ContrastivePairBatch:
    """Validate and wrap a pair batch.

    Shape checks are performed against the tensors' ``shape`` attribute
    so this helper works with torch, numpy-like, or test doubles.
    """

    a_shape = tuple(anchors.shape)
    p_shape = tuple(positives.shape)
    if len(a_shape) != 2 or len(p_shape) != 2:
        raise ValueError(
            f"anchors/positives must be 2-D [B, D]; got {a_shape} and {p_shape}"
        )
    if a_shape[0] != p_shape[0]:
        raise ValueError(
            f"anchors and positives must share batch size, got "
            f"{a_shape[0]} vs {p_shape[0]}"
        )
    if a_shape[1] != p_shape[1]:
        raise ValueError(
            f"anchors and positives must share feature dim, got "
            f"{a_shape[1]} vs {p_shape[1]}"
        )
    if packer_ids is not None:
        pid_shape = tuple(packer_ids.shape)
        if len(pid_shape) != 1 or pid_shape[0] != a_shape[0]:
            raise ValueError(
                f"packer_ids must be 1-D and match batch size {a_shape[0]}, "
                f"got shape {pid_shape}"
            )
    return ContrastivePairBatch(
        anchors=anchors, positives=positives, packer_ids=packer_ids
    )


# ---------------------------------------------------------------------------
# Lazy torch
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "contrastive loss requires torch. Install via "
            "``pip install -e \".[dl]\"`` (see AGENTS.md §2)."
        ) from exc
    return torch, F


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def info_nce_app_head(
    z_a: Any,
    z_p: Any,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> Any:
    """Symmetric InfoNCE for the app-semantic head.

    Treats ``(z_a[i], z_p[i])`` as a positive pair and all other batch
    members as negatives.  Returns the mean of the two symmetric
    directions (``a -> p`` and ``p -> a``), matching SimCLR.

    Requires batch size ≥ 2 (needs at least one negative).
    """

    torch, F = _require_torch()
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if z_a.shape != z_p.shape:
        raise ValueError(
            f"z_a and z_p must share shape, got {tuple(z_a.shape)} vs "
            f"{tuple(z_p.shape)}"
        )
    if z_a.dim() != 2:
        raise ValueError(
            f"z_a/z_p must be 2-D, got dim={z_a.dim()}"
        )
    batch_size = z_a.shape[0]
    if batch_size < 2:
        raise ValueError(
            f"info_nce_app_head needs batch size >= 2 for negatives, got "
            f"{batch_size}"
        )

    if normalize:
        z_a = F.normalize(z_a, dim=-1)
        z_p = F.normalize(z_p, dim=-1)

    logits = z_a @ z_p.t() / temperature  # [B, B]
    targets = torch.arange(batch_size, device=z_a.device)
    loss_a2p = F.cross_entropy(logits, targets)
    loss_p2a = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a2p + loss_p2a)


def info_nce_pack_residual(
    z_a: Any,
    z_p: Any,
    packer_ids: Any,
    *,
    temperature: float = 0.15,
    normalize: bool = True,
) -> Any:
    """Residual InfoNCE: treats ``Δ = z_p - z_a`` as the packer signature.

    Samples that share a ``packer_ids`` value are positives; samples with
    different packer ids are negatives.  This is a supervised contrastive
    objective (Khosla et al. 2020) on the delta embedding.

    Returns a zero tensor (with grad connected) when the batch contains
    only a single packer id — that situation is *not* an error (mini-batch
    composition can be uneven) but there are no negatives, so the loss is
    undefined; returning 0-with-grad keeps the optimiser step idempotent.
    """

    torch, F = _require_torch()
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if z_a.shape != z_p.shape:
        raise ValueError(
            f"z_a and z_p must share shape, got {tuple(z_a.shape)} vs "
            f"{tuple(z_p.shape)}"
        )
    if z_a.dim() != 2:
        raise ValueError(f"z_a/z_p must be 2-D, got dim={z_a.dim()}")
    batch_size = z_a.shape[0]
    pid_shape = tuple(packer_ids.shape)
    if len(pid_shape) != 1 or pid_shape[0] != batch_size:
        raise ValueError(
            f"packer_ids must be 1-D with len == batch_size {batch_size}, "
            f"got {pid_shape}"
        )

    delta = z_p - z_a
    if normalize:
        delta = F.normalize(delta, dim=-1)

    # Supervised contrastive: for each anchor i, positives are all j
    # where packer_ids[i] == packer_ids[j] and j != i.  Edge case: if a
    # packer_id is singleton in this batch, its rows have no positives
    # and we exclude them from the loss.
    same_packer = (packer_ids.unsqueeze(0) == packer_ids.unsqueeze(1))
    # Mask out self-similarity — an anchor is never its own positive.
    eye = torch.eye(batch_size, dtype=torch.bool, device=delta.device)
    pos_mask = same_packer & ~eye

    has_any_positive = pos_mask.any(dim=1)
    if not bool(has_any_positive.any()):
        # No valid anchor / positive pairing exists in this batch.
        return (delta.sum() * 0.0)  # 0 with autograd trace preserved

    sim = delta @ delta.t() / temperature  # [B, B]

    # Numerically stable log-softmax over "all non-self" columns.
    mask_logits = sim.masked_fill(eye, float("-inf"))
    log_prob = mask_logits - torch.logsumexp(mask_logits, dim=1, keepdim=True)
    # The self-column is ``-inf`` by construction; replace with 0 so that
    # the masked product below is well defined (``0 * -inf = nan`` in
    # IEEE-754 would otherwise poison the sum).  The ``pos_mask`` zeros
    # out those positions anyway.
    log_prob = log_prob.masked_fill(eye, 0.0)

    # Mean log-prob over positives for each anchor that has any.
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count

    # Only include anchors with at least one positive.
    loss = -mean_log_prob_pos[has_any_positive].mean()
    return loss


def compute_contrastive_loss(
    batch: ContrastivePairBatch,
    config: ContrastiveConfig,
) -> Any:
    """Combined ``L_app + λ_pack · L_pack`` contrastive loss.

    When ``batch.packer_ids`` is ``None`` or ``config.pack_loss_weight == 0``
    the residual branch is skipped entirely (keeps it out of autograd
    for the ablation run).
    """

    loss_app = info_nce_app_head(
        batch.anchors,
        batch.positives,
        temperature=config.temperature_app,
        normalize=config.normalize,
    )

    if config.pack_loss_weight == 0.0 or batch.packer_ids is None:
        return loss_app

    loss_pack = info_nce_pack_residual(
        batch.anchors,
        batch.positives,
        batch.packer_ids,
        temperature=config.temperature_pack,
        normalize=config.normalize,
    )
    return loss_app + config.pack_loss_weight * loss_pack
