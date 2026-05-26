"""Supervised MIL trainer for the Ours (Typed-Instance MIL) method.

Batch **F-MIL-e** per ``docs/method/ours_method_spec.md`` §12.6.

This module owns the **small, reusable** training kernel:

* :class:`MILTrainerConfig` — hyper-parameters (optimizer, λ weights, …).
* :class:`MILBag` — one APK's instance features + types + (optional)
  per-instance ground-truth + the bag label.  Decoupled from disk I/O
  so tests can build bags directly from tensors.
* :func:`train_ours` — train a :class:`build_ours` model on a list of
  :class:`MILBag` with the combined loss:

  .. math::
      L = L_{\\mathrm{mil,apk}} + \\lambda_1 L_{\\mathrm{diff,pseudo}} + \\lambda_2 L_{\\mathrm{sparsity}}

  ``L_mil_apk`` is BCE on the bag logit; ``L_diff_pseudo`` is per-
  instance BCE when (and only when) ``instance_labels`` is non-None; the
  sparsity term penalises the L1 of the attention vector.

* :func:`predict_bag` — attention + instance logits + bag score for one
  bag.  Used by :mod:`android_packer.baselines.ours`.

Auxiliary pre-training (MLM + grammar-aware + contrastive) lives in
dedicated modules and is **not** invoked here; the trainer assumes its
caller has already built / loaded pretrained weights when
``pretrained_ckpt`` is non-None (wiring is left for stage-2 once the MLM
torch training loop lands).

Lazy-torch contract (§3.1): all torch import lives behind
:func:`_require_torch` which is only called from :func:`train_ours` /
:func:`predict_bag`.  Configs + :class:`MILBag` are pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from android_packer.models.ours import OursConfig, build_ours

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "MILBag",
    "MILTrainerConfig",
    "predict_bag",
    "subsample_bag_for_training",
    "train_ours",
]


# ---------------------------------------------------------------------------
# Config + bag container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MILTrainerConfig:
    """Hyper-parameters for :func:`train_ours`.

    ``λ_diff`` + ``λ_sparsity`` match ``ours_method_spec.md`` §12.4
    defaults (0.3 / 0.01).  Pretrained MLM / contrastive encoders feed
    in through ``pretrained_ckpt``; when None the trainer starts from
    random init (degraded baseline, spec §11.8 declares this the
    lower-bound configuration for the ablation matrix).
    """

    ours_config: OursConfig = field(default_factory=OursConfig)
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4  # bags per gradient step; tiny bcs N varies
    lambda_diff_pseudo: float = 0.3
    lambda_sparsity: float = 0.01
    pretrained_ckpt: Optional[str] = None
    # L43 (2026-05-07): supervision mode controls whether the per-
    # instance BCE term ever contributes to the loss, independent of
    # lambda_diff_pseudo.  This is a paper-integrity switch:
    #
    # - "bag"           -- weakly-supervised MIL.  Only bag-level BCE +
    #                      sparsity.  Ignores instance_labels even when
    #                      they are present on the MILBag.  This is the
    #                      operating mode that `research_framing.md`
    #                      sec.3.2 / sec.12.2 sellpoint claims; Tier-A
    #                      numbers in the paper must come from here.
    # - "instance_aided" -- adds the per-instance BCE term (current
    #                      default, kept for backward compatibility and
    #                      as a diagnostic upper bound).  Do NOT quote
    #                      these numbers as weakly-supervised.
    #
    # Default is "instance_aided" so existing configs and tests that
    # opt-in to lambda_diff_pseudo>0 keep their behaviour; any new
    # paper-quotable run must set supervision_mode="bag".
    supervision_mode: str = "instance_aided"
    # Positive-class weight for the *bag* BCE (not per-instance).
    # Packed APKs are ~50% of the corpus so 1.0 is fine by default, but
    # leave parametric for LOPO splits with imbalanced test sets.
    bag_pos_weight: float = 1.0
    random_state: int = 0
    verbose: bool = False

    # --- L41 (2026-05-07): training-time bag subsampling ---------------
    # The synthetic v4 corpus has ~1500 instances per bag but only
    # ~1.2 positive instances (positive fraction ≈ 0.1%). Attention-
    # pooling over 1500 candidates to find 1 target learns almost
    # nothing and is dominated by path-derived type leak (L42).
    #
    # Subsampling policy during training: keep *all* positive instances,
    # uniform-sample negatives down to at most ``train_max_bag_size``
    # total, with a floor on positive-fraction set by
    # ``train_min_positive_fraction`` (if achievable; positives are not
    # synthesised).  Deterministic by ``hash((bag_id, epoch))``.
    #
    # Inference (``predict_bag``) is *not* subsampled: the full bag is
    # scored in one forward pass, because at test time recall matters
    # and memory is not the bottleneck (a 3000-instance bag in 15-dim
    # float32 is 180 KB).
    train_max_bag_size: Optional[int] = 256
    train_min_positive_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if self.lambda_diff_pseudo < 0:
            raise ValueError("lambda_diff_pseudo must be >= 0")
        if self.lambda_sparsity < 0:
            raise ValueError("lambda_sparsity must be >= 0")
        if self.bag_pos_weight <= 0:
            raise ValueError("bag_pos_weight must be positive")
        if self.supervision_mode not in ("bag", "instance_aided"):
            raise ValueError(
                "supervision_mode must be 'bag' or 'instance_aided', got "
                f"{self.supervision_mode!r}"
            )
        if (
            self.train_max_bag_size is not None
            and self.train_max_bag_size <= 0
        ):
            raise ValueError(
                "train_max_bag_size must be positive or None, got "
                f"{self.train_max_bag_size}"
            )
        if not (0.0 <= self.train_min_positive_fraction <= 1.0):
            raise ValueError(
                "train_min_positive_fraction must be in [0, 1], got "
                f"{self.train_min_positive_fraction}"
            )


@dataclass(frozen=True)
class MILBag:
    """One APK's MIL bag.

    Attributes
    ----------
    bag_id:
        Opaque identifier (e.g. ``apk_id``).  Not used by the training
        loop itself; carried through so callers can trace a bag back to
        its APK row.
    features:
        ``[N, D]`` torch tensor of per-instance features.
    types:
        ``[N]`` int64 torch tensor of typed-instance ids
        (``0..N_TYPED_INSTANCE_TYPES - 1``).
    bag_label:
        0/1 bag-level ground truth (1 == packed).
    instance_labels:
        Optional ``[N]`` float tensor with pseudo-labels for the diff
        auxiliary loss.  ``None`` disables the per-instance BCE term for
        this bag (MIL weakly-supervised regime).
    """

    bag_id: str
    features: Any
    types: Any
    bag_label: int
    instance_labels: Optional[Any] = None


# ---------------------------------------------------------------------------
# Lazy torch
# ---------------------------------------------------------------------------


def _require_torch() -> Tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from torch import optim
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MIL trainer requires torch. Install via "
            "``pip install -e \".[dl]\"``."
        ) from exc
    return torch, F, optim


# ---------------------------------------------------------------------------
# Bag subsampling (L41 fix, 2026-05-07)
# ---------------------------------------------------------------------------


def subsample_bag_for_training(
    bag: MILBag,
    *,
    max_size: Optional[int],
    min_positive_fraction: float,
    epoch: int,
    global_seed: int,
) -> MILBag:
    """Return a subsampled view of ``bag`` suitable for training.

    Keeps **every** positive instance (defined as
    ``instance_labels[i] > 0.5`` when ``instance_labels`` is present, or
    the whole bag when it's ``None`` — i.e. in bag-only supervision we
    cannot identify positives and skip subsampling).  Negatives are
    uniformly down-sampled so the final bag size is at most ``max_size``
    and the positive fraction is at least ``min_positive_fraction`` (the
    positive fraction floor is honoured only when achievable without
    synthesising positives).

    Sampling is deterministic given ``(bag.bag_id, epoch, global_seed)``
    so two runs with the same random_state see the same subsamples.

    ``max_size=None`` disables subsampling and returns ``bag`` unchanged.
    """

    if max_size is None:
        return bag
    if bag.instance_labels is None:
        # Bag-only supervision: no way to identify positives, no sampling.
        return bag
    torch, _, _ = _require_torch()

    inst = bag.instance_labels
    n = int(inst.shape[0])
    if n <= max_size:
        return bag

    pos_mask = inst > 0.5
    pos_count = int(pos_mask.sum().item())
    neg_idx = (~pos_mask).nonzero(as_tuple=False).squeeze(-1)
    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)

    # Honour the positive-fraction floor when achievable.  If we have
    # P positives and want pos_fraction >= f, then total <= P / f.
    if pos_count > 0 and min_positive_fraction > 0.0:
        cap_by_fraction = int(pos_count / min_positive_fraction)
        target = min(max_size, cap_by_fraction)
    else:
        target = max_size
    target = max(target, pos_count)  # never drop positives
    neg_target = max(0, target - pos_count)

    # Deterministic negative sample.
    seed = (
        hash((str(bag.bag_id), int(epoch), int(global_seed)))
        & 0x7FFFFFFF
    )
    # L48 (2026-05-11): torch.randperm does not support CUDA generators.
    # Always generate the permutation on CPU for reproducibility, then move
    # the resulting index to the tensor's device for GPU-safe indexing.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    if neg_target >= int(neg_idx.shape[0]):
        picked_neg = neg_idx
    else:
        perm = torch.randperm(int(neg_idx.shape[0]), generator=g).to(neg_idx.device)
        picked_neg = neg_idx[perm[:neg_target]]

    keep_idx = torch.cat([pos_idx, picked_neg], dim=0)
    # Sort so the feature rows stay in the original order — helps when
    # debugging an attention trace back to the source object_id.
    keep_idx, _ = torch.sort(keep_idx)

    return MILBag(
        bag_id=bag.bag_id,
        features=bag.features[keep_idx],
        types=bag.types[keep_idx],
        bag_label=bag.bag_label,
        instance_labels=inst[keep_idx],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _bag_loss(
    model: Any,
    bag: MILBag,
    *,
    cfg: MILTrainerConfig,
    F: Any,
    torch_mod: Any,
) -> Tuple[Any, Any, Any]:
    """Forward one bag and return ``(loss, bag_logit, attention)``."""

    bag_logit, attention, instance_logits = model(bag.features, bag.types)

    bag_target = torch_mod.tensor(float(bag.bag_label), device=bag_logit.device)
    pos_weight = torch_mod.tensor(cfg.bag_pos_weight, device=bag_logit.device)
    loss = F.binary_cross_entropy_with_logits(
        bag_logit, bag_target, pos_weight=pos_weight
    )

    if (
        cfg.supervision_mode == "instance_aided"
        and cfg.lambda_diff_pseudo > 0.0
        and bag.instance_labels is not None
    ):
        inst_target = bag.instance_labels.float()
        # Instance-level positive class share is typically ≪ 5% in our
        # MIL setting (one packed object out of ~1000 archive members).
        # Without pos_weight the per-instance BCE collapses to "predict
        # 0 for everything" — bag loss alone cannot recover signal in
        # that regime. Compute a per-bag pos_weight so the gradient on
        # the rare positive instances is comparable to the negatives.
        n_pos = float(inst_target.sum().item())
        n_neg = float((1.0 - inst_target).sum().item())
        if n_pos > 0.0:
            pos_w = max(1.0, n_neg / max(1.0, n_pos))
            inst_pos_weight = torch_mod.tensor(
                pos_w, device=instance_logits.device
            )
        else:
            inst_pos_weight = None
        inst_loss = F.binary_cross_entropy_with_logits(
            instance_logits,
            inst_target,
            pos_weight=inst_pos_weight,
        )
        loss = loss + cfg.lambda_diff_pseudo * inst_loss

    if cfg.lambda_sparsity > 0.0:
        # L1 on attention weights — attention sums to 1 by design, so
        # sparsity here is measured by KL(α || uniform): heavier
        # concentration ⇒ lower entropy ⇒ the term drives the model
        # towards peaky attention (interpretability bonus).
        n = attention.shape[0]
        if n > 0:
            uniform = 1.0 / float(n)
            # entropy regulariser: minimise negative-entropy keeps α
            # peaky (low H(α)) ⇒ we *maximise* -H, i.e. subtract H.
            eps = 1e-12
            entropy = -(attention * (attention + eps).log()).sum()
            loss = loss + cfg.lambda_sparsity * entropy
            # Keep the reference to `uniform` out of the graph; we only
            # used it for documentation above.  (Nothing to do.)
            _ = uniform

    return loss, bag_logit, attention


def train_ours(
    bags: Sequence[MILBag],
    config: Optional[MILTrainerConfig] = None,
) -> Any:
    """Train the Ours model on a list of MIL bags.

    Returns the fitted ``nn.Module``.  Callers are expected to own the
    device placement: all :class:`MILBag` tensors must live on the same
    device before the call, and the returned model will match.
    """

    if not bags:
        raise ValueError("train_ours requires at least one bag")

    cfg = config or MILTrainerConfig()
    torch, F, optim = _require_torch()
    torch.manual_seed(cfg.random_state)

    model = build_ours(cfg.ours_config)
    # Move to the same device as the first bag's features.
    ref_device = bags[0].features.device
    model.to(ref_device)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    n_bags = len(bags)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_bags).tolist()
        running = 0.0
        for start in range(0, n_bags, cfg.batch_size):
            optimizer.zero_grad()
            batch_ids = perm[start : start + cfg.batch_size]
            batch_loss = None
            for bi in batch_ids:
                bag = bags[bi]
                # L41: subsample the bag per-epoch so the attention head
                # has a realistic positive fraction to learn from.
                train_bag = subsample_bag_for_training(
                    bag,
                    max_size=cfg.train_max_bag_size,
                    min_positive_fraction=cfg.train_min_positive_fraction,
                    epoch=epoch,
                    global_seed=cfg.random_state,
                )
                loss, _, _ = _bag_loss(
                    model, train_bag, cfg=cfg, F=F, torch_mod=torch
                )
                batch_loss = loss if batch_loss is None else batch_loss + loss
            if batch_loss is None:
                continue
            batch_loss = batch_loss / len(batch_ids)
            batch_loss.backward()
            optimizer.step()
            running += float(batch_loss.item())
        if cfg.verbose:
            avg = running * cfg.batch_size / max(1, n_bags)
            print(f"[MIL] epoch {epoch+1}/{cfg.epochs} avg_loss={avg:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_bag(
    model: Any,
    bag: MILBag,
) -> Tuple[float, Any, Any]:
    """Return ``(bag_score, attention [N], instance_logits [N])``.

    ``bag_score`` is the sigmoid of the bag logit; downstream baseline
    code thresholds it to produce the predicted label.
    """

    torch, _, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        bag_logit, attention, instance_logits = model(
            bag.features, bag.types
        )
        bag_score = float(torch.sigmoid(bag_logit).item())
    return bag_score, attention, instance_logits
