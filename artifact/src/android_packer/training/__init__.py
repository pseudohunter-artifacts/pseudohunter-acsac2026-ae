"""Training orchestration for the Ours (Typed-Instance MIL) method.

This package lands as batch **F-MIL-c / F-MIL-d / F-MIL-e** per
``docs/method/ours_method_spec.md`` §12.6 and hosts:

* :mod:`android_packer.training.pretrain_mlm` — byte-level MLM + grammar-aware
  auxiliary loss (``L_mlm + λ_item · L_item_type``) on a **benign-only**
  corpus, using :func:`android_packer.features.dex_item_parser.parse_dex_item_spans`
  as a hard filter so packed / truncated DEX never contaminates pretraining.
* :mod:`android_packer.training.contrastive` — packed / unpacked contrastive
  pretraining on Track B v2 pairs (``F-MIL-d``).
* :mod:`android_packer.training.mil_trainer` — supervised MIL fine-tuning
  (``F-MIL-e``).

Lazy-torch contract (§3.1 of the method spec): **no** top-level ``import torch``
in this package.  Individual modules pull torch in at training-entry time so
``from android_packer.training import ...`` stays zero-cost in the stdlib
pipeline.
"""

from __future__ import annotations

from android_packer.training.contrastive import (
    ContrastiveConfig,
    ContrastivePairBatch,
    build_pair_batch,
    compute_contrastive_loss,
    info_nce_app_head,
    info_nce_pack_residual,
)
from android_packer.training.pretrain_mlm import (
    MLMCollator,
    MLMBatch,
    MLMCorpusBuilder,
    MLMCorpusStats,
    MLMExample,
    PretrainMLMConfig,
    build_mlm_example,
    build_mlm_corpus,
)

__all__ = [
    "ContrastiveConfig",
    "ContrastivePairBatch",
    "MLMBatch",
    "MLMCollator",
    "MLMCorpusBuilder",
    "MLMCorpusStats",
    "MLMExample",
    "PretrainMLMConfig",
    "build_mlm_example",
    "build_mlm_corpus",
    "build_pair_batch",
    "compute_contrastive_loss",
    "info_nce_app_head",
    "info_nce_pack_residual",
]
