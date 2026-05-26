"""Region / object / bag encoders and classifiers.

The Ours (Stage A main) model is ``build_ours`` in
:mod:`android_packer.models.ours` — a **Typed-Instance Multiple-Instance
Learning** architecture composed of :mod:`.typed_encoder` +
:mod:`.mil_head` (and optionally :mod:`.item_type_head` for grammar-aware
auxiliary pre-training).

PayloadHunter-Lite (``build_lite_region_scorer`` / ``build_lite_object_aggregator``)
is retained as a **first-class ablation baseline** — it is the "no MIL,
no typed routing, no grammar aux" corner of the ablation grid used in
§12.5 of the method spec.

All exports here are lazy w.r.t. torch: importing this package does
**not** require ``torch``.  Only the factory functions will trigger a
``_require_torch()`` call when invoked.
"""

from android_packer.models.byte_cnn import (
    ByteCnnRegionScorerConfig,
    build_byte_cnn_region_scorer,
)
from android_packer.models.item_type_head import (
    DEX_ITEM_TYPE_PAD_ID,
    ItemTypeHeadConfig,
    build_item_type_head,
)
from android_packer.models.mil_head import (
    AttentionPoolingConfig,
    MILPoolingKind,
    NoisyOrPoolingConfig,
    TopKPoolingConfig,
    build_attention_pooling,
    build_mil_pooling,
    build_noisy_or_pooling,
    build_topk_pooling,
)
from android_packer.models.ours import OursConfig, build_ours
from android_packer.models.payload_hunter_lite import (
    LiteObjectAggregatorConfig,
    LiteRegionScorerConfig,
    build_lite_object_aggregator,
    build_lite_region_scorer,
)
from android_packer.models.tokenizer import ByteTokenizer, ByteTokenizerEncoding
from android_packer.models.typed_encoder import (
    N_TYPED_INSTANCE_TYPES,
    TYPED_INSTANCE_TYPES,
    TypedEncoderConfig,
    build_typed_encoder,
    instance_type_id,
)

__all__ = [
    # Tokenizer
    "ByteTokenizer",
    "ByteTokenizerEncoding",
    # PayloadHunter-Lite (ablation baseline)
    "ByteCnnRegionScorerConfig",
    "build_byte_cnn_region_scorer",
    "LiteObjectAggregatorConfig",
    "LiteRegionScorerConfig",
    "build_lite_object_aggregator",
    "build_lite_region_scorer",
    # Typed-Instance MIL (Ours main)
    "TYPED_INSTANCE_TYPES",
    "N_TYPED_INSTANCE_TYPES",
    "TypedEncoderConfig",
    "build_typed_encoder",
    "instance_type_id",
    "TopKPoolingConfig",
    "NoisyOrPoolingConfig",
    "AttentionPoolingConfig",
    "MILPoolingKind",
    "build_topk_pooling",
    "build_noisy_or_pooling",
    "build_attention_pooling",
    "build_mil_pooling",
    "OursConfig",
    "build_ours",
    # Grammar-aware auxiliary head
    "DEX_ITEM_TYPE_PAD_ID",
    "ItemTypeHeadConfig",
    "build_item_type_head",
]
