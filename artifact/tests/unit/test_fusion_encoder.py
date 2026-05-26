from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requires [dl] extra")

from android_packer.features.full_feature_extractor import SCALAR_FEATURE_DIM
from android_packer.models.fusion_encoder import FusionEncoderConfig, build_fusion_encoder
from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES


def _tiny_config(**overrides) -> FusionEncoderConfig:
    values = {
        "bert_hidden_dim": 24,
        "bert_n_layers": 1,
        "bert_n_heads": 4,
        "bert_intermediate_dim": 48,
        "bert_max_length": 8,
        "stat_dim": SCALAR_FEATURE_DIM,
        "fusion_hidden_dim": 32,
        "output_dim": 16,
        "fusion_dropout": 0.0,
        "bert_dropout": 0.0,
    }
    values.update(overrides)
    return FusionEncoderConfig(**values)


def test_bert_only_path_aggregation_has_no_stat_projection() -> None:
    model = build_fusion_encoder(
        _tiny_config(use_gated_fusion=True, use_stat_features=False)
    )

    assert not hasattr(model, "stat_proj")
    assert not hasattr(model, "gate_logits")

    batch, length = 2, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.randn((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)

    embeddings, suspicion, normality = model(
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        stat,
        active_paths=("byte",),
    )

    assert embeddings.shape == (batch, 16)
    assert suspicion.shape == (batch,)
    assert normality.shape == (batch,)


def test_stat_only_branch_has_no_bert_aggregation_or_gate() -> None:
    model = build_fusion_encoder(
        _tiny_config(
            use_gated_fusion=True,
            use_bert_features=False,
            use_stat_features=True,
        )
    )

    assert hasattr(model, "stat_proj")
    assert not hasattr(model, "bert_aggregation")
    assert not hasattr(model, "gate_logits")

    batch, length = 2, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.randn((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)

    embeddings, suspicion, normality = model(
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        stat,
    )

    assert embeddings.shape == (batch, 16)
    assert suspicion.shape == (batch,)
    assert normality.shape == (batch,)


def test_active_paths_rejects_unknown_path() -> None:
    model = build_fusion_encoder(_tiny_config(use_stat_features=False))
    batch, length = 1, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown active path"):
        model(
            token_ids,
            token_types,
            mask,
            token_ids,
            token_types,
            mask,
            token_ids,
            token_types,
            mask,
            stat,
            active_paths=("dex",),
        )


def test_region_type_routing_requires_entry_type_ids() -> None:
    model = build_fusion_encoder(
        _tiny_config(use_stat_features=False, use_region_type_routing=True)
    )
    batch, length = 1, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)

    with pytest.raises(ValueError, match="entry_type_ids are required"):
        model(
            token_ids,
            token_types,
            mask,
            token_ids,
            token_types,
            mask,
            token_ids,
            token_types,
            mask,
            stat,
            active_paths=("dalvik", "byte"),
        )


def test_region_type_routing_accepts_entry_type_ids() -> None:
    model = build_fusion_encoder(
        _tiny_config(use_stat_features=False, use_region_type_routing=True)
    )
    batch, length = 3, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)
    entry_type_ids = torch.tensor(
        [
            ENTRY_COARSE_TYPES.index("dex"),
            ENTRY_COARSE_TYPES.index("elf"),
            ENTRY_COARSE_TYPES.index("asset"),
        ],
        dtype=torch.long,
    )

    embeddings, suspicion, normality = model(
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        stat,
        active_paths=("dalvik", "arm64", "byte"),
        entry_type_ids=entry_type_ids,
    )

    assert embeddings.shape == (batch, 16)
    assert suspicion.shape == (batch,)
    assert normality.shape == (batch,)


def test_region_type_routing_weights_are_configurable() -> None:
    model = build_fusion_encoder(
        _tiny_config(
            use_stat_features=False,
            use_region_type_routing=True,
            routing_dex_byte_weight=0.05,
            routing_elf_byte_weight=0.10,
            routing_byte_entry_weight=0.25,
            routing_unknown_weight=0.05,
        )
    )

    assert model.cfg.routing_dex_byte_weight == pytest.approx(0.05)
    assert model.cfg.routing_elf_byte_weight == pytest.approx(0.10)
    assert model.cfg.routing_byte_entry_weight == pytest.approx(0.25)
    assert model.cfg.routing_unknown_weight == pytest.approx(0.05)

    batch, length = 4, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)
    entry_type_ids = torch.tensor(
        [
            ENTRY_COARSE_TYPES.index("dex"),
            ENTRY_COARSE_TYPES.index("elf"),
            ENTRY_COARSE_TYPES.index("asset"),
            ENTRY_COARSE_TYPES.index("unknown"),
        ],
        dtype=torch.long,
    )

    embeddings, suspicion, normality = model(
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        stat,
        active_paths=("dalvik", "arm64", "byte"),
        entry_type_ids=entry_type_ids,
    )

    assert embeddings.shape == (batch, 16)
    assert suspicion.shape == (batch,)
    assert normality.shape == (batch,)


def test_path_dropout_keeps_at_least_one_path() -> None:
    model = build_fusion_encoder(
        _tiny_config(use_stat_features=False, path_dropout_prob=0.99)
    )
    model.train()
    batch, length = 2, 8
    token_ids = torch.zeros((batch, length), dtype=torch.long)
    token_types = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)

    embeddings, suspicion, normality = model(
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        token_ids,
        token_types,
        mask,
        stat,
        active_paths=("dalvik", "arm64", "byte"),
    )

    assert embeddings.shape == (batch, 16)
    assert suspicion.shape == (batch,)
    assert normality.shape == (batch,)


def test_batched_bert_streams_matches_legacy_stream_forwards() -> None:
    cfg_batched = _tiny_config(
        use_stat_features=False,
        use_region_type_routing=True,
        batch_bert_streams=True,
    )
    cfg_legacy = _tiny_config(
        use_stat_features=False,
        use_region_type_routing=True,
        batch_bert_streams=False,
    )
    batched = build_fusion_encoder(cfg_batched)
    legacy = build_fusion_encoder(cfg_legacy)
    legacy.load_state_dict(batched.state_dict())
    batched.eval()
    legacy.eval()

    batch, length = 4, 8
    dalvik_ids = torch.randint(0, 20, (batch, length), dtype=torch.long)
    native_ids = torch.randint(0, 20, (batch, length), dtype=torch.long)
    byte_ids = torch.randint(0, 20, (batch, length), dtype=torch.long)
    dalvik_types = torch.zeros((batch, length), dtype=torch.long)
    native_types = torch.ones((batch, length), dtype=torch.long)
    byte_types = torch.full((batch, length), 2, dtype=torch.long)
    mask = torch.ones((batch, length), dtype=torch.float32)
    stat = torch.zeros((batch, SCALAR_FEATURE_DIM), dtype=torch.float32)
    entry_type_ids = torch.tensor(
        [
            ENTRY_COARSE_TYPES.index("dex"),
            ENTRY_COARSE_TYPES.index("elf"),
            ENTRY_COARSE_TYPES.index("asset"),
            ENTRY_COARSE_TYPES.index("unknown"),
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out_batched = batched(
            dalvik_ids,
            dalvik_types,
            mask,
            native_ids,
            native_types,
            mask,
            byte_ids,
            byte_types,
            mask,
            stat,
            active_paths=("dalvik", "arm64", "byte"),
            entry_type_ids=entry_type_ids,
        )
        out_legacy = legacy(
            dalvik_ids,
            dalvik_types,
            mask,
            native_ids,
            native_types,
            mask,
            byte_ids,
            byte_types,
            mask,
            stat,
            active_paths=("dalvik", "arm64", "byte"),
            entry_type_ids=entry_type_ids,
        )

    for got, expected in zip(out_batched, out_legacy):
        assert torch.allclose(got, expected, atol=1e-6)
