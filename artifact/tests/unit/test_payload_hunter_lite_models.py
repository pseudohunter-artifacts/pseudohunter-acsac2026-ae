"""Smoke tests for PayloadHunter-Lite model skeletons.

Covers just enough of the contract in
:mod:`docs/method/ours_method_spec.md` §11.3 to catch regressions:

* Configs import without torch (stdlib-only).
* Factory instantiation returns a real ``nn.Module``.
* Forward pass produces expected shapes.
* Attention weights sum to 1 and are non-negative.

Heavier training / convergence tests are deferred to F-Lite-b where
the feature assembly lands.
"""

from __future__ import annotations

import pytest

from android_packer.models import (
    LiteObjectAggregatorConfig,
    LiteRegionScorerConfig,
)


def test_configs_importable_without_torch():
    """The dataclass configs must not require torch at import time.

    We cannot easily simulate "no torch" in the test env (torch is in
    the [dl] extra and may be installed), but we can at least verify
    that constructing the configs and reading their fields works
    without touching the heavy factories.
    """

    region_cfg = LiteRegionScorerConfig(feature_dim=8, hidden_dim=16)
    assert region_cfg.feature_dim == 8
    assert region_cfg.hidden_dim == 16
    assert region_cfg.dropout == pytest.approx(0.2)

    object_cfg = LiteObjectAggregatorConfig(input_dim=8, attn_hidden_dim=16)
    assert object_cfg.input_dim == 8
    assert object_cfg.return_attention is True


def test_config_validation_rejects_bad_values():
    from android_packer.models.payload_hunter_lite import (
        build_lite_object_aggregator,
        build_lite_region_scorer,
    )

    pytest.importorskip("torch")

    with pytest.raises(ValueError):
        build_lite_region_scorer(LiteRegionScorerConfig(feature_dim=0))
    with pytest.raises(ValueError):
        build_lite_region_scorer(LiteRegionScorerConfig(dropout=1.5))
    with pytest.raises(ValueError):
        build_lite_object_aggregator(
            LiteObjectAggregatorConfig(input_dim=-1)
        )


def test_region_scorer_forward_shape():
    torch = pytest.importorskip("torch")
    from android_packer.models.payload_hunter_lite import build_lite_region_scorer

    cfg = LiteRegionScorerConfig(
        feature_dim=8, hidden_dim=16, num_hidden_layers=2
    )
    scorer = build_lite_region_scorer(cfg)
    x = torch.randn(4, 8)
    y = scorer(x)
    # Sequential last layer is Linear(_, 1) so output is [batch, 1].
    assert y.shape == (4, 1)


def test_object_aggregator_returns_attention_weights_summing_to_one():
    torch = pytest.importorskip("torch")
    from android_packer.models.payload_hunter_lite import (
        build_lite_object_aggregator,
    )

    cfg = LiteObjectAggregatorConfig(
        input_dim=8, attn_hidden_dim=16, return_attention=True
    )
    aggregator = build_lite_object_aggregator(cfg)

    n = 5
    features = torch.randn(n, 8)
    logits = torch.randn(n)

    # Disable autograd so ``.min()`` / ``.sum()`` can be unwrapped as
    # Python floats without triggering the "requires_grad scalar"
    # UserWarning from torch.
    with torch.no_grad():
        object_logit, attn_weights = aggregator(features, logits)

    # scalar logit
    assert object_logit.dim() == 0
    # attention weights: non-negative, shape [N], sum to 1
    assert attn_weights.shape == (n,)
    assert float(attn_weights.min()) >= 0.0
    assert float(attn_weights.sum()) == pytest.approx(1.0, abs=1e-5)


def test_object_aggregator_respects_return_attention_flag():
    torch = pytest.importorskip("torch")
    from android_packer.models.payload_hunter_lite import (
        build_lite_object_aggregator,
    )

    cfg = LiteObjectAggregatorConfig(
        input_dim=4, attn_hidden_dim=8, return_attention=False
    )
    aggregator = build_lite_object_aggregator(cfg)

    features = torch.randn(3, 4)
    logits = torch.randn(3)

    out = aggregator(features, logits)
    # With return_attention=False we should get a single scalar back,
    # not a tuple.
    assert torch.is_tensor(out)
    assert out.dim() == 0


def test_object_aggregator_rejects_shape_mismatch():
    torch = pytest.importorskip("torch")
    from android_packer.models.payload_hunter_lite import (
        build_lite_object_aggregator,
    )

    cfg = LiteObjectAggregatorConfig(input_dim=4, attn_hidden_dim=8)
    aggregator = build_lite_object_aggregator(cfg)

    # Wrong feature dim.
    with pytest.raises(ValueError):
        aggregator(torch.randn(3, 2), torch.randn(3))

    # Mismatched N between features and logits.
    with pytest.raises(ValueError):
        aggregator(torch.randn(3, 4), torch.randn(4))
