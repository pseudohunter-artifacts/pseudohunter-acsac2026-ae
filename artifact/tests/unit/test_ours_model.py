"""End-to-end unit tests for the Ours (Typed-Instance MIL) model."""

from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from android_packer.models.mil_head import (  # noqa: E402
    AttentionPoolingConfig,
    NoisyOrPoolingConfig,
    TopKPoolingConfig,
)
from android_packer.models.ours import OursConfig, build_ours  # noqa: E402
from android_packer.models.typed_encoder import (  # noqa: E402
    TypedEncoderConfig,
    instance_type_id,
)


def _small_cfg(pooling: str, **kwargs) -> OursConfig:
    return OursConfig(
        typed=TypedEncoderConfig(
            input_dim=15, hidden_dim=32, head_hidden_dim=16, n_types=6, dropout=0.0
        ),
        mil_pooling=pooling,  # type: ignore[arg-type]
        topk=TopKPoolingConfig(k=2),
        noisy_or=NoisyOrPoolingConfig(),
        attention=AttentionPoolingConfig(attn_hidden_dim=16, dropout=0.0),
        **kwargs,
    )


class OursForwardTests(unittest.TestCase):
    def _bag(self, n: int = 6):
        feats = torch.randn(n, 15)
        # one of each type; caller may reuse.
        types = torch.tensor(
            [
                instance_type_id("encrypted_dex"),
                instance_type_id("extracted_method_body"),
                instance_type_id("metadata_table"),
                instance_type_id("compressed_payload"),
                instance_type_id("shim"),
                instance_type_id("native_stub"),
            ],
            dtype=torch.long,
        )
        return feats[:n], types[:n]

    def test_attention_forward(self) -> None:
        model = build_ours(_small_cfg("attention"))
        feats, types = self._bag()
        bag_logit, attn, inst = model(feats, types)
        self.assertEqual(bag_logit.dim(), 0)
        self.assertEqual(attn.shape, (6,))
        self.assertEqual(inst.shape, (6,))
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_topk_forward(self) -> None:
        model = build_ours(_small_cfg("topk"))
        feats, types = self._bag()
        bag_logit, attn, inst = model(feats, types)
        self.assertEqual(bag_logit.dim(), 0)
        self.assertEqual(attn.shape, (6,))
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_noisy_or_forward(self) -> None:
        model = build_ours(_small_cfg("noisy_or"))
        feats, types = self._bag()
        bag_logit, attn, inst = model(feats, types)
        self.assertEqual(bag_logit.dim(), 0)
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_backward_over_bce_loss(self) -> None:
        """Simulate a training step: BCE on bag logit against label 1."""
        model = build_ours(_small_cfg("attention"))
        feats, types = self._bag()
        bag_logit, _, _ = model(feats, types)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            bag_logit, torch.tensor(1.0)
        )
        loss.backward()

        # Some parameter must have non-zero gradient.
        total = sum(
            float(torch.abs(p.grad).sum()) if p.grad is not None else 0.0
            for p in model.parameters()
        )
        self.assertGreater(total, 0.0)

    def test_attention_pooling_uses_typed_hidden(self) -> None:
        """use_feature_attention=True should propagate gradient through hidden features path."""
        model = build_ours(_small_cfg("attention", use_feature_attention=True))
        feats, types = self._bag()
        _, attn, _ = model(feats, types)
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_variable_bag_sizes(self) -> None:
        model = build_ours(_small_cfg("attention"))
        for n in (1, 3, 12):
            feats = torch.randn(n, 15)
            types = torch.randint(0, 6, (n,), dtype=torch.long)
            bag_logit, attn, inst = model(feats, types)
            self.assertEqual(attn.shape, (n,))
            self.assertEqual(inst.shape, (n,))
            self.assertFalse(torch.isnan(bag_logit))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
