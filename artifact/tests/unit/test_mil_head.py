"""Unit tests for the Typed-Instance MIL pooling heads.

Covers the three pooling strategies declared in
``docs/method/ours_method_spec.md`` §12:
* top-k
* noisy-or
* gated attention (ABMIL)

Follows the dependency-gating convention from AGENTS.md §4:
``pytest.importorskip("torch")`` at module level so the core zero-dep
test suite stays green without ``[dl]`` installed.
"""

from __future__ import annotations

import math
import unittest

import pytest

torch = pytest.importorskip("torch")

from android_packer.models.mil_head import (  # noqa: E402
    AttentionPoolingConfig,
    NoisyOrPoolingConfig,
    TopKPoolingConfig,
    build_attention_pooling,
    build_mil_pooling,
    build_noisy_or_pooling,
    build_topk_pooling,
)


class TopKPoolingTests(unittest.TestCase):
    def test_topk_averages_top_scores(self) -> None:
        head = build_topk_pooling(TopKPoolingConfig(k=2))
        logits = torch.tensor([0.1, 3.0, -1.0, 2.0, 0.5])
        bag_logit, attn = head(logits)

        # top-2 = [3.0, 2.0] → mean = 2.5
        self.assertAlmostEqual(float(bag_logit), 2.5, places=5)
        self.assertEqual(attn.shape, (5,))
        # attention on indices 1 and 3 should each be 1/2, others 0.
        self.assertAlmostEqual(float(attn[1]), 0.5, places=5)
        self.assertAlmostEqual(float(attn[3]), 0.5, places=5)
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_topk_gradient_flows(self) -> None:
        head = build_topk_pooling(TopKPoolingConfig(k=2))
        logits = torch.tensor([0.1, 3.0, -1.0, 2.0, 0.5], requires_grad=True)
        bag_logit, _ = head(logits)
        bag_logit.backward()
        assert logits.grad is not None
        # Only the selected indices should have non-zero gradient.
        self.assertAlmostEqual(float(logits.grad[1]), 0.5, places=5)
        self.assertAlmostEqual(float(logits.grad[3]), 0.5, places=5)
        self.assertAlmostEqual(float(logits.grad[0]), 0.0, places=5)

    def test_topk_k_larger_than_n_is_clamped(self) -> None:
        head = build_topk_pooling(TopKPoolingConfig(k=10))
        logits = torch.tensor([1.0, 2.0, 3.0])
        bag_logit, attn = head(logits)
        self.assertAlmostEqual(float(bag_logit), 2.0, places=5)  # mean
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_topk_ratio_override(self) -> None:
        head = build_topk_pooling(TopKPoolingConfig(k=1, k_ratio=0.5))
        logits = torch.tensor([0.0, 4.0, 1.0, 3.0])  # N=4, ratio 0.5 → k=2
        bag_logit, _ = head(logits)
        self.assertAlmostEqual(float(bag_logit), 3.5, places=5)  # mean(4,3)

    def test_topk_empty_bag(self) -> None:
        head = build_topk_pooling(TopKPoolingConfig(k=1))
        logits = torch.zeros((0,))
        bag_logit, attn = head(logits)
        self.assertEqual(attn.shape, (0,))
        self.assertEqual(float(bag_logit), 0.0)


class NoisyOrPoolingTests(unittest.TestCase):
    def test_noisy_or_monotone_in_positives(self) -> None:
        head = build_noisy_or_pooling()
        small = torch.tensor([-5.0, -5.0, -5.0])
        large = torch.tensor([-5.0, -5.0, 5.0])
        bag_small, _ = head(small)
        bag_large, _ = head(large)
        self.assertLess(float(bag_small), float(bag_large))

    def test_noisy_or_matches_analytic_single_instance(self) -> None:
        head = build_noisy_or_pooling()
        logit = torch.tensor([1.2])
        bag_logit, _ = head(logit)
        # For a single instance, logit(1 - (1 - sigmoid(z))) = z exactly.
        self.assertAlmostEqual(float(bag_logit), 1.2, places=4)

    def test_noisy_or_attention_sums_to_one(self) -> None:
        head = build_noisy_or_pooling()
        logits = torch.tensor([-2.0, 0.0, 1.5, 3.0])
        _, attn = head(logits)
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)
        # Highest-logit instance should have highest attention weight.
        self.assertEqual(int(torch.argmax(attn)), 3)

    def test_noisy_or_gradient_flows(self) -> None:
        head = build_noisy_or_pooling()
        logits = torch.tensor([0.5, -0.2, 1.3], requires_grad=True)
        bag_logit, _ = head(logits)
        bag_logit.backward()
        assert logits.grad is not None
        # Every instance should contribute non-zero gradient under noisy-or.
        for i in range(3):
            self.assertGreater(abs(float(logits.grad[i])), 0.0)


class AttentionPoolingTests(unittest.TestCase):
    def test_attention_output_shapes(self) -> None:
        head = build_attention_pooling(
            AttentionPoolingConfig(feature_dim=8, attn_hidden_dim=16)
        )
        logits = torch.randn(5)
        feats = torch.randn(5, 8)
        bag_logit, attn = head(logits, instance_features=feats)
        self.assertEqual(bag_logit.dim(), 0)
        self.assertEqual(attn.shape, (5,))
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)

    def test_attention_logit_only_mode(self) -> None:
        # No feature_dim → degenerates to attention on the logit scalar.
        head = build_attention_pooling(
            AttentionPoolingConfig(feature_dim=None, attn_hidden_dim=8)
        )
        logits = torch.randn(4)
        bag_logit, attn = head(logits)
        self.assertEqual(attn.shape, (4,))
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=5)
        self.assertFalse(math.isnan(float(bag_logit)))

    def test_attention_gradient_flows_through_features(self) -> None:
        head = build_attention_pooling(
            AttentionPoolingConfig(feature_dim=4, attn_hidden_dim=8)
        )
        logits = torch.randn(3, requires_grad=True)
        feats = torch.randn(3, 4, requires_grad=True)
        bag_logit, _ = head(logits, instance_features=feats)
        bag_logit.backward()
        assert logits.grad is not None
        assert feats.grad is not None
        self.assertGreater(float(torch.abs(feats.grad).sum()), 0.0)

    def test_attention_rejects_mismatched_feature_dim(self) -> None:
        head = build_attention_pooling(
            AttentionPoolingConfig(feature_dim=4, attn_hidden_dim=8)
        )
        logits = torch.randn(3)
        bad_feats = torch.randn(3, 5)
        with self.assertRaises(ValueError):
            head(logits, instance_features=bad_feats)


class DispatchBuilderTests(unittest.TestCase):
    def test_build_mil_pooling_dispatches(self) -> None:
        topk = build_mil_pooling("topk", topk=TopKPoolingConfig(k=1))
        noisy = build_mil_pooling("noisy_or", noisy_or=NoisyOrPoolingConfig())
        attn = build_mil_pooling(
            "attention", attention=AttentionPoolingConfig(feature_dim=None)
        )
        logits = torch.randn(5)
        for head in (topk, noisy, attn):
            bag_logit, a = head(logits)
            self.assertEqual(bag_logit.dim(), 0)
            self.assertEqual(a.shape, (5,))

    def test_build_mil_pooling_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_mil_pooling("bogus")  # type: ignore[arg-type]


class ConfigValidationTests(unittest.TestCase):
    def test_topk_rejects_zero_k_and_zero_ratio(self) -> None:
        with self.assertRaises(ValueError):
            build_topk_pooling(TopKPoolingConfig(k=0, k_ratio=0.0))

    def test_topk_rejects_out_of_range_ratio(self) -> None:
        with self.assertRaises(ValueError):
            build_topk_pooling(TopKPoolingConfig(k=1, k_ratio=1.5))

    def test_noisy_or_rejects_bad_eps(self) -> None:
        with self.assertRaises(ValueError):
            build_noisy_or_pooling(NoisyOrPoolingConfig(eps=0.0))

    def test_attention_rejects_bad_hidden(self) -> None:
        with self.assertRaises(ValueError):
            build_attention_pooling(AttentionPoolingConfig(attn_hidden_dim=0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
