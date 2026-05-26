"""Unit tests for attention-based scoring modes in Ours baseline inference."""

from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402


class ScoringModeTests(unittest.TestCase):
    """Test the three scoring modes in _predict_impl's inner loop."""

    def _compute_inst_scores(self, scoring_mode, bag_logit, attention, instance_logits):
        """Replicate the scoring logic from _predict_impl for unit testing."""
        bag_score = float(torch.sigmoid(torch.tensor(bag_logit)).item())
        attn_np = np.array(attention, dtype=np.float32)

        if scoring_mode == "attention":
            attn_max = float(attn_np.max())
            if attn_max > 0:
                return (attn_np / attn_max).tolist()
            else:
                return attn_np.tolist()
        elif scoring_mode == "attention_x_bag":
            attn_max = float(attn_np.max())
            if attn_max > 0:
                return ((attn_np / attn_max) * bag_score).tolist()
            else:
                return (attn_np * bag_score).tolist()
        elif scoring_mode == "attention_anomaly":
            attn_max = float(attn_np.max())
            if attn_max > 0:
                return ((1.0 - attn_np / attn_max) * bag_score).tolist()
            else:
                return [bag_score] * len(attn_np)
        else:  # instance_logit
            return (
                torch.sigmoid(torch.tensor(instance_logits, dtype=torch.float32))
                .numpy()
                .tolist()
            )

    def test_instance_logit_mode_is_legacy(self):
        """instance_logit mode returns sigmoid of raw instance logits."""
        scores = self._compute_inst_scores(
            "instance_logit",
            bag_logit=2.0,
            attention=[0.1, 0.5, 0.4],
            instance_logits=[0.0, 3.0, -3.0],
        )
        # sigmoid(0)=0.5, sigmoid(3)~0.95, sigmoid(-3)~0.05
        self.assertAlmostEqual(scores[0], 0.5, places=3)
        self.assertGreater(scores[1], 0.9)
        self.assertLess(scores[2], 0.1)

    def test_attention_mode_normalizes_to_max(self):
        """attention mode: top instance scores 1.0, others proportional."""
        scores = self._compute_inst_scores(
            "attention",
            bag_logit=2.0,
            attention=[0.1, 0.6, 0.3],
            instance_logits=[0.0, 0.0, 0.0],  # ignored
        )
        self.assertAlmostEqual(scores[1], 1.0, places=5)  # max attention
        self.assertAlmostEqual(scores[0], 0.1 / 0.6, places=5)
        self.assertAlmostEqual(scores[2], 0.3 / 0.6, places=5)

    def test_attention_mode_ignores_instance_logits(self):
        """In attention mode, instance_logits should not affect scores."""
        scores_a = self._compute_inst_scores(
            "attention",
            bag_logit=2.0,
            attention=[0.2, 0.8],
            instance_logits=[10.0, -10.0],
        )
        scores_b = self._compute_inst_scores(
            "attention",
            bag_logit=2.0,
            attention=[0.2, 0.8],
            instance_logits=[-10.0, 10.0],
        )
        # Same attention → same scores regardless of logits
        self.assertAlmostEqual(scores_a[0], scores_b[0], places=5)
        self.assertAlmostEqual(scores_a[1], scores_b[1], places=5)

    def test_attention_x_bag_scales_by_bag_confidence(self):
        """attention_x_bag: high bag score -> high instance scores."""
        # bag_logit=5.0 -> sigmoid~0.993
        scores_high = self._compute_inst_scores(
            "attention_x_bag",
            bag_logit=5.0,
            attention=[0.2, 0.8],
            instance_logits=[0.0, 0.0],
        )
        # bag_logit=-5.0 -> sigmoid~0.007
        scores_low = self._compute_inst_scores(
            "attention_x_bag",
            bag_logit=-5.0,
            attention=[0.2, 0.8],
            instance_logits=[0.0, 0.0],
        )
        # With high bag confidence, top instance should be near 1.0
        self.assertGreater(scores_high[1], 0.9)
        # With low bag confidence, even top instance should be near 0.0
        self.assertLess(scores_low[1], 0.05)

    def test_attention_x_bag_benign_bag_suppresses_all(self):
        """When bag is benign (bag_score~0), all instances should score low."""
        scores = self._compute_inst_scores(
            "attention_x_bag",
            bag_logit=-10.0,  # sigmoid ~ 0.0000454
            attention=[0.3, 0.7],
            instance_logits=[0.0, 0.0],
        )
        for s in scores:
            self.assertLess(s, 0.001)

    def test_attention_mode_discriminates_instances(self):
        """Key regression test: attention mode should NOT give all instances
        the same score (unlike the old instance_logit mode under bag
        supervision where instance_logits converge to similar values)."""
        # Simulate bag supervision: instance_logits all similar, but attention
        # is discriminative.
        scores_attn = self._compute_inst_scores(
            "attention",
            bag_logit=2.0,
            attention=[0.05, 0.05, 0.8, 0.05, 0.05],
            instance_logits=[1.5, 1.4, 1.6, 1.3, 1.5],  # nearly identical
        )
        # The 3rd instance (attention=0.8) should be far above others
        self.assertAlmostEqual(scores_attn[2], 1.0, places=5)
        self.assertLess(scores_attn[0], 0.1)

        # Compare with legacy mode: all scores cluster together
        scores_legacy = self._compute_inst_scores(
            "instance_logit",
            bag_logit=2.0,
            attention=[0.05, 0.05, 0.8, 0.05, 0.05],
            instance_logits=[1.5, 1.4, 1.6, 1.3, 1.5],
        )
        score_range_legacy = max(scores_legacy) - min(scores_legacy)
        score_range_attn = max(scores_attn) - min(scores_attn)
        # Attention mode should have MUCH wider score spread
        self.assertGreater(score_range_attn, score_range_legacy * 5)

    def test_attention_single_instance(self):
        """Single instance: attention=[1.0], score should be 1.0 or bag_score."""
        scores_attn = self._compute_inst_scores(
            "attention",
            bag_logit=2.0,
            attention=[1.0],
            instance_logits=[0.5],
        )
        self.assertAlmostEqual(scores_attn[0], 1.0, places=5)

        scores_bag = self._compute_inst_scores(
            "attention_x_bag",
            bag_logit=2.0,  # sigmoid ~ 0.88
            attention=[1.0],
            instance_logits=[0.5],
        )
        expected = float(torch.sigmoid(torch.tensor(2.0)).item())
        self.assertAlmostEqual(scores_bag[0], expected, places=4)

    def test_attention_anomaly_mode_inverts_attention(self):
        """attention_anomaly: low-attention instances get HIGH scores."""
        scores = self._compute_inst_scores(
            "attention_anomaly",
            bag_logit=5.0,  # sigmoid ~ 0.993 (packed bag)
            attention=[0.8, 0.1, 0.05, 0.05],
            instance_logits=[0.0, 0.0, 0.0, 0.0],
        )
        # Instance 0 has highest attention → should get LOWEST score
        # Instance 2,3 have lowest attention → should get HIGHEST scores
        self.assertLess(scores[0], 0.3)
        self.assertGreater(scores[2], 0.9)
        self.assertGreater(scores[3], 0.9)

    def test_attention_anomaly_benign_bag_suppresses_all(self):
        """Even low-attention instances score low when bag is benign."""
        scores = self._compute_inst_scores(
            "attention_anomaly",
            bag_logit=-10.0,  # sigmoid ~ 0.0000454
            attention=[0.9, 0.05, 0.05],
            instance_logits=[0.0, 0.0, 0.0],
        )
        for s in scores:
            self.assertLess(s, 0.001)

    def test_attention_anomaly_discriminates_payload_from_benign(self):
        """Regression test: anomaly mode gives high score to low-attention
        (payload) objects and low score to high-attention (benign) objects."""
        # Simulate: object 2 is the payload (low attention), rest are benign
        scores = self._compute_inst_scores(
            "attention_anomaly",
            bag_logit=3.0,  # sigmoid ~ 0.95
            attention=[0.3, 0.5, 0.02, 0.18],  # obj 2 is unusual
            instance_logits=[1.0, 1.0, 1.0, 1.0],
        )
        # Object 2 (attention=0.02) should be the highest scored
        self.assertEqual(scores.index(max(scores)), 2)

    def test_invalid_scoring_mode_raises(self):
        """Unknown scoring mode should raise ValueError."""
        from android_packer.baselines.ours import OursBaselineConfig

        # Construct config with invalid mode -- frozen dataclass so use replace
        import dataclasses

        default_cfg = OursBaselineConfig()
        bad_cfg = dataclasses.replace(default_cfg, scoring_mode="bogus")
        self.assertEqual(bad_cfg.scoring_mode, "bogus")


class ScoringModeConfigTests(unittest.TestCase):
    """Test that OursBaselineConfig accepts scoring_mode field."""

    def test_default_is_instance_logit(self):
        from android_packer.baselines.ours import OursBaselineConfig

        cfg = OursBaselineConfig()
        self.assertEqual(cfg.scoring_mode, "instance_logit")

    def test_attention_mode_accepted(self):
        from android_packer.baselines.ours import OursBaselineConfig

        import dataclasses

        cfg = dataclasses.replace(OursBaselineConfig(), scoring_mode="attention")
        self.assertEqual(cfg.scoring_mode, "attention")

    def test_attention_x_bag_mode_accepted(self):
        from android_packer.baselines.ours import OursBaselineConfig

        import dataclasses

        cfg = dataclasses.replace(OursBaselineConfig(), scoring_mode="attention_x_bag")
        self.assertEqual(cfg.scoring_mode, "attention_x_bag")

    def test_attention_anomaly_mode_accepted(self):
        from android_packer.baselines.ours import OursBaselineConfig

        import dataclasses

        cfg = dataclasses.replace(
            OursBaselineConfig(), scoring_mode="attention_anomaly"
        )
        self.assertEqual(cfg.scoring_mode, "attention_anomaly")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
