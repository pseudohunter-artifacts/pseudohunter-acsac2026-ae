"""Unit tests for :mod:`android_packer.training.contrastive` (F-MIL-d)."""

from __future__ import annotations

import pytest

from android_packer.training.contrastive import (
    ContrastiveConfig,
    ContrastivePairBatch,
    build_pair_batch,
    compute_contrastive_loss,
    info_nce_app_head,
    info_nce_pack_residual,
)


torch = pytest.importorskip("torch", reason="requires [dl] extra")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestContrastiveConfig:
    def test_defaults_valid(self):
        cfg = ContrastiveConfig()
        assert cfg.temperature_app > 0
        assert cfg.pack_loss_weight >= 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature_app": 0.0},
            {"temperature_app": -0.1},
            {"temperature_pack": -0.1},
            {"pack_loss_weight": -0.001},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            ContrastiveConfig(**kwargs)


# ---------------------------------------------------------------------------
# build_pair_batch — validation
# ---------------------------------------------------------------------------


class TestBuildPairBatch:
    def test_accepts_aligned_shapes(self):
        a = torch.randn(4, 16)
        p = torch.randn(4, 16)
        batch = build_pair_batch(a, p)
        assert isinstance(batch, ContrastivePairBatch)
        assert batch.packer_ids is None

    def test_rejects_mismatched_batch(self):
        a = torch.randn(4, 16)
        p = torch.randn(5, 16)
        with pytest.raises(ValueError):
            build_pair_batch(a, p)

    def test_rejects_mismatched_feature_dim(self):
        a = torch.randn(4, 16)
        p = torch.randn(4, 8)
        with pytest.raises(ValueError):
            build_pair_batch(a, p)

    def test_rejects_bad_packer_ids_shape(self):
        a = torch.randn(4, 16)
        p = torch.randn(4, 16)
        pid = torch.zeros(3, dtype=torch.long)
        with pytest.raises(ValueError):
            build_pair_batch(a, p, pid)

    def test_accepts_packer_ids(self):
        a = torch.randn(4, 16)
        p = torch.randn(4, 16)
        pid = torch.tensor([0, 0, 1, 1])
        batch = build_pair_batch(a, p, pid)
        assert batch.packer_ids is pid


# ---------------------------------------------------------------------------
# info_nce_app_head
# ---------------------------------------------------------------------------


class TestInfoNCEAppHead:
    def test_loss_is_positive_and_backprops(self):
        torch.manual_seed(0)
        z_a = torch.randn(6, 32, requires_grad=True)
        z_p = torch.randn(6, 32, requires_grad=True)
        loss = info_nce_app_head(z_a, z_p, temperature=0.1)
        assert loss.item() > 0
        loss.backward()
        assert z_a.grad is not None
        assert torch.any(z_a.grad != 0)

    def test_perfect_alignment_yields_low_loss(self):
        torch.manual_seed(0)
        z = torch.randn(6, 32)
        # positives == anchors => cosine sim 1.0 on the diagonal.
        loss_perfect = info_nce_app_head(z, z.clone(), temperature=0.1)
        # Unrelated pairs should give a higher loss.
        z_shuffled = z[torch.randperm(z.shape[0])]
        loss_shuffled = info_nce_app_head(z, z_shuffled, temperature=0.1)
        assert loss_perfect.item() < loss_shuffled.item()

    def test_rejects_batch_size_one(self):
        z = torch.randn(1, 16)
        with pytest.raises(ValueError):
            info_nce_app_head(z, z.clone())

    def test_rejects_shape_mismatch(self):
        z_a = torch.randn(4, 16)
        z_p = torch.randn(4, 8)
        with pytest.raises(ValueError):
            info_nce_app_head(z_a, z_p)


# ---------------------------------------------------------------------------
# info_nce_pack_residual
# ---------------------------------------------------------------------------


class TestInfoNCEPackResidual:
    def test_pulls_same_packer_together(self):
        """Pairs sharing a packer id should produce less loss than all-different."""

        torch.manual_seed(0)
        B, D = 6, 32
        z_a = torch.randn(B, D)
        # Construct z_p such that pairs 0..2 share a packer-family delta
        # and pairs 3..5 share a different one.
        delta_f0 = torch.randn(1, D)
        delta_f1 = torch.randn(1, D)
        z_p_good = z_a.clone()
        z_p_good[:3] = z_a[:3] + delta_f0 + 0.01 * torch.randn(3, D)
        z_p_good[3:] = z_a[3:] + delta_f1 + 0.01 * torch.randn(3, D)
        pid_good = torch.tensor([0, 0, 0, 1, 1, 1])
        loss_good = info_nce_pack_residual(z_a, z_p_good, pid_good)

        # If we instead scramble the ids, pairs no longer cluster.
        pid_bad = torch.tensor([0, 1, 0, 1, 0, 1])
        loss_bad = info_nce_pack_residual(z_a, z_p_good, pid_bad)
        assert loss_good.item() < loss_bad.item()

    def test_singleton_batch_falls_back_to_zero(self):
        z_a = torch.randn(3, 16, requires_grad=True)
        z_p = torch.randn(3, 16, requires_grad=True)
        pid = torch.tensor([0, 1, 2])  # no two share an id
        loss = info_nce_pack_residual(z_a, z_p, pid)
        assert loss.item() == pytest.approx(0.0)
        # 0-with-grad must still backprop cleanly.
        loss.backward()

    def test_rejects_bad_pid_shape(self):
        z_a = torch.randn(4, 16)
        z_p = torch.randn(4, 16)
        pid = torch.zeros(5, dtype=torch.long)
        with pytest.raises(ValueError):
            info_nce_pack_residual(z_a, z_p, pid)


# ---------------------------------------------------------------------------
# compute_contrastive_loss — combined
# ---------------------------------------------------------------------------


class TestComputeContrastiveLoss:
    def test_combined_loss_gradients_reach_both_heads(self):
        torch.manual_seed(0)
        a = torch.randn(6, 32, requires_grad=True)
        p = torch.randn(6, 32, requires_grad=True)
        pid = torch.tensor([0, 0, 0, 1, 1, 1])
        batch = build_pair_batch(a, p, pid)
        cfg = ContrastiveConfig(pack_loss_weight=1.0)
        loss = compute_contrastive_loss(batch, cfg)
        loss.backward()
        assert a.grad is not None
        assert p.grad is not None

    def test_pack_weight_zero_skips_residual_branch(self):
        """Ablation contract: ``pack_loss_weight==0`` must match app-only."""

        torch.manual_seed(0)
        a = torch.randn(6, 32, requires_grad=True)
        p = torch.randn(6, 32, requires_grad=True)
        pid = torch.tensor([0, 0, 0, 1, 1, 1])
        batch = build_pair_batch(a, p, pid)
        cfg = ContrastiveConfig(pack_loss_weight=0.0)
        loss_combined = compute_contrastive_loss(batch, cfg)

        # Recompute manually with only app head.
        loss_app_only = info_nce_app_head(
            a, p, temperature=cfg.temperature_app, normalize=cfg.normalize
        )
        assert torch.allclose(loss_combined, loss_app_only)

    def test_no_packer_ids_skips_residual_branch(self):
        a = torch.randn(6, 32)
        p = torch.randn(6, 32)
        batch = build_pair_batch(a, p, packer_ids=None)
        cfg = ContrastiveConfig(pack_loss_weight=1.0)
        loss = compute_contrastive_loss(batch, cfg)
        # Single value returned even when residual would apply, because
        # packer_ids is absent.
        assert loss.dim() == 0
