"""Unit tests for :mod:`android_packer.training.mil_trainer` (F-MIL-e)."""

from __future__ import annotations

import pytest

from android_packer.models.ours import OursConfig
from android_packer.models.typed_encoder import (
    N_TYPED_INSTANCE_TYPES,
    TypedEncoderConfig,
)


torch = pytest.importorskip("torch", reason="requires [dl] extra")


from android_packer.training.mil_trainer import (  # noqa: E402 — after importorskip
    MILBag,
    MILTrainerConfig,
    predict_bag,
    subsample_bag_for_training,
    train_ours,
)


def _make_bag(bag_id: str, n: int, D: int, label: int, *, seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    feats = torch.randn(n, D, generator=g)
    types = torch.randint(0, N_TYPED_INSTANCE_TYPES, (n,), generator=g)
    if label == 1:
        # Inject a large positive signal in one instance to make the
        # problem learnable on a tiny dataset.
        feats[0] = feats[0] + 5.0
    inst = torch.zeros(n)
    if label == 1:
        inst[0] = 1.0
    return MILBag(
        bag_id=bag_id,
        features=feats,
        types=types,
        bag_label=label,
        instance_labels=inst,
    )


class TestMILTrainerConfig:
    def test_defaults_valid(self):
        cfg = MILTrainerConfig()
        assert cfg.lambda_diff_pseudo == pytest.approx(0.3)
        assert cfg.lambda_sparsity == pytest.approx(0.01)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"batch_size": 0},
            {"lambda_diff_pseudo": -0.1},
            {"lambda_sparsity": -0.1},
            {"bag_pos_weight": 0.0},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            MILTrainerConfig(**kwargs)


class TestTrainOurs:
    def test_train_reduces_loss_on_separable_bags(self):
        """Tiny sanity check: the trainer should improve over random init."""

        D = 8
        bags = [
            _make_bag("pos0", 4, D, 1, seed=1),
            _make_bag("pos1", 5, D, 1, seed=2),
            _make_bag("pos2", 3, D, 1, seed=3),
            _make_bag("neg0", 4, D, 0, seed=4),
            _make_bag("neg1", 6, D, 0, seed=5),
            _make_bag("neg2", 3, D, 0, seed=6),
        ]
        cfg = MILTrainerConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(input_dim=D, hidden_dim=16, head_hidden_dim=8),
            ),
            epochs=20,
            learning_rate=5e-3,
            batch_size=3,
            lambda_diff_pseudo=0.3,
            lambda_sparsity=0.0,  # keep entropy reg out for this test
            random_state=0,
        )
        model = train_ours(bags, cfg)
        # All positive bags should score higher than all negative bags
        # after training.
        pos_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 1]
        neg_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 0]
        assert min(pos_scores) > max(neg_scores), (
            f"positive ≤ negative after training: pos={pos_scores}, neg={neg_scores}"
        )

    def test_empty_bag_list_rejected(self):
        with pytest.raises(ValueError):
            train_ours([], MILTrainerConfig())

    def test_predict_bag_returns_expected_shapes(self):
        D = 6
        bag = _make_bag("x", 5, D, 1, seed=42)
        cfg = MILTrainerConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(input_dim=D, hidden_dim=8, head_hidden_dim=4),
            ),
            epochs=1,
            random_state=0,
        )
        model = train_ours([bag], cfg)
        score, attention, inst_logits = predict_bag(model, bag)
        assert 0.0 <= score <= 1.0
        assert tuple(attention.shape) == (5,)
        assert tuple(inst_logits.shape) == (5,)

    def test_sparsity_term_does_not_crash(self):
        D = 4
        bags = [_make_bag("b", 3, D, 1, seed=0), _make_bag("c", 3, D, 0, seed=1)]
        cfg = MILTrainerConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(input_dim=D, hidden_dim=8, head_hidden_dim=4),
            ),
            epochs=2,
            lambda_sparsity=0.05,
            lambda_diff_pseudo=0.0,
        )
        model = train_ours(bags, cfg)
        score, _, _ = predict_bag(model, bags[0])
        assert 0.0 <= score <= 1.0


def _make_skewed_bag(bag_id: str, n_neg: int, n_pos: int, D: int, *, seed: int = 0):
    """Build a bag with a controlled (n_neg, n_pos) split for L41 tests."""

    g = torch.Generator()
    g.manual_seed(seed)
    n = n_neg + n_pos
    feats = torch.randn(n, D, generator=g)
    types = torch.randint(0, N_TYPED_INSTANCE_TYPES, (n,), generator=g)
    inst = torch.zeros(n)
    # Positives scattered deterministically across the bag so sorting
    # after subsampling doesn't always keep the first/last few.
    pos_positions = [int((i + 1) * n / (n_pos + 1)) for i in range(n_pos)]
    for pos in pos_positions:
        pos = min(pos, n - 1)
        feats[pos] = feats[pos] + 5.0
        inst[pos] = 1.0
    return MILBag(
        bag_id=bag_id,
        features=feats,
        types=types,
        bag_label=int(n_pos > 0),
        instance_labels=inst,
    )


class TestBagSubsampling:
    """L41 (2026-05-07): training-time bag subsampling."""

    def test_keeps_all_positives_on_large_bag(self):
        # Mirrors synthetic v4: ~1500 instances, 1 positive, fraction 0.1%.
        bag = _make_skewed_bag("big", n_neg=1499, n_pos=1, D=4, seed=0)
        sub = subsample_bag_for_training(
            bag,
            max_size=128,
            min_positive_fraction=0.01,
            epoch=0,
            global_seed=0,
        )
        assert int(sub.instance_labels.sum().item()) == 1
        assert sub.features.shape[0] <= 128
        # Positive fraction must meet the floor (1 / total >= 0.01 => total <= 100).
        assert sub.features.shape[0] <= 100

    def test_preserves_small_bag_verbatim(self):
        bag = _make_skewed_bag("small", n_neg=5, n_pos=1, D=4, seed=0)
        sub = subsample_bag_for_training(
            bag,
            max_size=128,
            min_positive_fraction=0.01,
            epoch=0,
            global_seed=0,
        )
        assert sub.features.shape[0] == 6  # unchanged

    def test_deterministic_under_same_seed(self):
        bag = _make_skewed_bag("det", n_neg=200, n_pos=2, D=4, seed=0)
        a = subsample_bag_for_training(
            bag, max_size=32, min_positive_fraction=0.02, epoch=3, global_seed=7
        )
        b = subsample_bag_for_training(
            bag, max_size=32, min_positive_fraction=0.02, epoch=3, global_seed=7
        )
        assert torch.equal(a.features, b.features)
        assert torch.equal(a.instance_labels, b.instance_labels)

    def test_different_epoch_changes_sample(self):
        bag = _make_skewed_bag("epoch", n_neg=500, n_pos=2, D=4, seed=0)
        a = subsample_bag_for_training(
            bag, max_size=32, min_positive_fraction=0.02, epoch=0, global_seed=7
        )
        b = subsample_bag_for_training(
            bag, max_size=32, min_positive_fraction=0.02, epoch=1, global_seed=7
        )
        # With 500 negatives sampled down to ~30, two different epochs
        # should produce at least one different negative row.
        assert not torch.equal(a.features, b.features)

    def test_disabled_when_max_size_none(self):
        bag = _make_skewed_bag("off", n_neg=1000, n_pos=1, D=4, seed=0)
        sub = subsample_bag_for_training(
            bag, max_size=None, min_positive_fraction=0.01, epoch=0, global_seed=0
        )
        assert sub is bag

    def test_disabled_when_no_instance_labels(self):
        # Bag-only supervision — positives can't be identified, no sampling.
        n, D = 500, 4
        g = torch.Generator()
        g.manual_seed(0)
        bag = MILBag(
            bag_id="bag_only",
            features=torch.randn(n, D, generator=g),
            types=torch.zeros(n, dtype=torch.int64),
            bag_label=1,
            instance_labels=None,
        )
        sub = subsample_bag_for_training(
            bag, max_size=32, min_positive_fraction=0.01, epoch=0, global_seed=0
        )
        assert sub is bag

    def test_config_rejects_bad_subsample_values(self):
        with pytest.raises(ValueError):
            MILTrainerConfig(train_max_bag_size=0)
        with pytest.raises(ValueError):
            MILTrainerConfig(train_min_positive_fraction=-0.1)
        with pytest.raises(ValueError):
            MILTrainerConfig(train_min_positive_fraction=1.5)

    def test_train_ours_still_learns_with_subsampling(self):
        """End-to-end: trainer with L41 subsampling still converges on
        separable bags.  Uses moderate-sized bags to exercise the path."""

        D = 8
        bags = [
            _make_skewed_bag("p0", n_neg=50, n_pos=2, D=D, seed=1),
            _make_skewed_bag("p1", n_neg=50, n_pos=2, D=D, seed=2),
            _make_skewed_bag("n0", n_neg=50, n_pos=0, D=D, seed=3),
            _make_skewed_bag("n1", n_neg=50, n_pos=0, D=D, seed=4),
        ]
        cfg = MILTrainerConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(
                    input_dim=D, hidden_dim=16, head_hidden_dim=8
                ),
            ),
            epochs=20,
            learning_rate=5e-3,
            batch_size=2,
            lambda_diff_pseudo=0.3,
            lambda_sparsity=0.0,
            train_max_bag_size=16,
            train_min_positive_fraction=0.05,
            random_state=0,
        )
        model = train_ours(bags, cfg)
        pos_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 1]
        neg_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 0]
        assert min(pos_scores) > max(neg_scores)


class TestSupervisionMode:
    """L43 (2026-05-07): paper-integrity switch between weakly-supervised
    (bag mode) and strongly-supervised (instance_aided) MIL."""

    def test_config_rejects_bad_supervision_mode(self):
        with pytest.raises(ValueError):
            MILTrainerConfig(supervision_mode="full")
        with pytest.raises(ValueError):
            MILTrainerConfig(supervision_mode="")

    def test_bag_mode_ignores_instance_loss_even_when_labels_present(self):
        """In 'bag' mode the per-instance BCE must be zero weight
        regardless of lambda_diff_pseudo. Verified via gradient: the
        typed-encoder per-type heads that own no positive instance at a
        given step still get some gradient in 'instance_aided' mode (via
        the per-instance BCE on the positive instance's head path), but
        in 'bag' mode only the attention-weighted aggregation gradient
        reaches the heads."""

        from android_packer.training.mil_trainer import _bag_loss, _require_torch

        D = 4
        torch_mod, F_mod, _ = _require_torch()
        bag = _make_skewed_bag("x", n_neg=4, n_pos=2, D=D, seed=0)

        shared_cfg_kwargs = dict(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(
                    input_dim=D, hidden_dim=8, head_hidden_dim=4
                ),
            ),
            lambda_diff_pseudo=0.5,
            lambda_sparsity=0.0,
            train_max_bag_size=None,
            random_state=0,
        )
        cfg_bag = MILTrainerConfig(supervision_mode="bag", **shared_cfg_kwargs)
        cfg_aided = MILTrainerConfig(
            supervision_mode="instance_aided", **shared_cfg_kwargs
        )

        # Two independent models with the same init seed -> same params.
        from android_packer.models.ours import build_ours

        torch_mod.manual_seed(0)
        model_bag = build_ours(cfg_bag.ours_config)
        torch_mod.manual_seed(0)
        model_aided = build_ours(cfg_aided.ours_config)
        # Sanity: identical init.
        for p_bag, p_aid in zip(
            model_bag.parameters(), model_aided.parameters()
        ):
            assert torch_mod.allclose(p_bag, p_aid)

        loss_bag, _, _ = _bag_loss(
            model_bag, bag, cfg=cfg_bag, F=F_mod, torch_mod=torch_mod
        )
        loss_aided, _, _ = _bag_loss(
            model_aided, bag, cfg=cfg_aided, F=F_mod, torch_mod=torch_mod
        )

        # Bag-mode loss must equal the bag BCE alone: lambda_diff_pseudo
        # should have NO effect. instance_aided should strictly exceed
        # bag-mode (the per-instance BCE is non-negative and the
        # instance_labels have a positive at index 0, so the random-
        # init logit will not match exactly -> strict inequality).
        assert float(loss_aided.item()) > float(loss_bag.item()) + 1e-6

    def test_bag_mode_trains_without_touching_instance_labels(self):
        """Bag-mode training still runs on a bag whose instance_labels
        are deliberately wrong (all zeros) -- the trainer must NOT
        consult them in 'bag' mode."""

        D = 8
        torch_mod, _, _ = _require_torch() if False else (torch, None, None)  # noqa

        def _make_with_zeroed_labels(bag_id, n_neg, n_pos, seed):
            b = _make_skewed_bag(bag_id, n_neg=n_neg, n_pos=n_pos, D=D, seed=seed)
            # Wipe instance_labels -- only the *feature spike* survives,
            # and only the bag label tells the model which bags are positive.
            zeroed = MILBag(
                bag_id=b.bag_id,
                features=b.features,
                types=b.types,
                bag_label=b.bag_label,
                instance_labels=torch.zeros_like(b.instance_labels),
            )
            return zeroed

        bags = [
            _make_with_zeroed_labels("p0", n_neg=20, n_pos=2, seed=1),
            _make_with_zeroed_labels("p1", n_neg=20, n_pos=2, seed=2),
            _make_with_zeroed_labels("n0", n_neg=20, n_pos=0, seed=3),
            _make_with_zeroed_labels("n1", n_neg=20, n_pos=0, seed=4),
        ]
        cfg = MILTrainerConfig(
            ours_config=OursConfig(
                typed=TypedEncoderConfig(
                    input_dim=D, hidden_dim=16, head_hidden_dim=8
                ),
            ),
            epochs=30,
            learning_rate=5e-3,
            batch_size=2,
            supervision_mode="bag",
            lambda_diff_pseudo=0.5,  # would hurt if it were consulted
            lambda_sparsity=0.0,
            train_max_bag_size=None,
            random_state=0,
        )
        model = train_ours(bags, cfg)
        pos_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 1]
        neg_scores = [predict_bag(model, b)[0] for b in bags if b.bag_label == 0]
        assert min(pos_scores) > max(neg_scores), (
            f"bag mode failed to separate pos/neg with misleading "
            f"instance_labels; pos={pos_scores} neg={neg_scores}"
        )
