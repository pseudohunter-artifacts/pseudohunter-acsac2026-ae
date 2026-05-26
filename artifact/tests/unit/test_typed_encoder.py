"""Unit tests for the typed per-instance encoder (Ours F-MIL-b)."""

from __future__ import annotations

import unittest

import pytest

from android_packer.models.typed_encoder import (
    N_TYPED_INSTANCE_TYPES,
    TYPED_INSTANCE_TYPES,
    instance_type_id,
)

torch = pytest.importorskip("torch")

from android_packer.models.typed_encoder import (  # noqa: E402
    TypedEncoderConfig,
    build_typed_encoder,
)


class TypedInstanceVocabularyTests(unittest.TestCase):
    def test_canonical_ordering_matches_labeling_adapter(self) -> None:
        # These must stay a superset of _PAYLOAD_KINDS + _LOADER_KINDS
        # from labeling/injected_packer_adapter.py. If that set grows,
        # TYPED_INSTANCE_TYPES must be extended in the same PR.
        from android_packer.labeling.injected_packer_adapter import (
            _LOADER_KINDS,
            _PAYLOAD_KINDS,
        )

        expected = set(_PAYLOAD_KINDS) | set(_LOADER_KINDS)
        self.assertSetEqual(set(TYPED_INSTANCE_TYPES), expected)
        self.assertEqual(len(TYPED_INSTANCE_TYPES), len(expected))

    def test_instance_type_id_stable(self) -> None:
        self.assertEqual(instance_type_id("encrypted_dex"), 0)
        # L42 (2026-05-07): ``benign_other`` added as the 7th (last) type
        # so benign APK objects have their own head instead of being
        # mis-routed through ``shim`` by the path-based default fallback.
        self.assertEqual(
            instance_type_id("benign_other"), N_TYPED_INSTANCE_TYPES - 1
        )

    def test_instance_type_id_rejects_unknown(self) -> None:
        with self.assertRaises(KeyError):
            instance_type_id("not_a_type")


class TypedEncoderShapeTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        cfg = TypedEncoderConfig(
            input_dim=15, hidden_dim=32, head_hidden_dim=16, n_types=6
        )
        enc = build_typed_encoder(cfg)
        feats = torch.randn(7, 15)
        types = torch.tensor([0, 1, 2, 3, 4, 5, 0], dtype=torch.long)
        hidden, logits = enc(feats, types)
        self.assertEqual(hidden.shape, (7, 16))
        self.assertEqual(logits.shape, (7,))

    def test_forward_empty_bag(self) -> None:
        enc = build_typed_encoder(
            TypedEncoderConfig(input_dim=4, hidden_dim=8, head_hidden_dim=4, n_types=6)
        )
        feats = torch.zeros((0, 4))
        types = torch.zeros((0,), dtype=torch.long)
        hidden, logits = enc(feats, types)
        self.assertEqual(hidden.shape, (0, 4))
        self.assertEqual(logits.shape, (0,))

    def test_per_type_routing_different_heads(self) -> None:
        """Same input routed through different type heads must differ."""
        cfg = TypedEncoderConfig(
            input_dim=4, hidden_dim=8, head_hidden_dim=4, n_types=6, dropout=0.0
        )
        enc = build_typed_encoder(cfg)
        enc.eval()  # disable dropout regardless of 0.0 setting
        x = torch.randn(1, 4)
        outs = []
        for t in range(6):
            feat = x.clone()
            typ = torch.tensor([t], dtype=torch.long)
            h, _ = enc(feat, typ)
            outs.append(h.detach().clone())
        # At least two heads should produce different outputs (they are
        # independently initialised; the probability of collision is ~0).
        some_pair_differs = any(
            not torch.allclose(outs[i], outs[j])
            for i in range(6)
            for j in range(i + 1, 6)
        )
        self.assertTrue(some_pair_differs)

    def test_rejects_out_of_range_type_id(self) -> None:
        enc = build_typed_encoder(
            TypedEncoderConfig(input_dim=4, hidden_dim=8, head_hidden_dim=4, n_types=3)
        )
        feats = torch.randn(2, 4)
        types = torch.tensor([0, 5], dtype=torch.long)
        with self.assertRaises(ValueError):
            enc(feats, types)

    def test_gradient_flows_through_selected_head_only(self) -> None:
        cfg = TypedEncoderConfig(
            input_dim=4, hidden_dim=8, head_hidden_dim=4, n_types=3, dropout=0.0
        )
        enc = build_typed_encoder(cfg)
        feats = torch.randn(2, 4, requires_grad=True)
        types = torch.tensor([0, 0], dtype=torch.long)
        _, logits = enc(feats, types)
        logits.sum().backward()
        # Gradients must exist on feats.
        assert feats.grad is not None
        self.assertGreater(float(torch.abs(feats.grad).sum()), 0.0)

        # Head 0 parameters should have gradient; head 2 should not.
        grad_head_0 = sum(
            float(torch.abs(p.grad).sum())
            for p in enc.type_proj[0].parameters()
            if p.grad is not None
        )
        grad_head_2 = sum(
            float(torch.abs(p.grad).sum()) if p.grad is not None else 0.0
            for p in enc.type_proj[2].parameters()
        )
        self.assertGreater(grad_head_0, 0.0)
        self.assertEqual(grad_head_2, 0.0)


class TypedEncoderConfigValidationTests(unittest.TestCase):
    def test_rejects_bad_dims(self) -> None:
        with self.assertRaises(ValueError):
            build_typed_encoder(TypedEncoderConfig(input_dim=0))
        with self.assertRaises(ValueError):
            build_typed_encoder(TypedEncoderConfig(hidden_dim=0))
        with self.assertRaises(ValueError):
            build_typed_encoder(TypedEncoderConfig(head_hidden_dim=0))
        with self.assertRaises(ValueError):
            build_typed_encoder(TypedEncoderConfig(num_trunk_layers=0))
        with self.assertRaises(ValueError):
            build_typed_encoder(TypedEncoderConfig(dropout=1.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
