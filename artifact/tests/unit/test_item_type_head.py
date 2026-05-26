"""Unit tests for the DEX item-type auxiliary head (grammar-aware aux)."""

from __future__ import annotations

import unittest

import pytest

from android_packer.features.dex_item_parser import DEX_ITEM_TYPES
from android_packer.models.item_type_head import (
    DEX_ITEM_TYPE_PAD_ID,
    ItemTypeHeadConfig,
)

torch = pytest.importorskip("torch")

from android_packer.models.item_type_head import build_item_type_head  # noqa: E402


class ItemTypeHeadTests(unittest.TestCase):
    def test_n_item_types_default_matches_parser(self) -> None:
        cfg = ItemTypeHeadConfig()
        self.assertEqual(cfg.n_item_types, len(DEX_ITEM_TYPES))

    def test_forward_shape(self) -> None:
        head = build_item_type_head(
            ItemTypeHeadConfig(hidden_size=32, n_item_types=len(DEX_ITEM_TYPES))
        )
        h = torch.randn(2, 13, 32)
        logits = head(h)
        self.assertEqual(logits.shape, (2, 13, len(DEX_ITEM_TYPES)))

    def test_cross_entropy_ignores_pad(self) -> None:
        head = build_item_type_head(
            ItemTypeHeadConfig(hidden_size=16, n_item_types=4)
        )
        h = torch.randn(3, 5, 16)
        logits = head(h)  # [3, 5, 4]
        # 2 real tokens, 3 pad → ignore_index should mask them out.
        targets = torch.tensor(
            [[0, 1, DEX_ITEM_TYPE_PAD_ID, DEX_ITEM_TYPE_PAD_ID, DEX_ITEM_TYPE_PAD_ID]]
            * 3
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4),
            targets.reshape(-1),
            ignore_index=DEX_ITEM_TYPE_PAD_ID,
        )
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(float(loss), 0.0)

    def test_gradient_flows(self) -> None:
        head = build_item_type_head(
            ItemTypeHeadConfig(hidden_size=8, n_item_types=4)
        )
        h = torch.randn(1, 6, 8, requires_grad=True)
        targets = torch.zeros(1, 6, dtype=torch.long)
        logits = head(h)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4), targets.reshape(-1)
        )
        loss.backward()
        assert h.grad is not None
        self.assertGreater(float(torch.abs(h.grad).sum()), 0.0)

    def test_rejects_bad_config(self) -> None:
        with self.assertRaises(ValueError):
            build_item_type_head(ItemTypeHeadConfig(hidden_size=0))
        with self.assertRaises(ValueError):
            build_item_type_head(ItemTypeHeadConfig(n_item_types=1))
        with self.assertRaises(ValueError):
            build_item_type_head(ItemTypeHeadConfig(dropout=1.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
