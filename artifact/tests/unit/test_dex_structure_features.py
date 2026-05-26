"""Unit tests for dex_structure_features.py (Tier 1A, Group H, 12-dim).

Verifies the contract described in ``improvement_plan_L47.md`` §1A:
- Feature vector always has length N_DEX_STRUCTURE_FEATURES (12).
- Zero vector returned for non-DEX bytes.
- Section fractions sum to 1.0 for DEX objects.
- dominant_section is normalised to [0, 1].
- cross_section_count_log2 = log2(1) = 0 for a pure single-section window.
- All values are finite and non-negative.
- Name inventory matches DEX_STRUCTURE_FEATURE_NAMES (pinned order).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _dex_fixtures import build_minimal_dex  # noqa: E402

from android_packer.features.dex_structure_features import (  # noqa: E402
    DEX_STRUCTURE_FEATURE_NAMES,
    N_DEX_STRUCTURE_FEATURES,
    extract_dex_structure_features,
)
from android_packer.features.dex_item_parser import DEX_ITEM_TYPES  # noqa: E402


class NameInventoryTests(unittest.TestCase):
    """The feature name tuple is pinned — once a checkpoint is trained on it
    we must never reorder or rename entries."""

    def test_count_is_12(self) -> None:
        self.assertEqual(N_DEX_STRUCTURE_FEATURES, 12)

    def test_feature_names_length(self) -> None:
        self.assertEqual(len(DEX_STRUCTURE_FEATURE_NAMES), 12)

    def test_section_dims_are_first_10(self) -> None:
        for i, item_type in enumerate(DEX_ITEM_TYPES):
            self.assertEqual(DEX_STRUCTURE_FEATURE_NAMES[i], f"dex_sec_{item_type}")

    def test_derived_dims_are_last_two(self) -> None:
        self.assertEqual(DEX_STRUCTURE_FEATURE_NAMES[-2], "dex_dominant_section")
        self.assertEqual(DEX_STRUCTURE_FEATURE_NAMES[-1], "dex_cross_section_count_log2")


class NonDexObjectTests(unittest.TestCase):
    """Non-DEX bytes (random, ELF, ZIP, etc.) must return the zero vector."""

    def _assert_zero_vec(self, data: bytes, offset: int = 0) -> None:
        feats = extract_dex_structure_features(data, offset, len(data) - offset)
        self.assertEqual(len(feats), N_DEX_STRUCTURE_FEATURES)
        self.assertEqual(feats, [0.0] * N_DEX_STRUCTURE_FEATURES)

    def test_empty_bytes(self) -> None:
        self._assert_zero_vec(b"")

    def test_random_bytes(self) -> None:
        self._assert_zero_vec(b"\xff" * 4096)

    def test_elf_magic(self) -> None:
        self._assert_zero_vec(b"\x7fELF" + b"\x00" * 4096)

    def test_zero_region_size_returns_zero_vec(self) -> None:
        dex, _ = build_minimal_dex()
        feats = extract_dex_structure_features(dex, 0, 0)
        self.assertEqual(feats, [0.0] * N_DEX_STRUCTURE_FEATURES)

    def test_negative_region_size_is_clamped(self) -> None:
        dex, _ = build_minimal_dex()
        feats = extract_dex_structure_features(dex, 0, -10)
        self.assertEqual(feats, [0.0] * N_DEX_STRUCTURE_FEATURES)


class ValidDexTests(unittest.TestCase):
    """Well-formed DEX must produce meaningful, valid feature vectors."""

    def setUp(self) -> None:
        self.dex, self.layout = build_minimal_dex()

    def test_returns_correct_length(self) -> None:
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        self.assertEqual(len(feats), N_DEX_STRUCTURE_FEATURES)

    def test_all_values_are_finite(self) -> None:
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        for i, v in enumerate(feats):
            self.assertTrue(math.isfinite(v), f"feature[{i}] = {v} is not finite")

    def test_all_values_non_negative(self) -> None:
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        for i, v in enumerate(feats):
            self.assertGreaterEqual(v, 0.0, f"feature[{i}] = {v} is negative")

    def test_section_fractions_sum_to_one(self) -> None:
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        section_sum = sum(feats[:len(DEX_ITEM_TYPES)])
        self.assertAlmostEqual(section_sum, 1.0, places=6)

    def test_dominant_section_in_0_1(self) -> None:
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        dominant = feats[len(DEX_ITEM_TYPES)]  # second-to-last
        self.assertGreaterEqual(dominant, 0.0)
        self.assertLessEqual(dominant, 1.0)

    def test_header_region_is_dominated_by_header_section(self) -> None:
        # The header spans bytes [0, 0x70). Request the first 0x70 bytes.
        header_size = 0x70
        feats = extract_dex_structure_features(self.dex, 0, header_size)
        # DEX_ITEM_TYPES[0] == "header"
        header_frac = feats[0]
        self.assertGreater(header_frac, 0.9,
                           f"Expected header section to dominate first 0x70 bytes, got {header_frac}")

    def test_string_ids_region_is_dominated_by_string_ids(self) -> None:
        off = self.layout.string_ids_off
        size = self.layout.string_ids_count * 4  # each string_id is 4 bytes
        if size == 0:
            self.skipTest("No string_ids in fixture")
        feats = extract_dex_structure_features(self.dex, off, size)
        # DEX_ITEM_TYPES[1] == "string_ids"
        string_ids_frac = feats[1]
        self.assertGreater(string_ids_frac, 0.9,
                           f"Expected string_ids section to dominate, got {string_ids_frac}")

    def test_pure_single_section_window_has_zero_cross_section_count(self) -> None:
        # The header is at [0, 0x70) and covers exactly one section.
        header_size = 0x70
        feats = extract_dex_structure_features(self.dex, 0, header_size)
        cross_section_log2 = feats[-1]
        # log2(0 + 1) = 0 → no section boundaries within a pure window
        self.assertAlmostEqual(cross_section_log2, 0.0, places=6)

    def test_full_dex_cross_section_count_positive(self) -> None:
        # A full DEX spanning all sections must have > 1 section boundary.
        feats = extract_dex_structure_features(self.dex, 0, len(self.dex))
        cross_section_log2 = feats[-1]
        self.assertGreater(cross_section_log2, 0.0,
                           "Full DEX window should span multiple sections")


class RegionWindowBoundaryTests(unittest.TestCase):
    """Region window logic: offset + partial coverage."""

    def setUp(self) -> None:
        self.dex, self.layout = build_minimal_dex()

    def test_window_beyond_dex_end_returns_valid_vec(self) -> None:
        # offset_start > len(dex) — no coverage, all bytes go to "other"
        feats = extract_dex_structure_features(self.dex, len(self.dex) + 100, 512)
        self.assertEqual(len(feats), N_DEX_STRUCTURE_FEATURES)
        # All bytes uncovered — the two-pointer sweep credits them to "other"
        # (last index), so section_fracs[-1] should be 1.0
        other_frac = feats[len(DEX_ITEM_TYPES) - 1]
        self.assertAlmostEqual(other_frac, 1.0, places=6)

    def test_partial_window_fracs_still_sum_to_one(self) -> None:
        # Take a 64-byte window starting mid-header.
        feats = extract_dex_structure_features(self.dex, 32, 64)
        section_sum = sum(feats[:len(DEX_ITEM_TYPES)])
        self.assertAlmostEqual(section_sum, 1.0, places=6)

    def test_cross_section_window_has_positive_boundary_count(self) -> None:
        # A window that spans header + string_ids has at least 1 boundary.
        # string_ids starts right after header (offset 0x70).
        feats = extract_dex_structure_features(
            self.dex,
            0,  # start at header
            self.layout.string_ids_off + self.layout.string_ids_count * 4,  # end after string_ids
        )
        cross_section_log2 = feats[-1]
        self.assertGreater(cross_section_log2, 0.0)


class HandcraftedConfigIntegrationTests(unittest.TestCase):
    """Smoke test: HandcraftedFeatureConfig.include_dex_structure toggles Group H."""

    def test_feature_names_extended_when_enabled(self) -> None:
        from android_packer.features.handcrafted import (
            HandcraftedFeatureConfig,
            handcrafted_feature_names,
        )
        cfg_off = HandcraftedFeatureConfig(include_dex_structure=False)
        cfg_on = HandcraftedFeatureConfig(include_dex_structure=True)
        names_off = handcrafted_feature_names(cfg_off)
        names_on = handcrafted_feature_names(cfg_on)
        extra = set(names_on) - set(names_off)
        self.assertEqual(extra, set(DEX_STRUCTURE_FEATURE_NAMES))
        self.assertEqual(len(names_on) - len(names_off), N_DEX_STRUCTURE_FEATURES)

    def test_feature_names_not_extended_when_disabled(self) -> None:
        from android_packer.features.handcrafted import (
            HandcraftedFeatureConfig,
            handcrafted_feature_names,
        )
        cfg_off = HandcraftedFeatureConfig(include_dex_structure=False)
        names_off = handcrafted_feature_names(cfg_off)
        for n in DEX_STRUCTURE_FEATURE_NAMES:
            self.assertNotIn(n, names_off)


if __name__ == "__main__":
    unittest.main()
