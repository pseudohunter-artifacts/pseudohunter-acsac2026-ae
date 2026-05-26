"""Unit tests for the 8 DEX-aware scalar structural features (F2a).

Status: this module is an **ablation-only** input for Ours (see
``docs/method/ours_method_spec.md`` §3.2.1 and ``docs/research_framing.md``
§4.3 / §5.3). These tests nail down the feature names and the 0/1
semantics so that downstream ablation configs
(``configs/eval/ablation/ours_with_scalar_struct.json``) keep producing
comparable numbers if we ever rewire the feature extractor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _dex_fixtures import build_minimal_dex  # noqa: E402

from android_packer.features.dex_features import (  # noqa: E402
    DexStructuralFeatureConfig,
    extract_region_structural_features,
)


DEX_MAGIC = b"dex\n035\x00"


class FeatureInventoryTests(unittest.TestCase):
    """The set of feature names is frozen by spec §3.2.1."""

    EXPECTED = {
        "dex_magic_present",
        "dex_header_plausible_file_size",
        "map_list_offset_plausible",
        "string_ids_offset_aligned",
        "nearby_zip_local_header",
        "object_path_is_dex_like",
        "object_path_is_asset",
        "object_path_is_lib_so",
    }

    def test_all_features_returned_with_default_config(self) -> None:
        features = extract_region_structural_features(
            raw_bytes=DEX_MAGIC + b"\x00" * 4096,
            object_path="classes.dex",
            offset_start=0,
            offset_end=4096,
        )
        self.assertEqual(set(features), self.EXPECTED)

    def test_values_are_float_0_or_1(self) -> None:
        features = extract_region_structural_features(
            raw_bytes=DEX_MAGIC + b"\x00" * 4096,
            object_path="assets/payload.bin",
            offset_start=0,
            offset_end=4096,
        )
        for name, value in features.items():
            self.assertIsInstance(value, float, f"{name} must be float")
            self.assertIn(value, (0.0, 1.0), f"{name} must be 0.0 or 1.0, got {value}")


class DexMagicTests(unittest.TestCase):
    def test_magic_detected_for_version_035_through_039(self) -> None:
        for version in (b"035\x00", b"036\x00", b"037\x00", b"038\x00", b"039\x00"):
            data = b"dex\n" + version + b"\x00" * 64
            features = extract_region_structural_features(
                raw_bytes=data, object_path="whatever", offset_start=0, offset_end=len(data)
            )
            self.assertEqual(features["dex_magic_present"], 1.0)

    def test_magic_absent_on_random_bytes(self) -> None:
        data = b"\xff" * 4096
        features = extract_region_structural_features(
            raw_bytes=data, object_path="whatever", offset_start=0, offset_end=len(data)
        )
        self.assertEqual(features["dex_magic_present"], 0.0)

    def test_magic_absent_on_too_short_buffer(self) -> None:
        features = extract_region_structural_features(
            raw_bytes=b"dex", object_path="whatever", offset_start=0, offset_end=3
        )
        self.assertEqual(features["dex_magic_present"], 0.0)


class HeaderPlausibilityTests(unittest.TestCase):
    def test_minimal_dex_passes_plausibility_checks(self) -> None:
        dex_bytes, layout = build_minimal_dex()
        # Pad to 4 KiB to match the region-window convention.
        padded = dex_bytes + b"\x00" * (4096 - len(dex_bytes))
        features = extract_region_structural_features(
            raw_bytes=padded, object_path="classes.dex",
            offset_start=0, offset_end=4096,
        )
        self.assertEqual(features["dex_magic_present"], 1.0)
        self.assertEqual(features["map_list_offset_plausible"], 1.0)
        self.assertEqual(features["string_ids_offset_aligned"], 1.0)
        # Silence the unused variable linter.
        _ = layout

    def test_file_size_implausible_on_random_bytes(self) -> None:
        data = b"\xff" * 4096
        features = extract_region_structural_features(
            raw_bytes=data, object_path="x", offset_start=0, offset_end=len(data)
        )
        self.assertEqual(features["dex_header_plausible_file_size"], 0.0)


class ZipHeaderProximityTests(unittest.TestCase):
    def test_zip_header_in_first_512_bytes_fires(self) -> None:
        data = b"\x00" * 100 + b"PK\x03\x04" + b"\x00" * 4000
        features = extract_region_structural_features(
            raw_bytes=data, object_path="x", offset_start=0, offset_end=len(data)
        )
        self.assertEqual(features["nearby_zip_local_header"], 1.0)

    def test_zip_header_absent_fires_zero(self) -> None:
        data = b"\x00" * 4096
        features = extract_region_structural_features(
            raw_bytes=data, object_path="x", offset_start=0, offset_end=len(data)
        )
        self.assertEqual(features["nearby_zip_local_header"], 0.0)


class ObjectPathHeuristicsTests(unittest.TestCase):
    def _features_for_path(self, path: str) -> dict:
        return extract_region_structural_features(
            raw_bytes=DEX_MAGIC + b"\x00" * 8,
            object_path=path, offset_start=0, offset_end=16,
        )

    def test_classes_dex_is_dex_like(self) -> None:
        for path in ("classes.dex", "classes2.dex", "assets/classes.dex"):
            f = self._features_for_path(path)
            self.assertEqual(f["object_path_is_dex_like"], 1.0, path)

    def test_arbitrary_bin_is_not_dex_like(self) -> None:
        f = self._features_for_path("assets/payload.bin")
        self.assertEqual(f["object_path_is_dex_like"], 0.0)

    def test_assets_prefix_detected(self) -> None:
        f = self._features_for_path("assets/payload.dat")
        self.assertEqual(f["object_path_is_asset"], 1.0)

    def test_assets_prefix_not_detected_for_other_paths(self) -> None:
        f = self._features_for_path("res/raw/payload.dat")
        self.assertEqual(f["object_path_is_asset"], 0.0)

    def test_lib_so_detected(self) -> None:
        f = self._features_for_path("lib/arm64-v8a/libfoo.so")
        self.assertEqual(f["object_path_is_lib_so"], 1.0)

    def test_lib_so_not_detected_for_dex(self) -> None:
        f = self._features_for_path("classes.dex")
        self.assertEqual(f["object_path_is_lib_so"], 0.0)


class ConfigFlagsTests(unittest.TestCase):
    """Config booleans must toggle feature inclusion without crashing."""

    def test_include_flags_do_not_crash(self) -> None:
        cfg = DexStructuralFeatureConfig(
            include_header_magic=False,
            include_map_list_hints=False,
            include_alignment_hints=False,
            include_string_ids_sanity=False,
        )
        features = extract_region_structural_features(
            raw_bytes=DEX_MAGIC + b"\x00" * 4096,
            object_path="classes.dex",
            offset_start=0, offset_end=4096,
            config=cfg,
        )
        # Path-based features are always emitted regardless of config.
        self.assertIn("nearby_zip_local_header", features)
        self.assertIn("object_path_is_dex_like", features)
        self.assertIn("object_path_is_asset", features)
        self.assertIn("object_path_is_lib_so", features)


if __name__ == "__main__":
    unittest.main()
