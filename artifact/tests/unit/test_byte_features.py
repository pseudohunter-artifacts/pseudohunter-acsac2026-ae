"""Tests for the byte-level feature extractor and APK byte loader."""

from __future__ import annotations

import io
import math
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.features import (
    ByteFeatureConfig,
    ObjectByteLoader,
    extract_region_bytes,
    region_byte_features,
)


class ByteFeatureShapeTests(unittest.TestCase):
    def test_empty_region_is_all_zero(self):
        fv = region_byte_features(b"")
        # Every declared feature must be present in the order, and
        # every value must either be absent (implicit zero) or 0.0.
        self.assertGreater(len(fv.order), 0)
        for name in fv.order:
            self.assertEqual(fv.values.get(name, 0.0), 0.0)
        self.assertTrue(all(v == 0.0 for v in fv.to_dense()))

    def test_unigram_length_is_always_256(self):
        fv = region_byte_features(b"hello")
        unigram = [n for n in fv.order if n.startswith("u")]
        self.assertEqual(len(unigram), 256)

    def test_dense_respects_feature_order(self):
        fv = region_byte_features(b"ABC")
        order = fv.order
        dense = fv.to_dense()
        # Round-trip through to_dense with the same order must match.
        self.assertEqual(dense, fv.to_dense(order))
        # Projecting into a tiny custom layout drops unknown keys to 0.
        custom = ["u065", "u066", "u067", "not_a_feature"]
        projected = fv.to_dense(custom)
        # 'A' -> 0x41 = 65, 'B' -> 66, 'C' -> 67.
        self.assertAlmostEqual(projected[0], 1 / 3, places=5)
        self.assertAlmostEqual(projected[1], 1 / 3, places=5)
        self.assertAlmostEqual(projected[2], 1 / 3, places=5)
        self.assertEqual(projected[3], 0.0)


class UnigramAndEntropyTests(unittest.TestCase):
    def test_all_zeros_region(self):
        data = b"\x00" * 1024
        fv = region_byte_features(data)
        self.assertAlmostEqual(fv.values["u000"], 1.0)
        self.assertEqual(fv.values["s_entropy"], 0.0)
        self.assertEqual(fv.values["s_zero_ratio"], 1.0)
        self.assertEqual(fv.values["s_longest_zero_run_ratio"], 1.0)
        self.assertEqual(fv.values["s_printable_ratio"], 0.0)

    def test_uniform_random_region_has_high_entropy(self):
        data = bytes(range(256)) * 16  # every byte equally represented
        fv = region_byte_features(data)
        # Entropy of uniform distribution over 256 symbols == 8 bits.
        self.assertAlmostEqual(fv.values["s_entropy"], 8.0, places=3)
        # Duplicate block ratio should be near 1 because the pattern
        # repeats every 256 bytes and 16-byte blocks tile that cleanly.
        self.assertGreater(fv.values["s_duplicate_block_ratio"], 0.5)

    def test_printable_text_region(self):
        text = b"The quick brown fox jumps over the lazy dog." * 20
        fv = region_byte_features(text)
        self.assertGreater(fv.values["s_printable_ratio"], 0.9)
        self.assertEqual(fv.values["s_high_byte_ratio"], 0.0)
        # Entropy of English ASCII is well below 8.
        self.assertLess(fv.values["s_entropy"], 5.0)

    def test_high_byte_ratio_counts_non_ascii(self):
        data = bytes([0x80, 0xFF, 0x90, 0x7F, 0x30])
        fv = region_byte_features(data)
        # 3 of 5 bytes are >= 0x80.
        self.assertAlmostEqual(fv.values["s_high_byte_ratio"], 3 / 5, places=5)


class BigramHashTests(unittest.TestCase):
    def test_bigram_bucket_count_matches_hash_dim(self):
        cfg = ByteFeatureConfig(bigram_hash_dim=64)
        fv = region_byte_features(b"ABCDEFG", config=cfg)
        bucket_names = [n for n in fv.order if n.startswith("b")]
        self.assertEqual(len(bucket_names), 64)

    def test_bigram_hash_is_deterministic_across_instances(self):
        cfg = ByteFeatureConfig(bigram_hash_dim=256)
        a = region_byte_features(b"\x10\x20\x10\x20\x10\x20", config=cfg)
        b = region_byte_features(b"\x10\x20\x10\x20\x10\x20", config=cfg)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_bigram_short_region_safe(self):
        # Single byte: no bigrams, buckets all zero but order is stable.
        fv = region_byte_features(b"\x42")
        bucket_vals = [v for k, v in fv.values.items() if k.startswith("b")]
        self.assertEqual(sum(bucket_vals), 0.0)


class ConfigValidationTests(unittest.TestCase):
    def test_non_positive_bigram_dim_is_rejected(self):
        with self.assertRaises(ValueError):
            ByteFeatureConfig(bigram_hash_dim=0)

    def test_non_positive_entropy_chunk_is_rejected(self):
        with self.assertRaises(ValueError):
            ByteFeatureConfig(entropy_chunk_size=-1)

    def test_non_positive_duplicate_block_is_rejected(self):
        with self.assertRaises(ValueError):
            ByteFeatureConfig(duplicate_block_size=0)

    def test_disable_bigram_removes_all_buckets(self):
        cfg = ByteFeatureConfig(include_bigram=False)
        fv = region_byte_features(b"ABCDEF", config=cfg)
        self.assertFalse(any(n.startswith("b") for n in fv.order))

    def test_disable_scalars_removes_scalar_keys(self):
        cfg = ByteFeatureConfig(include_scalars=False)
        fv = region_byte_features(b"ABCDEF", config=cfg)
        self.assertFalse(any(n.startswith("s_") for n in fv.order))


class LengthScalarTests(unittest.TestCase):
    def test_length_log1p_is_monotonic(self):
        short = region_byte_features(b"x" * 16).values["s_length_log1p"]
        medium = region_byte_features(b"x" * 4096).values["s_length_log1p"]
        long = region_byte_features(b"x" * (1 << 16)).values["s_length_log1p"]
        self.assertLess(short, medium)
        self.assertLess(medium, long)


class ObjectByteLoaderTests(unittest.TestCase):
    def _make_apk(self, tmpdir: Path, members: dict[str, bytes]) -> Path:
        apk_path = tmpdir / "test.apk"
        with zipfile.ZipFile(apk_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return apk_path

    def test_slice_returns_region_window(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"assets/payload.bin": b"ABCDEFGHIJ"})
            loader = ObjectByteLoader()
            region = loader.region_bytes(apk, "assets/payload.bin", 2, 5)
            self.assertEqual(region, b"CDE")

    def test_second_call_hits_lru_cache(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"classes.dex": b"0123456789"})
            loader = ObjectByteLoader()
            loader.region_bytes(apk, "classes.dex", 0, 3)
            info1 = loader.cache_info()
            loader.region_bytes(apk, "classes.dex", 5, 10)
            info2 = loader.cache_info()
            # Only one (apk_path, object_path) pair was touched; the
            # second call must be a cache hit, not a miss.
            self.assertEqual(info2.misses, info1.misses)
            self.assertGreater(info2.hits, info1.hits)

    def test_out_of_range_offset_returns_truncated_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"assets/x.bin": b"12345"})
            loader = ObjectByteLoader()
            self.assertEqual(loader.region_bytes(apk, "assets/x.bin", 3, 99), b"45")
            self.assertEqual(loader.region_bytes(apk, "assets/x.bin", 99, 200), b"")

    def test_empty_window_returns_empty_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"assets/x.bin": b"abc"})
            loader = ObjectByteLoader()
            self.assertEqual(loader.region_bytes(apk, "assets/x.bin", 2, 2), b"")

    def test_cache_clear_drops_state(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"assets/x.bin": b"abc"})
            loader = ObjectByteLoader()
            loader.region_bytes(apk, "assets/x.bin", 0, 3)
            self.assertGreater(loader.cache_info().currsize, 0)
            loader.cache_clear()
            self.assertEqual(loader.cache_info().currsize, 0)

    def test_different_instances_do_not_share_cache(self):
        # Independence prevents cross-test bleed-over and lets callers
        # bound their working set explicitly.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"a.bin": b"hello"})
            a = ObjectByteLoader()
            b = ObjectByteLoader()
            a.region_bytes(apk, "a.bin", 0, 5)
            self.assertEqual(a.cache_info().currsize, 1)
            self.assertEqual(b.cache_info().currsize, 0)

    def test_extract_region_bytes_primitive_matches_loader(self):
        # The uncached primitive must produce identical bytes to the
        # loader on the same (apk, object, offset) triple; the cache
        # is only an efficiency wrapper.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = self._make_apk(td, {"assets/x.bin": b"0123456789"})
            loader = ObjectByteLoader()
            for start, end in [(0, 4), (3, 7), (9, 10), (0, 10)]:
                primitive = extract_region_bytes(apk, "assets/x.bin", start, end)
                cached = loader.region_bytes(apk, "assets/x.bin", start, end)
                self.assertEqual(primitive, cached)


if __name__ == "__main__":
    unittest.main()
