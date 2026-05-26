import unittest

from android_packer.apkio.objects import ApkObject
from android_packer.regioning.windows import byte_entropy, iter_regions, printable_ratio


class RegioningTests(unittest.TestCase):
    def test_window_generation_includes_tail_once(self):
        metadata = ApkObject(
            apk_id="a" * 64,
            object_id="apk:000001",
            object_path="assets/payload.bin",
            object_type="asset_blob",
            size=10,
            sha256="b" * 64,
            depth=0,
            container_path="sample.apk",
            compression="stored",
            compressed_size=10,
        )
        regions = list(
            iter_regions(
                metadata,
                b"0123456789",
                window_size=4,
                stride=3,
                include_tail=True,
            )
        )
        self.assertEqual([(r.offset_start, r.offset_end) for r in regions], [(0, 4), (3, 7), (6, 10)])

    def test_entropy_and_printable_ratio(self):
        self.assertEqual(byte_entropy(b""), 0.0)
        self.assertAlmostEqual(byte_entropy(b"\x00\x00\xff\xff"), 1.0)
        self.assertEqual(printable_ratio(b"ABC\x00"), 0.75)


if __name__ == "__main__":
    unittest.main()
