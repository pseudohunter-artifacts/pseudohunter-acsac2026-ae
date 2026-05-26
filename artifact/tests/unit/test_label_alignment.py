import unittest

from android_packer.labeling import build_training_labels
from android_packer.labeling.alignment import interval_iou, interval_overlap


class LabelAlignmentTests(unittest.TestCase):
    def test_interval_math(self):
        self.assertEqual(interval_overlap(0, 10, 5, 15), 5)
        self.assertEqual(interval_overlap(0, 5, 5, 10), 0)
        self.assertAlmostEqual(interval_iou(0, 10, 5, 15), 5 / 15)

    def test_build_region_object_and_apk_labels(self):
        regions = [
            _region("apk1", "obj1", "r0", "assets/payload.bin", 0, 4),
            _region("apk1", "obj1", "r1", "assets/payload.bin", 4, 8),
            _region("apk1", "obj2", "r2", "assets/benign.bin", 0, 4),
        ]
        labels = [
            {
                "apk_id": "apk1",
                "object_path": "assets/payload.bin",
                "offset_start": 2,
                "offset_end": 6,
                "label": "hidden_executable_payload",
                "transform_family": "xor",
                "payload_sha256": "a" * 64,
            }
        ]

        result = build_training_labels(regions, labels)

        self.assertEqual([row.label_id for row in result.region_labels], [1, 1, 0])
        self.assertEqual(result.region_labels[0].overlap_bytes, 2)
        self.assertEqual(result.region_labels[0].overlap_ratio, 0.5)
        self.assertEqual(result.object_labels[0].label_id, 1)
        self.assertEqual(result.object_labels[0].positive_region_count, 2)
        self.assertEqual(result.object_labels[1].label_id, 0)
        self.assertEqual(result.apk_labels[0].label_id, 1)
        self.assertEqual(result.apk_labels[0].positive_object_count, 1)

    def test_min_overlap_ratio_can_filter_partial_regions(self):
        regions = [
            _region("apk1", "obj1", "r0", "assets/payload.bin", 0, 10),
            _region("apk1", "obj1", "r1", "assets/payload.bin", 10, 20),
        ]
        labels = [
            {
                "apk_id": "apk1",
                "object_path": "assets/payload.bin",
                "offset_start": 9,
                "offset_end": 12,
                "label": "hidden_executable_payload",
                "transform_family": "split_xor",
                "payload_sha256": "b" * 64,
            }
        ]

        result = build_training_labels(regions, labels, min_overlap_ratio=0.25)

        self.assertEqual([row.label_id for row in result.region_labels], [0, 0])
        self.assertEqual(result.object_labels[0].label_id, 1)
        self.assertEqual(result.apk_labels[0].label_id, 1)

    def test_overlap_bytes_uses_union_when_payload_intervals_overlap(self):
        regions = [_region("apk1", "obj1", "r0", "assets/payload.bin", 0, 10)]
        labels = [
            {
                "apk_id": "apk1",
                "object_path": "assets/payload.bin",
                "offset_start": 0,
                "offset_end": 6,
                "label": "hidden_executable_payload",
                "transform_family": "xor",
                "payload_sha256": "a" * 64,
            },
            {
                "apk_id": "apk1",
                "object_path": "assets/payload.bin",
                "offset_start": 4,
                "offset_end": 10,
                "label": "hidden_executable_payload",
                "transform_family": "xor",
                "payload_sha256": "a" * 64,
            },
        ]

        result = build_training_labels(regions, labels)

        # The two payload intervals overlap on [4, 6); the union covers the
        # full region, so ``overlap_bytes`` must equal ``region_size`` (10)
        # rather than 6 + 6 = 12 clipped back to 10.
        self.assertEqual(result.region_labels[0].overlap_bytes, 10)
        self.assertEqual(result.region_labels[0].overlap_ratio, 1.0)
        self.assertEqual(result.region_labels[0].matched_label_count, 2)


def _region(
    apk_id: str,
    object_id: str,
    region_id: str,
    object_path: str,
    start: int,
    end: int,
) -> dict:
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "region_id": region_id,
        "object_path": object_path,
        "object_type": "asset_blob",
        "offset_start": start,
        "offset_end": end,
        "size": end - start,
        "sha256": "f" * 64,
        "entropy": 1.0,
        "printable_ratio": 0.5,
    }


if __name__ == "__main__":
    unittest.main()
