"""Tests for dataset split builders."""

import unittest

from android_packer.splits import (
    DatasetSplit,
    SplitConfig,
    build_split,
    by_package_split,
    by_transform_split,
)


def _records() -> list[dict]:
    return [
        {"apk_id": "apk_xor_1", "transform_family": "xor", "package_name": "org.a"},
        {"apk_id": "apk_xor_2", "transform_family": "xor", "package_name": "org.b"},
        {"apk_id": "apk_base64_1", "transform_family": "base64", "package_name": "org.a"},
        {"apk_id": "apk_split_1", "transform_family": "split_xor", "package_name": "org.c"},
        {"apk_id": "apk_path_1", "transform_family": "path_randomized", "package_name": "org.d"},
    ]


class ByTransformSplitTests(unittest.TestCase):
    def test_seen_vs_unseen_packer(self):
        split = by_transform_split(
            _records(),
            train_families=("xor", "base64"),
            test_families=("split_xor",),
        )
        self.assertEqual(split.strategy, "by_transform")
        self.assertEqual(
            split.train,
            ("apk_base64_1", "apk_xor_1", "apk_xor_2"),
        )
        self.assertEqual(split.test, ("apk_split_1",))
        # ``path_randomized`` is in neither partition: it should land in
        # ``unassigned`` rather than silently disappearing.
        self.assertEqual(split.unassigned, ("apk_path_1",))
        self.assertEqual(split.val, ())

    def test_val_partition_is_supported(self):
        split = by_transform_split(
            _records(),
            train_families=("xor",),
            val_families=("base64",),
            test_families=("split_xor", "path_randomized"),
        )
        self.assertEqual(split.val, ("apk_base64_1",))
        self.assertEqual(split.test, ("apk_path_1", "apk_split_1"))
        self.assertEqual(split.unassigned, ())

    def test_overlapping_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            build_split(
                _records(),
                "by_transform",
                SplitConfig(
                    id_field="apk_id",
                    group_field="transform_family",
                    train_groups=("xor",),
                    test_groups=("xor",),
                ),
            )

    def test_missing_id_field_raises(self):
        with self.assertRaises(KeyError):
            by_transform_split(
                [{"transform_family": "xor"}],
                train_families=("xor",),
            )


class ByPackageSplitTests(unittest.TestCase):
    def test_packages_are_isolated_across_partitions(self):
        split = by_package_split(
            _records(),
            train_packages=("org.a",),
            val_packages=("org.b",),
            test_packages=("org.c", "org.d"),
        )
        self.assertEqual(
            split.train,
            ("apk_base64_1", "apk_xor_1"),
        )
        self.assertEqual(split.val, ("apk_xor_2",))
        self.assertEqual(split.test, ("apk_path_1", "apk_split_1"))
        self.assertEqual(split.unassigned, ())


class SerialisationTests(unittest.TestCase):
    def test_to_dict_round_trip(self):
        split = by_transform_split(
            _records(),
            train_families=("xor",),
            test_families=("base64",),
        )
        payload = split.to_dict()
        self.assertEqual(payload["strategy"], "by_transform")
        self.assertEqual(payload["train"], ["apk_xor_1", "apk_xor_2"])
        self.assertEqual(payload["test"], ["apk_base64_1"])
        # ``group_to_ids`` must be sorted + sliced deterministically.
        self.assertEqual(
            payload["group_to_ids"]["xor"],
            ["apk_xor_1", "apk_xor_2"],
        )

    def test_counts_reports_partition_sizes(self):
        split = by_transform_split(
            _records(),
            train_families=("xor",),
            test_families=("base64", "split_xor"),
        )
        self.assertEqual(
            split.counts(),
            {"train": 2, "val": 0, "test": 2, "unassigned": 1},
        )


if __name__ == "__main__":
    unittest.main()
