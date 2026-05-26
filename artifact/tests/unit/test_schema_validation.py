"""Tests for optional JSON Schema validation."""

import tempfile
import unittest
from pathlib import Path

from android_packer.utils.jsonl import write_jsonl
from android_packer.utils.schema import (
    SchemaValidationError,
    iter_validation_errors,
    load_schema,
    schema_path,
    validate_jsonl,
    validate_record,
)


def _valid_region_row(**overrides) -> dict:
    row = {
        "apk_id": "apk1",
        "object_id": "classes.dex",
        "region_id": "classes.dex:r000000",
        "object_path": "classes.dex",
        "object_type": "dex",
        "offset_start": 0,
        "offset_end": 4096,
        "size": 4096,
        "sha256": "f" * 64,
        "entropy": 7.12,
        "printable_ratio": 0.31,
    }
    row.update(overrides)
    return row


class SchemaHelperTests(unittest.TestCase):
    def test_schema_path_accepts_bare_name(self):
        with_suffix = schema_path("region_metadata.schema.json")
        bare = schema_path("region_metadata")
        self.assertEqual(with_suffix, bare)
        self.assertTrue(bare.exists())

    def test_load_schema_returns_mapping(self):
        schema = load_schema("region_metadata")
        self.assertEqual(schema["title"], "Region Metadata")


class ValidateRecordTests(unittest.TestCase):
    def test_valid_region_passes(self):
        validate_record(_valid_region_row(), "region_metadata")

    def test_missing_required_field_mentions_field_name(self):
        row = _valid_region_row()
        row.pop("entropy")
        with self.assertRaises(SchemaValidationError) as ctx:
            validate_record(row, "region_metadata")
        self.assertIn("entropy", str(ctx.exception))

    def test_wrong_type_is_reported(self):
        row = _valid_region_row(entropy="high")  # should be number
        with self.assertRaises(SchemaValidationError) as ctx:
            validate_record(row, "region_metadata")
        self.assertIn("entropy", str(ctx.exception))

    def test_out_of_range_value_is_reported(self):
        row = _valid_region_row(printable_ratio=1.5)
        with self.assertRaises(SchemaValidationError):
            validate_record(row, "region_metadata")


class ValidateJsonlTests(unittest.TestCase):
    def test_valid_jsonl_reports_record_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.jsonl"
            write_jsonl(
                path,
                [
                    _valid_region_row(region_id="r0"),
                    _valid_region_row(region_id="r1"),
                ],
            )
            self.assertEqual(validate_jsonl(path, "region_metadata"), 2)

    def test_jsonl_with_bad_row_surfaces_first_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.jsonl"
            write_jsonl(
                path,
                [
                    _valid_region_row(region_id="r0"),
                    _valid_region_row(region_id="r1", size=0),
                ],
            )
            with self.assertRaises(SchemaValidationError):
                validate_jsonl(path, "region_metadata")

    def test_iter_validation_errors_reports_per_record_index(self):
        records = [
            _valid_region_row(region_id="r0"),
            _valid_region_row(region_id="r1", entropy=-1),
            _valid_region_row(region_id="r2"),
            _valid_region_row(region_id="r3", printable_ratio=2),
        ]
        errors = list(iter_validation_errors(records, "region_metadata"))
        self.assertEqual([index for index, _ in errors], [1, 3])


if __name__ == "__main__":
    unittest.main()
