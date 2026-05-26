"""Unit tests for the minimal DEX item parser (F2b).

These tests lock in the parser's behavioural contract with
``training/pretrain_mlm.py``:

1. Legal benign DEX → returns ``DexItemSpan`` list covering at least
   header / string_ids / method_ids / class_defs / code_item /
   string_data.
2. Packed / mangled / truncated buffers → ``DexParseError`` (so F5 can
   exclude them from the MLM corpus automatically).
3. ``region_item_type_labels`` projects spans onto a byte window without
   going out of bounds.

The fixture (``_dex_fixtures.build_minimal_dex``) hand-assembles a DEX
buffer that exercises every branch of the parser — see that file for
layout rationale.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    # _dex_fixtures.py is adjacent to this file; the ``pythonpath = ["src"]``
    # setting in pyproject.toml does not cover tests themselves.
    sys.path.insert(0, str(_TESTS_DIR))

from _dex_fixtures import DEX_HEADER_SIZE, DEX_MAGIC_035, build_minimal_dex  # noqa: E402

from android_packer.features.dex_item_parser import (  # noqa: E402
    DEX_ITEM_TYPES,
    DexItemSpan,
    DexParseError,
    parse_dex_item_spans,
    region_item_type_labels,
)


_ITEM_INDEX = {name: idx for idx, name in enumerate(DEX_ITEM_TYPES)}


class ParseMinimalDexTests(unittest.TestCase):
    """Spec §3.2.1b: legal DEX must surface the required item types."""

    def setUp(self) -> None:
        self.dex_bytes, self.layout = build_minimal_dex()

    def test_returns_nonempty_sorted_spans(self) -> None:
        spans = parse_dex_item_spans(self.dex_bytes)
        self.assertGreater(len(spans), 0)
        offsets = [span.offset for span in spans]
        self.assertEqual(offsets, sorted(offsets),
                         "spans must be sorted by offset ascending")

    def test_header_span_is_first_and_covers_header(self) -> None:
        spans = parse_dex_item_spans(self.dex_bytes)
        header_spans = [s for s in spans if s.item_type == _ITEM_INDEX["header"]]
        self.assertEqual(len(header_spans), 1)
        self.assertEqual(header_spans[0].offset, 0)
        self.assertEqual(header_spans[0].size, DEX_HEADER_SIZE)

    def test_required_item_types_are_present(self) -> None:
        """Spec hard requirement: at least 5 distinct item types surface."""
        spans = parse_dex_item_spans(self.dex_bytes)
        present = {span.item_type for span in spans}
        for required in (
            "header",
            "string_ids",
            "method_ids",
            "class_defs",
            "code_item",
            "string_data",
        ):
            self.assertIn(_ITEM_INDEX[required], present,
                          f"missing item_type={required}")

    def test_fixed_size_section_extents_match_layout(self) -> None:
        """string_ids / method_ids / class_defs must report exact byte extents."""
        spans = parse_dex_item_spans(self.dex_bytes)
        by_type = {s.item_type: s for s in spans if s.item_type != _ITEM_INDEX["other"]}
        self.assertEqual(
            by_type[_ITEM_INDEX["string_ids"]].offset, self.layout.string_ids_off
        )
        self.assertEqual(
            by_type[_ITEM_INDEX["string_ids"]].size,
            self.layout.string_ids_count * 4,
        )
        self.assertEqual(
            by_type[_ITEM_INDEX["method_ids"]].offset, self.layout.method_ids_off
        )
        self.assertEqual(
            by_type[_ITEM_INDEX["method_ids"]].size,
            self.layout.method_ids_count * 8,
        )
        self.assertEqual(
            by_type[_ITEM_INDEX["class_defs"]].offset, self.layout.class_defs_off
        )
        self.assertEqual(
            by_type[_ITEM_INDEX["class_defs"]].size,
            self.layout.class_defs_count * 32,
        )

    def test_string_data_span_covers_all_string_data_bytes(self) -> None:
        spans = parse_dex_item_spans(self.dex_bytes)
        [sd] = [s for s in spans if s.item_type == _ITEM_INDEX["string_data"]]
        self.assertEqual(sd.offset, self.layout.string_data_off)
        # Each string_data item = 1 byte len prefix + "sN" (2 bytes) + NUL
        expected_size = self.layout.string_data_count * 4
        self.assertEqual(sd.size, expected_size)

    def test_all_spans_stay_within_buffer(self) -> None:
        spans = parse_dex_item_spans(self.dex_bytes)
        buffer_len = len(self.dex_bytes)
        for span in spans:
            self.assertGreaterEqual(span.offset, 0)
            self.assertGreaterEqual(span.size, 0)
            self.assertLessEqual(span.offset + span.size, buffer_len)

    def test_accepts_bytearray_and_memoryview(self) -> None:
        spans_bytes = parse_dex_item_spans(self.dex_bytes)
        spans_ba = parse_dex_item_spans(bytearray(self.dex_bytes))
        spans_mv = parse_dex_item_spans(memoryview(self.dex_bytes))
        self.assertEqual(spans_bytes, spans_ba)
        self.assertEqual(spans_bytes, spans_mv)


class ParseRejectionTests(unittest.TestCase):
    """Spec §3.2.1b: malformed / non-benign buffers must raise DexParseError."""

    def test_too_short_for_header(self) -> None:
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(b"")
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(b"\x00" * 64)  # shorter than 112 bytes

    def test_missing_magic(self) -> None:
        dex, _layout = build_minimal_dex()
        bad = bytearray(dex)
        bad[0:8] = b"PNG\r\n\x1a\n\x00"
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(bytes(bad))

    def test_unknown_magic_version(self) -> None:
        dex, _layout = build_minimal_dex()
        bad = bytearray(dex)
        bad[4:8] = b"099\x00"  # not in 035..039
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(bytes(bad))

    def test_all_zero_bytes(self) -> None:
        """The typical shape of a region that's been signature-stripped to 0."""
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(b"\x00" * 4096)

    def test_truncated_below_file_size(self) -> None:
        dex, _layout = build_minimal_dex()
        truncated = dex[: len(dex) - 16]  # drop part of map_list
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(truncated)

    def test_map_off_out_of_range(self) -> None:
        dex, layout = build_minimal_dex()
        import struct as _struct
        bad = bytearray(dex)
        # Set map_off (at 0x34) to something past EOF.
        _struct.pack_into("<I", bad, 0x34, len(dex) + 1024)
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(bytes(bad))

    def test_header_size_mismatch(self) -> None:
        dex, _layout = build_minimal_dex()
        import struct as _struct
        bad = bytearray(dex)
        _struct.pack_into("<I", bad, 0x24, 0x80)  # wrong header_size
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(bytes(bad))

    def test_random_bytes_with_dex_prefix(self) -> None:
        """Rejecting a region that *starts* with DEX magic but has random tail."""
        data = DEX_MAGIC_035 + b"\x99" * 4096
        with self.assertRaises(DexParseError):
            parse_dex_item_spans(data)


class RegionItemTypeLabelsTests(unittest.TestCase):
    """Spec §3.2.1b: region_item_type_labels projects spans to per-byte labels."""

    def setUp(self) -> None:
        self.dex_bytes, self.layout = build_minimal_dex()
        self.spans = parse_dex_item_spans(self.dex_bytes)

    def test_labels_length_matches_region(self) -> None:
        labels = region_item_type_labels(self.spans, region_offset=0, region_length=128)
        self.assertEqual(len(labels), 128)

    def test_zero_length_region_returns_empty_list(self) -> None:
        labels = region_item_type_labels(self.spans, region_offset=0, region_length=0)
        self.assertEqual(labels, [])

    def test_negative_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            region_item_type_labels(self.spans, region_offset=0, region_length=-1)

    def test_header_bytes_labelled_as_header(self) -> None:
        labels = region_item_type_labels(self.spans, region_offset=0,
                                         region_length=DEX_HEADER_SIZE)
        self.assertTrue(all(lbl == _ITEM_INDEX["header"] for lbl in labels))

    def test_uncovered_bytes_labelled_as_other(self) -> None:
        # A region past end-of-DEX is entirely uncovered.
        far_offset = self.layout.total_size + 10_000
        labels = region_item_type_labels(self.spans, region_offset=far_offset,
                                         region_length=64)
        self.assertTrue(all(lbl == _ITEM_INDEX["other"] for lbl in labels))

    def test_region_straddling_header_and_string_ids(self) -> None:
        region_len = self.layout.string_ids_off + self.layout.string_ids_count * 4 + 4
        labels = region_item_type_labels(self.spans, region_offset=0,
                                         region_length=region_len)
        self.assertEqual(len(labels), region_len)
        # Header prefix
        self.assertTrue(all(lbl == _ITEM_INDEX["header"]
                            for lbl in labels[:DEX_HEADER_SIZE]))
        # string_ids middle
        ids_start = self.layout.string_ids_off
        ids_end = ids_start + self.layout.string_ids_count * 4
        self.assertTrue(all(lbl == _ITEM_INDEX["string_ids"]
                            for lbl in labels[ids_start:ids_end]))

    def test_region_on_non_dex_bytes_all_other(self) -> None:
        labels = region_item_type_labels(spans=[], region_offset=0, region_length=32)
        self.assertTrue(all(lbl == _ITEM_INDEX["other"] for lbl in labels))


class ItemTypeVocabularyStabilityTests(unittest.TestCase):
    """Spec: DEX_ITEM_TYPES order is frozen after the first checkpoint ships.

    If this test fails you must bump the auxiliary head's ``num_classes``
    in every downstream checkpoint *and* update this list, not the other
    way around.
    """

    def test_vocabulary_order_is_stable(self) -> None:
        expected = (
            "header",
            "string_ids",
            "type_ids",
            "proto_ids",
            "field_ids",
            "method_ids",
            "class_defs",
            "code_item",
            "string_data",
            "other",
        )
        self.assertEqual(DEX_ITEM_TYPES, expected)

    def test_other_is_last(self) -> None:
        self.assertEqual(DEX_ITEM_TYPES[-1], "other")


class DexItemSpanDataclassTests(unittest.TestCase):
    def test_is_frozen(self) -> None:
        span = DexItemSpan(offset=0, size=10, item_type=_ITEM_INDEX["header"])
        with self.assertRaises(Exception):
            span.offset = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
