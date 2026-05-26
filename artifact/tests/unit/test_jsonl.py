import json
import tempfile
import unittest
from pathlib import Path

from android_packer.utils.jsonl import read_jsonl, write_jsonl


class JsonlTests(unittest.TestCase):
    def test_write_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "rows.jsonl"
            rows = [{"a": 1}, {"b": "x"}, {"c": [1, 2, 3]}]

            written = write_jsonl(path, rows)

            self.assertEqual(written, 3)
            self.assertEqual(list(read_jsonl(path)), rows)

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

            self.assertEqual(list(read_jsonl(path)), [{"a": 1}, {"b": 2}])

    def test_malformed_line_raises_with_path_and_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text(
                '{"a": 1}\n{"b": 2}\nnot-json\n{"c": 3}\n',
                encoding="utf-8",
            )

            with self.assertRaises(json.JSONDecodeError) as ctx:
                list(read_jsonl(path))

            message = str(ctx.exception)
            self.assertIn("line=3", message)
            self.assertIn("broken.jsonl", message)


if __name__ == "__main__":
    unittest.main()
