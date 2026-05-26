"""Tests for the byte-level baseline CLIs and the shared apk-index helper."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("scikit-learn not installed") from exc


_require_sklearn()

from android_packer.cli import run_ngram_baseline as run_cli  # noqa: E402
from android_packer.cli import train_ngram_baseline as train_cli  # noqa: E402
from android_packer.cli._apk_index import build_apk_index  # noqa: E402


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _build_fake_apk(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


class BuildApkIndexTests(unittest.TestCase):
    def test_explicit_rows_are_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "idx.jsonl"
            _write_jsonl(
                path,
                [
                    {"apk_id": "a", "apk_path": "/tmp/a.apk"},
                    {"apk_id": "b", "apk_path": "/tmp/b.apk"},
                ],
            )
            index = build_apk_index(path)
            self.assertEqual(sorted(index), ["a", "b"])
            self.assertEqual(index["a"], Path("/tmp/a.apk"))

    def test_manifest_rows_are_coerced(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "generated_apk_id": "pack-0001",
                        "generated_apk_path": "/out/pack-0001.apk",
                        "transform_family": "xor",
                    }
                ],
            )
            index = build_apk_index(path)
            self.assertEqual(index["pack-0001"], Path("/out/pack-0001.apk"))

    def test_later_row_wins_on_duplicate_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "idx.jsonl"
            _write_jsonl(
                path,
                [
                    {"apk_id": "a", "apk_path": "/old.apk"},
                    {"apk_id": "a", "apk_path": "/new.apk"},
                ],
            )
            index = build_apk_index(path)
            self.assertEqual(index["a"], Path("/new.apk"))

    def test_rows_with_empty_ids_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "idx.jsonl"
            _write_jsonl(
                path,
                [
                    {"apk_path": "/nope.apk"},  # no id
                    {"apk_id": "", "apk_path": "/nope2.apk"},  # blank id
                    {"apk_id": "a", "apk_path": "/a.apk"},
                ],
            )
            index = build_apk_index(path)
            self.assertEqual(list(index), ["a"])


class NgramTrainRunCliEndToEndTests(unittest.TestCase):
    def _prepare_corpus(self, tmp: Path):
        """Create a balanced training corpus with two APKs and write its files.

        Returns paths to (region_labels.jsonl, apk_index.jsonl).
        """

        benign_text = (b"The quick brown fox jumps over the lazy dog. " * 64)[:2048]
        payload = bytes(((i * 73) ^ 0x5A) & 0xFF for i in range(2048))

        apk_a = tmp / "apk_a.apk"
        apk_b = tmp / "apk_b.apk"
        _build_fake_apk(
            apk_a,
            {"assets/readme.txt": benign_text, "assets/secret.bin": payload},
        )
        _build_fake_apk(
            apk_b,
            {"assets/readme.txt": benign_text, "assets/secret.bin": payload},
        )

        rows = []
        for apk_id in ("apk_a", "apk_b"):
            rows.append(
                {
                    "apk_id": apk_id,
                    "object_id": "assets/readme.txt",
                    "region_id": f"{apk_id}:r0",
                    "object_path": "assets/readme.txt",
                    "offset_start": 0,
                    "offset_end": 1024,
                    "label_id": 0,
                }
            )
            rows.append(
                {
                    "apk_id": apk_id,
                    "object_id": "assets/readme.txt",
                    "region_id": f"{apk_id}:r1",
                    "object_path": "assets/readme.txt",
                    "offset_start": 1024,
                    "offset_end": 2048,
                    "label_id": 0,
                }
            )
            rows.append(
                {
                    "apk_id": apk_id,
                    "object_id": "assets/secret.bin",
                    "region_id": f"{apk_id}:r2",
                    "object_path": "assets/secret.bin",
                    "offset_start": 0,
                    "offset_end": 1024,
                    "label_id": 1,
                }
            )
            rows.append(
                {
                    "apk_id": apk_id,
                    "object_id": "assets/secret.bin",
                    "region_id": f"{apk_id}:r3",
                    "object_path": "assets/secret.bin",
                    "offset_start": 1024,
                    "offset_end": 2048,
                    "label_id": 1,
                }
            )

        labels_path = tmp / "region_labels.jsonl"
        _write_jsonl(labels_path, rows)

        index_path = tmp / "apk_index.jsonl"
        _write_jsonl(
            index_path,
            [
                {"apk_id": "apk_a", "apk_path": str(apk_a)},
                {"apk_id": "apk_b", "apk_path": str(apk_b)},
            ],
        )
        return labels_path, index_path

    def test_train_then_run_happy_path(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            labels, index = self._prepare_corpus(td)

            model_path = td / "model.pkl"
            train_report = td / "train_report.json"
            rc_train = train_cli.main(
                [
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--model-out",
                    str(model_path),
                    "--report-out",
                    str(train_report),
                ]
            )
            self.assertEqual(rc_train, 0)
            self.assertTrue(model_path.exists())
            self.assertGreater(model_path.stat().st_size, 0)
            train_data = json.loads(train_report.read_text(encoding="utf-8"))
            self.assertEqual(train_data["baseline"], "ngram_logreg")
            self.assertEqual(train_data["apk_index_size"], 2)

            region_out = td / "region_pred.jsonl"
            object_out = td / "object_pred.jsonl"
            apk_out = td / "apk_pred.jsonl"
            report_out = td / "report.json"
            rc_run = run_cli.main(
                [
                    "--model",
                    str(model_path),
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--region-predictions-out",
                    str(region_out),
                    "--object-predictions-out",
                    str(object_out),
                    "--apk-predictions-out",
                    str(apk_out),
                    "--report-out",
                    str(report_out),
                ]
            )
            self.assertEqual(rc_run, 0)
            self.assertTrue(region_out.exists())
            self.assertTrue(object_out.exists())
            self.assertTrue(apk_out.exists())
            report_data = json.loads(report_out.read_text(encoding="utf-8"))
            self.assertEqual(report_data["baseline"], "ngram_logreg")
            self.assertEqual(report_data["counts"]["regions"], 8)

            # Rough sanity: at least one payload region must be flagged.
            region_rows = [
                json.loads(line)
                for line in region_out.read_text(encoding="utf-8").splitlines()
            ]
            positive_truth = [r for r in region_rows if r["true_label_id"] == 1]
            predicted_positive = [
                r for r in positive_truth if r["predicted_label_id"] == 1
            ]
            self.assertGreater(len(predicted_positive), 0)

    def test_empty_apk_index_rejects_training(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            labels = td / "labels.jsonl"
            labels.write_text("")  # empty labels, but that's fine
            index = td / "empty.jsonl"
            index.write_text("")  # empty index
            rc = train_cli.main(
                [
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--model-out",
                    str(td / "m.pkl"),
                    "--report-out",
                    str(td / "r.json"),
                ]
            )
            self.assertEqual(rc, 2)

    def test_run_cli_threshold_override(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            labels, index = self._prepare_corpus(td)
            model_path = td / "model.pkl"
            train_cli.main(
                [
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--model-out",
                    str(model_path),
                    "--report-out",
                    str(td / "train.json"),
                    "--threshold",
                    "0.5",
                ]
            )

            # Predict with a crushingly strict threshold; predicted
            # positive count must be less than or equal to the default.
            region_out = td / "region_pred.jsonl"
            rc = run_cli.main(
                [
                    "--model",
                    str(model_path),
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--region-predictions-out",
                    str(region_out),
                    "--object-predictions-out",
                    str(td / "o.jsonl"),
                    "--apk-predictions-out",
                    str(td / "a.jsonl"),
                    "--report-out",
                    str(td / "r.json"),
                    "--threshold",
                    "0.999",
                ]
            )
            self.assertEqual(rc, 0)
            region_rows = [
                json.loads(line)
                for line in region_out.read_text(encoding="utf-8").splitlines()
            ]
            strict_positives = sum(
                1 for r in region_rows if r["predicted_label_id"] == 1
            )

            region_out2 = td / "region_pred_loose.jsonl"
            run_cli.main(
                [
                    "--model",
                    str(model_path),
                    "--region-labels",
                    str(labels),
                    "--apk-index",
                    str(index),
                    "--region-predictions-out",
                    str(region_out2),
                    "--object-predictions-out",
                    str(td / "o2.jsonl"),
                    "--apk-predictions-out",
                    str(td / "a2.jsonl"),
                    "--report-out",
                    str(td / "r2.json"),
                    "--threshold",
                    "0.001",
                ]
            )
            region_rows_loose = [
                json.loads(line)
                for line in region_out2.read_text(encoding="utf-8").splitlines()
            ]
            loose_positives = sum(
                1 for r in region_rows_loose if r["predicted_label_id"] == 1
            )
            self.assertLessEqual(strict_positives, loose_positives)


if __name__ == "__main__":
    unittest.main()
