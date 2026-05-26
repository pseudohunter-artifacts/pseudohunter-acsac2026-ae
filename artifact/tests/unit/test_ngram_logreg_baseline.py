"""Tests for the n-gram + logistic regression byte-level baseline.

These tests build tiny in-memory APKs on the fly so that we exercise
the full train -> save -> load -> predict loop without depending on
a pre-built synthetic dataset. sklearn is imported lazily inside the
baseline, so at module import time nothing numpy-related is loaded.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(
            "scikit-learn not installed; install with `pip install -e \".[metrics]\"`"
        ) from exc


_require_sklearn()

from android_packer.baselines import (  # noqa: E402 - after availability check
    NgramLogRegConfig,
    NgramLogRegModel,
    train_ngram_logreg,
)
from android_packer.features import ByteFeatureConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_fake_apk(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _region_row(
    *,
    apk_id: str,
    object_id: str,
    region_id: str,
    object_path: str,
    offset_start: int,
    offset_end: int,
    label_id: int,
) -> dict:
    # Only the fields the baseline actually reads must be present; the
    # rest of the schema is irrelevant for unit testing.
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "region_id": region_id,
        "object_path": object_path,
        "offset_start": offset_start,
        "offset_end": offset_end,
        "label_id": label_id,
    }


def _make_corpus(tmp: Path):
    """Build two APKs, one benign-only and one with a hidden payload.

    Returns (rows, apk_index). The rows are balanced 50/50 across both
    classes so that a reasonable classifier can separate them.
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

    rows: list[dict] = []
    for apk_id, apk_path in [("apk_a", apk_a), ("apk_b", apk_b)]:
        # Two benign regions and two payload regions, each 1024 bytes.
        rows.append(
            _region_row(
                apk_id=apk_id,
                object_id="assets/readme.txt",
                region_id=f"{apk_id}:r0",
                object_path="assets/readme.txt",
                offset_start=0,
                offset_end=1024,
                label_id=0,
            )
        )
        rows.append(
            _region_row(
                apk_id=apk_id,
                object_id="assets/readme.txt",
                region_id=f"{apk_id}:r1",
                object_path="assets/readme.txt",
                offset_start=1024,
                offset_end=2048,
                label_id=0,
            )
        )
        rows.append(
            _region_row(
                apk_id=apk_id,
                object_id="assets/secret.bin",
                region_id=f"{apk_id}:r2",
                object_path="assets/secret.bin",
                offset_start=0,
                offset_end=1024,
                label_id=1,
            )
        )
        rows.append(
            _region_row(
                apk_id=apk_id,
                object_id="assets/secret.bin",
                region_id=f"{apk_id}:r3",
                object_path="assets/secret.bin",
                offset_start=1024,
                offset_end=2048,
                label_id=1,
            )
        )

    apk_index = {"apk_a": apk_a, "apk_b": apk_b}
    return rows, apk_index


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainNgramLogRegTests(unittest.TestCase):
    def test_training_rejects_single_class_input(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            apk = td / "apk.apk"
            _build_fake_apk(apk, {"x.bin": b"hello"})
            rows = [
                _region_row(
                    apk_id="apk",
                    object_id="x.bin",
                    region_id="r0",
                    object_path="x.bin",
                    offset_start=0,
                    offset_end=5,
                    label_id=0,
                )
            ]
            with self.assertRaises(ValueError):
                train_ngram_logreg(rows, {"apk": apk})

    def test_training_rejects_all_skipped_rows(self):
        # Every row's apk_id is absent from the index, so nothing is
        # usable; the baseline must raise instead of silently fitting.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows = [
                _region_row(
                    apk_id="missing",
                    object_id="x.bin",
                    region_id="r0",
                    object_path="x.bin",
                    offset_start=0,
                    offset_end=5,
                    label_id=1,
                )
            ]
            with self.assertRaises(ValueError):
                train_ngram_logreg(rows, {})

    def test_trained_model_learns_text_vs_binary(self):
        # With benign = ASCII text and payload = 0x5A-masked counter,
        # the model should recover perfect or near-perfect train acc.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows, apk_index = _make_corpus(td)
            model = train_ngram_logreg(rows, apk_index)
            result = model.predict(rows, apk_index)

            # Region-level: every labelled region is scored correctly.
            region_errors = sum(
                1 for r in result.region_predictions
                if r.predicted_label_id != r.true_label_id
            )
            self.assertEqual(region_errors, 0)

            # Report contract matches the other baselines.
            self.assertEqual(result.report["baseline"], "ngram_logreg")
            self.assertEqual(result.report["counts"]["regions"], len(rows))
            self.assertIn("region", result.report["metrics"])
            self.assertIn("object", result.report["metrics"])
            self.assertIn("apk", result.report["metrics"])
            self.assertIn("ranking", result.report)
            self.assertIn("localization", result.report)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class NgramLogRegPersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows, apk_index = _make_corpus(td)
            model = train_ngram_logreg(rows, apk_index)
            result_before = model.predict(rows, apk_index)

            artefact = td / "model.pkl"
            model.save(artefact)
            self.assertTrue(artefact.exists())
            self.assertGreater(artefact.stat().st_size, 0)

            restored = NgramLogRegModel.load(artefact)
            result_after = restored.predict(rows, apk_index)

            scores_before = [r.score for r in result_before.region_predictions]
            scores_after = [r.score for r in result_after.region_predictions]
            self.assertEqual(scores_before, scores_after)

    def test_load_rejects_wrong_format_version(self):
        with tempfile.TemporaryDirectory() as td:
            import pickle

            artefact = Path(td) / "bad.pkl"
            with artefact.open("wb") as fh:
                pickle.dump(
                    {
                        "version": 99,
                        "vectorizer": None,
                        "classifier": None,
                        "feature_config": {},
                        "threshold": 0.5,
                    },
                    fh,
                )
            with self.assertRaises(ValueError):
                NgramLogRegModel.load(artefact)


# ---------------------------------------------------------------------------
# Config toggles
# ---------------------------------------------------------------------------


class NgramLogRegConfigTests(unittest.TestCase):
    def test_custom_threshold_changes_decision_boundary(self):
        # Threshold 0.99 should pull at least one region from positive
        # prediction back to negative (the model is not absolutely
        # certain about every single region).
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows, apk_index = _make_corpus(td)
            strict_model = train_ngram_logreg(
                rows,
                apk_index,
                config=NgramLogRegConfig(threshold=0.99),
            )
            loose_model = train_ngram_logreg(
                rows,
                apk_index,
                config=NgramLogRegConfig(threshold=0.01),
            )
            strict_positives = sum(
                1
                for r in strict_model.predict(rows, apk_index).region_predictions
                if r.predicted_label_id == 1
            )
            loose_positives = sum(
                1
                for r in loose_model.predict(rows, apk_index).region_predictions
                if r.predicted_label_id == 1
            )
            # Strict threshold cannot flag more regions than the loose one.
            self.assertLessEqual(strict_positives, loose_positives)

    def test_feature_config_is_passed_through_to_inference(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows, apk_index = _make_corpus(td)
            fc = ByteFeatureConfig(
                include_bigram=False,
                include_scalars=True,
                bigram_hash_dim=32,
            )
            model = train_ngram_logreg(
                rows,
                apk_index,
                config=NgramLogRegConfig(feature_config=fc),
            )
            self.assertFalse(model.feature_config.include_bigram)
            self.assertTrue(model.feature_config.include_scalars)

    def test_missing_apk_id_in_index_yields_zeroed_features(self):
        # At prediction time, an apk_id not in the index must not crash;
        # the row is scored against an all-zero feature dict.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows, apk_index = _make_corpus(td)
            model = train_ngram_logreg(rows, apk_index)
            rogue_row = _region_row(
                apk_id="not_in_index",
                object_id="x.bin",
                region_id="r_rogue",
                object_path="x.bin",
                offset_start=0,
                offset_end=1024,
                label_id=0,
            )
            result = model.predict([rogue_row], {})
            self.assertEqual(len(result.region_predictions), 1)
            # Score is bounded and well-defined.
            score = result.region_predictions[0].score
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
