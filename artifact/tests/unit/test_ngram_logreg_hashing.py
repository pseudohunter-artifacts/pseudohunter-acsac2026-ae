"""Equivalence tests for HashingVectorizer vs DictVectorizer paths.

F0b introduced ``NgramLogRegConfig.use_hashing_vectorizer`` so that
``train_ngram_logreg`` can avoid the ~8 GB peak of ``DictVectorizer`` at
~160k regions. These tests make sure:

1. The default path (``FeatureHasher``) produces a usable trained model.
2. On small inputs the two vectorizers produce AUROC within a small
   tolerance, i.e. switching backends does not silently change the
   learned baseline.
3. The config flag round-trips through ``save`` / ``load``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.baselines import NgramLogRegConfig, train_ngram_logreg


def _build_fake_apk(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _row(apk_id: str, object_path: str, region_id: str, start: int, end: int, label: int) -> dict:
    return {
        "apk_id": apk_id,
        "object_id": object_path,
        "region_id": region_id,
        "object_path": object_path,
        "offset_start": start,
        "offset_end": end,
        "label_id": label,
    }


def _make_small_corpus(tmp: Path):
    """Two APKs, each with balanced benign/payload regions (100 rows)."""

    benign = (b"The quick brown fox jumps over the lazy dog. " * 64)[:2048]
    payload = bytes(((i * 73) ^ 0x5A) & 0xFF for i in range(2048))

    rows: list[dict] = []
    apk_index: dict = {}
    for i in range(25):
        apk_id = f"apk_{i}"
        apk_path = tmp / f"{apk_id}.apk"
        _build_fake_apk(
            apk_path,
            {"assets/readme.txt": benign, "assets/secret.bin": payload},
        )
        apk_index[apk_id] = apk_path
        # 2 benign regions + 2 payload regions per APK = 100 rows total.
        rows.append(_row(apk_id, "assets/readme.txt", f"{apk_id}:r0", 0, 1024, 0))
        rows.append(_row(apk_id, "assets/readme.txt", f"{apk_id}:r1", 1024, 2048, 0))
        rows.append(_row(apk_id, "assets/secret.bin", f"{apk_id}:r2", 0, 1024, 1))
        rows.append(_row(apk_id, "assets/secret.bin", f"{apk_id}:r3", 1024, 2048, 1))
    return rows, apk_index


class HashingVectorizerEquivalenceTests(unittest.TestCase):
    def test_default_config_uses_hashing_vectorizer(self):
        cfg = NgramLogRegConfig()
        # Documented contract: the default is hashing; flipping it off is
        # a deliberate, narrow-use override.
        self.assertTrue(cfg.use_hashing_vectorizer)
        self.assertEqual(cfg.hashing_n_features, 262144)

    def test_hashing_path_trains_and_reports_auroc(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows, apk_index = _make_small_corpus(tmp)
            cfg = NgramLogRegConfig()  # hashing is default
            model = train_ngram_logreg(rows, apk_index, config=cfg)
            result = model.predict(rows, apk_index)
            region_auroc = result.report["metrics"]["region"]["auroc"]
            # Text vs XOR-masked counter is trivially separable.
            self.assertIsNotNone(region_auroc)
            self.assertGreaterEqual(region_auroc, 0.95)

    def test_dict_and_hashing_agree_on_small_data(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows, apk_index = _make_small_corpus(tmp)

            hashing_cfg = NgramLogRegConfig(use_hashing_vectorizer=True)
            dict_cfg = NgramLogRegConfig(use_hashing_vectorizer=False)

            hashing_model = train_ngram_logreg(rows, apk_index, config=hashing_cfg)
            dict_model = train_ngram_logreg(rows, apk_index, config=dict_cfg)

            h_auroc = hashing_model.predict(rows, apk_index).report["metrics"]["region"]["auroc"]
            d_auroc = dict_model.predict(rows, apk_index).report["metrics"]["region"]["auroc"]

            # Both paths should achieve strong AUROC on this trivial task;
            # absolute difference must stay small (hash collisions are
            # negligible at 2**18 buckets vs ~1300 real features).
            self.assertIsNotNone(h_auroc)
            self.assertIsNotNone(d_auroc)
            self.assertLess(
                abs(h_auroc - d_auroc),
                0.02,
                f"AUROC drift too large: hashing={h_auroc}, dict={d_auroc}",
            )

    def test_hashing_model_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows, apk_index = _make_small_corpus(tmp)
            model = train_ngram_logreg(rows, apk_index, config=NgramLogRegConfig())
            # Scores before round-trip.
            before = model.predict(rows, apk_index)
            ckpt_path = tmp / "model.pkl"
            model.save(ckpt_path)
            from android_packer.baselines import NgramLogRegModel
            reloaded = NgramLogRegModel.load(ckpt_path)
            after = reloaded.predict(rows, apk_index)
            # Pairwise score match: pickle round-trip of FeatureHasher
            # + LogisticRegression must be exact.
            before_scores = [r.score for r in before.region_predictions]
            after_scores = [r.score for r in after.region_predictions]
            self.assertEqual(before_scores, after_scores)


if __name__ == "__main__":
    unittest.main()
