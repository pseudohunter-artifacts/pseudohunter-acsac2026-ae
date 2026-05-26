"""Unit tests for the byte_cnn baseline."""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import fields
from pathlib import Path

import pytest


def _write_apk(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "assets/payload.bin",
            bytes([250, 251, 252, 253]) * 64,
        )
        archive.writestr("classes.dex", b"dex\n035\x00" + b"A" * 256)


def _rows(apk_id: str = "apk1") -> list[dict]:
    return [
        {
            "apk_id": apk_id,
            "object_id": "payload",
            "region_id": "p0",
            "object_path": "assets/payload.bin",
            "offset_start": 0,
            "offset_end": 64,
            "label_id": 1,
            "transform_family": "xor",
        },
        {
            "apk_id": apk_id,
            "object_id": "payload",
            "region_id": "p1",
            "object_path": "assets/payload.bin",
            "offset_start": 64,
            "offset_end": 128,
            "label_id": 1,
            "transform_family": "xor",
        },
        {
            "apk_id": apk_id,
            "object_id": "classes",
            "region_id": "c0",
            "object_path": "classes.dex",
            "offset_start": 0,
            "offset_end": 64,
            "label_id": 0,
            "transform_family": "base64",
        },
        {
            "apk_id": apk_id,
            "object_id": "classes",
            "region_id": "c1",
            "object_path": "classes.dex",
            "offset_start": 64,
            "offset_end": 128,
            "label_id": 0,
            "transform_family": "base64",
        },
    ]


def _tiny_cfg():
    from android_packer.baselines.byte_cnn import ByteCnnBaselineConfig
    from android_packer.models.byte_cnn import ByteCnnRegionScorerConfig

    return ByteCnnBaselineConfig(
        model_config=ByteCnnRegionScorerConfig(
            max_length=64,
            embedding_dim=8,
            conv_channels=8,
            kernel_sizes=(3, 5),
            hidden_dim=16,
            dropout=0.0,
        ),
        epochs=3,
        batch_size=2,
        learning_rate=5e-3,
        positive_class_weight=2.0,
        random_state=0,
        device="cpu",
    )


def test_prediction_fields_match_ngram_schema():
    from android_packer.baselines.byte_cnn import (
        ByteCnnApkPrediction,
        ByteCnnObjectPrediction,
        ByteCnnRegionPrediction,
    )
    from android_packer.baselines.ngram_logreg import (
        NgramLogRegApkPrediction,
        NgramLogRegObjectPrediction,
        NgramLogRegRegionPrediction,
    )

    assert {f.name for f in fields(ByteCnnRegionPrediction)} == {
        f.name for f in fields(NgramLogRegRegionPrediction)
    }
    assert {f.name for f in fields(ByteCnnObjectPrediction)} == {
        f.name for f in fields(NgramLogRegObjectPrediction)
    }
    assert {f.name for f in fields(ByteCnnApkPrediction)} == {
        f.name for f in fields(NgramLogRegApkPrediction)
    }


def test_train_predict_and_save_load_round_trip():
    pytest.importorskip("torch")
    from android_packer.baselines.byte_cnn import ByteCnnModel, train_byte_cnn

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        apk = tmp / "toy.apk"
        _write_apk(apk)
        rows = _rows()
        apk_index = {"apk1": apk}
        cfg = _tiny_cfg()

        model = train_byte_cnn(rows, apk_index, cfg)
        before = model.predict(rows, apk_index)
        assert before.report["baseline"] == "byte_cnn"
        assert before.report["counts"] == {"regions": 4, "objects": 2, "apks": 1}
        assert set(before.report["metrics"].keys()) == {"region", "object", "apk"}

        path = tmp / "byte_cnn.pt"
        model.save(path)
        loaded = ByteCnnModel.load(path)
        after = loaded.predict(rows, apk_index)

        assert [r.score for r in before.region_predictions] == [
            r.score for r in after.region_predictions
        ]


def test_bad_train_mode_is_rejected():
    from android_packer.baselines.byte_cnn import (
        ByteCnnBaselineConfig,
        run_byte_cnn_baseline,
    )

    cfg = ByteCnnBaselineConfig(train_mode="bogus")
    with pytest.raises(ValueError, match="unsupported train_mode"):
        run_byte_cnn_baseline([], {}, cfg)


def test_predict_rejects_missing_apk_index_entry():
    pytest.importorskip("torch")
    from android_packer.baselines.byte_cnn import train_byte_cnn

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        apk = tmp / "toy.apk"
        _write_apk(apk)
        rows = _rows()
        model = train_byte_cnn(rows, {"apk1": apk}, _tiny_cfg())

        with pytest.raises(KeyError, match="missing from apk_index"):
            model.predict(_rows("missing_apk"), {"apk1": apk})


def test_report_includes_threshold_and_top_k_calibration_diagnostics():
    from android_packer.baselines.byte_cnn import (
        ByteCnnApkPrediction,
        ByteCnnObjectPrediction,
        ByteCnnRegionPrediction,
        _build_report,
    )

    region_predictions = [
        ByteCnnRegionPrediction("apk1", "obj_pos", "r1", "payload.bin", 0, 10, 0.91, 1, 1),
        ByteCnnRegionPrediction("apk1", "obj_fp", "r2", "benign.bin", 0, 10, 0.80, 1, 0),
        ByteCnnRegionPrediction("apk1", "obj_neg", "r3", "classes.dex", 0, 10, 0.20, 0, 0),
    ]
    object_predictions = [
        ByteCnnObjectPrediction("apk1", "obj_pos", "payload.bin", 0.91, 1, 1, 1, 1),
        ByteCnnObjectPrediction("apk1", "obj_fp", "benign.bin", 0.80, 1, 0, 1, 0),
        ByteCnnObjectPrediction("apk1", "obj_neg", "classes.dex", 0.20, 0, 0, 1, 0),
    ]
    apk_predictions = [
        ByteCnnApkPrediction("apk1", 0.91, 1, 1, 3, 1, 3, 1),
    ]

    report = _build_report(
        region_predictions,
        object_predictions,
        apk_predictions,
        threshold=0.5,
    )

    assert report["calibration"]["object"]["at_default_threshold"]["f1"] == 0.666667
    assert report["calibration"]["object"]["best_f1"]["threshold"] == 0.91
    assert report["calibration"]["object"]["best_f1"]["f1"] == 1.0
    assert report["calibration"]["object_top_k"]["1"]["precision"] == 1.0
    assert report["calibration"]["object_top_k"]["3"]["precision"] == 0.333333
    assert report["calibration"]["region"]["best_f1"]["threshold"] == 0.91
