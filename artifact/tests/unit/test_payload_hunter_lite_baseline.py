"""Unit tests for PayloadHunter-Lite baseline (F-Lite-c).

Covers:

* Config defaults (train_mode, feature_dim, object loss weight).
* Empty-input early return.
* End-to-end same_set smoke test: train on 6 toy regions, predict,
  assert the report shape matches ``ngram_logreg`` field-for-field,
  and the trained model beats random on the toy data.
* holdout_transform orchestration with 2 transform families, 3
  regions each.
* ``_derive_package_name`` heuristic correctness.
* Byte-loader zero-fill path when ``include_byte_distribution=False``.
* Model save / load round-trip preserves feature names and restores
  inference-time parity.

Tests use ``pytest.importorskip("torch")`` so they are cleanly
skipped in environments without the ``[dl]`` optional extra.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def _mk_row(apk_id, object_id, region_id, offset, entropy, label,
            transform_family="xor", object_path="assets/payload.bin"):
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "region_id": region_id,
        "object_path": object_path,
        "offset_start": offset,
        "offset_end": offset + 64,
        "entropy": entropy,
        "label_id": label,
        "transform_family": transform_family,
    }


def _toy_corpus():
    """Build 12 regions across 4 objects in 2 APKs.

    Object `apk1/obj1` is all-positive (hidden payload); `apk1/obj2`
    is all-negative. `apk2/obj1` is all-positive; `apk2/obj2` all
    negative. Entropy is set so `entropy_delta_entry` has signal
    (positive objects have higher entropy than negative ones).
    """
    rows = []
    # APK 1 (transform xor)
    rows += [
        _mk_row("apk1", "obj1", "r1", 0, 7.9, 1, "xor"),
        _mk_row("apk1", "obj1", "r2", 64, 7.95, 1, "xor"),
        _mk_row("apk1", "obj1", "r3", 128, 7.88, 1, "xor"),
        _mk_row("apk1", "obj2", "r4", 0, 3.2, 0, "xor", "classes.dex"),
        _mk_row("apk1", "obj2", "r5", 64, 3.5, 0, "xor", "classes.dex"),
        _mk_row("apk1", "obj2", "r6", 128, 3.1, 0, "xor", "classes.dex"),
    ]
    # APK 2 (transform base64)
    rows += [
        _mk_row("apk2", "obj1", "r7", 0, 7.85, 1, "base64"),
        _mk_row("apk2", "obj1", "r8", 64, 7.9, 1, "base64"),
        _mk_row("apk2", "obj1", "r9", 128, 7.95, 1, "base64"),
        _mk_row("apk2", "obj2", "r10", 0, 3.3, 0, "base64", "classes.dex"),
        _mk_row("apk2", "obj2", "r11", 64, 3.4, 0, "base64", "classes.dex"),
        _mk_row("apk2", "obj2", "r12", 128, 3.2, 0, "base64", "classes.dex"),
    ]
    return rows


# ---------------------------------------------------------------------------
# Config-level tests (no torch needed)
# ---------------------------------------------------------------------------


def test_config_defaults():
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
    )

    cfg = PayloadHunterLiteConfig()
    assert cfg.train_mode == "same_set"
    assert cfg.scorer_config.feature_dim == 15
    assert cfg.aggregator_config.input_dim == 15
    assert cfg.threshold == 0.5
    assert 0.0 < cfg.object_loss_weight <= 1.0
    assert cfg.positive_class_weight >= 1.0


def test_derive_package_name_handles_versioned_apk_id():
    from android_packer.baselines.payload_hunter_lite import _derive_package_name

    # Synthetic task name pattern: pkg underscores, version int, hex
    # prefix, transform family.
    assert (
        _derive_package_name("org_fdroid_fdroid_1023052_985f5181_xor")
        == "org.fdroid.fdroid"
    )
    assert (
        _derive_package_name("com_termux_1002_e6265a57_base64")
        == "com.termux"
    )
    # No digit segment -> return as-is (fallback).
    assert _derive_package_name("weird_no_version") == "weird.no.version"


def test_unsupported_train_mode_raises():
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        run_payload_hunter_lite_baseline,
    )

    cfg = PayloadHunterLiteConfig(train_mode="bogus")
    with pytest.raises(ValueError, match="unsupported train_mode"):
        run_payload_hunter_lite_baseline([_mk_row("a", "o", "r", 0, 5.0, 1)], None, cfg)


def test_empty_input_returns_empty_report():
    from android_packer.baselines.payload_hunter_lite import (
        run_payload_hunter_lite_baseline,
    )

    result = run_payload_hunter_lite_baseline([], None)
    assert result.region_predictions == []
    assert result.object_predictions == []
    assert result.apk_predictions == []
    assert result.report["counts"] == {"regions": 0, "objects": 0, "apks": 0}


# ---------------------------------------------------------------------------
# End-to-end smoke tests (require torch)
# ---------------------------------------------------------------------------


def test_same_set_smoke_end_to_end():
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        run_payload_hunter_lite_baseline,
    )
    from android_packer.features.handcrafted import HandcraftedFeatureConfig

    rows = _toy_corpus()
    cfg = PayloadHunterLiteConfig(
        # Disable Group-B byte features (we have no bytes on toy rows).
        handcrafted_config=HandcraftedFeatureConfig(
            include_byte_distribution=False
        ),
        # Small net (7 dims: 1 raw + 3 deltas + 3 position).
        epochs=40,
        batch_size=8,
        learning_rate=5e-3,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="in-sample"):
        result = run_payload_hunter_lite_baseline(rows, apk_index=None, config=cfg)

    # Report shape mirrors ngram_logreg.
    r = result.report
    assert r["baseline"] == "payload_hunter_lite"
    assert set(r["metrics"].keys()) == {"region", "object", "apk"}
    assert r["counts"] == {"regions": 12, "objects": 4, "apks": 2}

    # In-sample same_set should easily beat random on this toy corpus.
    # AUROC >= 0.85 is a very loose lower bound; with 40 epochs on 12
    # rows the model should get very close to 1.0.
    assert r["metrics"]["region"]["auroc"] >= 0.85
    # Every positive object in the toy corpus is separable; AUROC 1.0
    # is expected but allow a cushion for seed variance.
    assert r["metrics"]["object"]["auroc"] >= 0.75


def test_holdout_transform_runs_leave_one_family_out():
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        run_payload_hunter_lite_baseline,
    )
    from android_packer.features.handcrafted import HandcraftedFeatureConfig

    rows = _toy_corpus()  # 2 transform families: xor + base64.
    cfg = PayloadHunterLiteConfig(
        handcrafted_config=HandcraftedFeatureConfig(
            include_byte_distribution=False
        ),
        train_mode="holdout_transform",
        epochs=10,
        batch_size=8,
        learning_rate=5e-3,
        random_state=0,
    )
    result = run_payload_hunter_lite_baseline(rows, apk_index=None, config=cfg)
    # The stitched report must cover all 12 test regions (leave-one
    # fold => each row is in the test set exactly once).
    assert result.report["counts"]["regions"] == 12
    assert result.report["folds"] == ["base64", "xor"]
    assert result.report["train_mode"] == "holdout_transform"


def test_holdout_package_derives_package_name_when_absent():
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        run_payload_hunter_lite_baseline,
    )
    from android_packer.features.handcrafted import HandcraftedFeatureConfig

    rows = _toy_corpus()
    # Strip any pre-populated package_name so the derive-from-apk_id
    # path is exercised. (The toy corpus doesn't populate it.)
    for r in rows:
        r.pop("package_name", None)
    # Rename apk_ids so they split into 2 derived packages.
    for r in rows:
        if r["apk_id"] == "apk1":
            r["apk_id"] = "org_fdroid_fdroid_100_abc_xor"
        else:
            r["apk_id"] = "com_termux_200_def_base64"

    cfg = PayloadHunterLiteConfig(
        handcrafted_config=HandcraftedFeatureConfig(
            include_byte_distribution=False
        ),
        train_mode="holdout_package",
        epochs=5,
        batch_size=8,
        random_state=0,
    )
    result = run_payload_hunter_lite_baseline(rows, apk_index=None, config=cfg)
    # Two derived package folds.
    assert sorted(result.report["folds"]) == ["com.termux", "org.fdroid.fdroid"]


def test_holdout_requires_at_least_two_groups():
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        run_payload_hunter_lite_baseline,
    )
    from android_packer.features.handcrafted import HandcraftedFeatureConfig

    rows = [_mk_row("apk1", "obj1", "r1", 0, 7.9, 1, "xor")]
    cfg = PayloadHunterLiteConfig(
        handcrafted_config=HandcraftedFeatureConfig(
            include_byte_distribution=False
        ),
        train_mode="holdout_transform",
    )
    with pytest.raises(ValueError, match=">= 2 distinct"):
        run_payload_hunter_lite_baseline(rows, apk_index=None, config=cfg)


def test_save_load_round_trip_preserves_predictions():
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        PayloadHunterLiteModel,
        train_payload_hunter_lite,
    )
    from android_packer.features.handcrafted import HandcraftedFeatureConfig

    rows = _toy_corpus()
    cfg = PayloadHunterLiteConfig(
        handcrafted_config=HandcraftedFeatureConfig(
            include_byte_distribution=False
        ),
        epochs=5,
        batch_size=8,
        random_state=0,
    )
    model = train_payload_hunter_lite(rows, apk_index=None, config=cfg)

    # Predict before save.
    result_before = model.predict(rows)
    scores_before = [r.score for r in result_before.region_predictions]

    # Round-trip through disk.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(path)
        loaded = PayloadHunterLiteModel.load(path)

    # Sanity: feature_names preserved.
    assert loaded.feature_names == model.feature_names

    # Predict after load.
    result_after = loaded.predict(rows)
    scores_after = [r.score for r in result_after.region_predictions]

    # Same scores (byte-identical within rounding).
    assert scores_before == scores_after


# ---------------------------------------------------------------------------
# Device-resolution tests (2026-05-01 GPU opt-in)
# ---------------------------------------------------------------------------


def test_config_device_default_is_auto():
    """``PayloadHunterLiteConfig.device`` defaults to 'auto' so the
    orchestrator picks CUDA when available without any user opt-in."""
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
    )

    cfg = PayloadHunterLiteConfig()
    assert cfg.device == "auto"


def test_resolve_device_auto_respects_cuda_availability():
    """``_resolve_device('auto')`` picks cuda iff available, else cpu."""
    torch = pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import _resolve_device

    dev = _resolve_device("auto")
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_resolve_device_cpu_forced():
    """Explicit 'cpu' resolves to cpu regardless of CUDA presence."""
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import _resolve_device

    dev = _resolve_device("cpu")
    assert dev.type == "cpu"


def test_resolve_device_unknown_falls_back_to_cpu_with_warning():
    """Garbage device strings warn and fall back to cpu instead of
    raising, so an orchestrator that picks up a stale device name
    keeps running."""
    pytest.importorskip("torch")
    import warnings as _warnings
    from android_packer.baselines.payload_hunter_lite import _resolve_device

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        dev = _resolve_device("elephant")
    assert dev.type == "cpu"
    assert any("elephant" in str(w.message) for w in caught)


def test_resolve_device_cuda_requested_without_cuda_warns():
    """If 'cuda' is requested but unavailable, fall back to cpu with
    a warning rather than crashing."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine; cannot test fallback.")
    import warnings as _warnings
    from android_packer.baselines.payload_hunter_lite import _resolve_device

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        dev = _resolve_device("cuda")
    assert dev.type == "cpu"
    assert any("cuda" in str(w.message).lower() for w in caught)


def test_save_always_writes_cpu_tensors():
    """Checkpoints must serialise CPU-side state_dicts so a CUDA-trained
    model reloads cleanly on a CPU-only machine."""
    torch = pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        train_payload_hunter_lite,
    )

    rows = _toy_corpus()
    cfg = PayloadHunterLiteConfig(
        epochs=2,
        batch_size=8,
        random_state=0,
        device="auto",
    )
    model = train_payload_hunter_lite(rows, apk_index=None, config=cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

    # Every tensor in both state dicts must live on cpu.
    for sd_key in ("region_scorer_state_dict", "object_aggregator_state_dict"):
        for tensor_name, tensor_value in payload[sd_key].items():
            assert tensor_value.device.type == "cpu", (
                f"{sd_key}.{tensor_name} was saved on {tensor_value.device!r}, "
                "expected cpu."
            )


def test_train_with_cpu_device_end_to_end():
    """Full end-to-end train+predict smoke with ``device='cpu'``
    explicitly, to guarantee the cpu branch keeps working for
    machines that opt out of GPU (or for reproducibility runs)."""
    pytest.importorskip("torch")
    from android_packer.baselines.payload_hunter_lite import (
        PayloadHunterLiteConfig,
        train_payload_hunter_lite,
    )

    rows = _toy_corpus()
    cfg = PayloadHunterLiteConfig(
        epochs=3,
        batch_size=8,
        random_state=0,
        device="cpu",
    )
    model = train_payload_hunter_lite(rows, apk_index=None, config=cfg)
    result = model.predict(rows)
    # The model should confidently separate the toy corpus (6 positive
    # regions have entropy ~7.9, 6 negative regions have ~3.2-3.5).
    scores_pos = [
        r.score for r in result.region_predictions if r.true_label_id == 1
    ]
    scores_neg = [
        r.score for r in result.region_predictions if r.true_label_id == 0
    ]
    # Average positive score strictly higher than average negative.
    assert sum(scores_pos) / len(scores_pos) > sum(scores_neg) / len(scores_neg)
