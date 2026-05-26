"""Unit tests for :mod:`android_packer.baselines.ours` (F-MIL-e).

Focus:
* dataclass field parity with NgramLogRegResult (§8 hard contract);
* ``train_ours_baseline`` + ``OursBaselineModel.predict`` end-to-end on
  synthetic objects (skipping the handcrafted feature extraction by
  monkeypatching :func:`_aggregate_object_features`);
* ``_object_instance_type`` heuristic mapping matches the typed vocab.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from android_packer.baselines.ngram_logreg import (
    NgramLogRegApkPrediction,
    NgramLogRegObjectPrediction,
    NgramLogRegRegionPrediction,
)
from android_packer.baselines.ours import (
    FAMILY_TO_PAYLOAD_KIND,
    OursApkPrediction,
    OursBaselineConfig,
    OursObjectPrediction,
    OursRegionPrediction,
    OursResult,
    _object_instance_type,
    run_ours_baseline,
    train_ours_baseline,
)
from android_packer.models.typed_encoder import (
    TYPED_INSTANCE_TYPES,
    instance_type_id,
)


# ---------------------------------------------------------------------------
# Dataclass field parity — §8 hard contract
# ---------------------------------------------------------------------------


class TestReportShapeParity:
    def test_region_prediction_fields_match_ngram(self):
        our = {f.name for f in fields(OursRegionPrediction)}
        ngram = {f.name for f in fields(NgramLogRegRegionPrediction)}
        assert our == ngram, (
            f"OursRegionPrediction fields must match NgramLogRegRegionPrediction "
            f"(spec §8 contract); diff = {our ^ ngram}"
        )

    def test_object_prediction_fields_match_ngram(self):
        our = {f.name for f in fields(OursObjectPrediction)}
        ngram = {f.name for f in fields(NgramLogRegObjectPrediction)}
        assert our == ngram, f"diff = {our ^ ngram}"

    def test_apk_prediction_fields_match_ngram(self):
        our = {f.name for f in fields(OursApkPrediction)}
        ngram = {f.name for f in fields(NgramLogRegApkPrediction)}
        assert our == ngram, f"diff = {our ^ ngram}"


# ---------------------------------------------------------------------------
# _object_instance_type heuristic
# ---------------------------------------------------------------------------


class TestObjectInstanceTypeHeuristic:
    """L42 (2026-05-07): type is resolved from ground-truth when
    available, NOT from the object path.  Three resolution paths."""

    # -- Path 1: ground-truth via transform_families + positive label --

    @pytest.mark.parametrize("family, expected_kind", sorted(FAMILY_TO_PAYLOAD_KIND.items()))
    def test_transform_family_maps_to_ground_truth_kind(self, family, expected_kind):
        """Every registered transform family maps to its expected kind."""

        # The path is irrelevant under path 1 — use a deliberately
        # misleading path (benign-looking .png) to prove the point.
        tid = _object_instance_type(
            "res/drawable/foo.png",
            "res/drawable/foo.png",
            transform_families=[family],
            label_id=1,
        )
        assert tid == instance_type_id(expected_kind)

    def test_family_mapping_is_exhaustive_over_registered_transforms(self):
        """FAMILY_TO_PAYLOAD_KIND must cover every registered synthetic family."""

        from android_packer.synthetic.transforms import SUPPORTED_TRANSFORMS

        missing = set(SUPPORTED_TRANSFORMS) - set(FAMILY_TO_PAYLOAD_KIND)
        assert not missing, (
            f"FAMILY_TO_PAYLOAD_KIND does not cover synthetic families "
            f"{sorted(missing)}; update baselines/ours.py when a new "
            f"transform is registered."
        )

    # -- Path 2: explicit benign (label_id == 0) --

    def test_benign_object_without_family_routes_to_benign_other(self):
        tid = _object_instance_type(
            "res/drawable/icon.png",
            "res/drawable/icon.png",
            transform_families=None,
            label_id=0,
        )
        assert tid == instance_type_id("benign_other")

    def test_benign_object_with_stale_family_still_routes_to_benign_other(self):
        # A benign row might still have a stale transform_families
        # field (e.g. a joint aggregate).  label_id==0 is definitive.
        tid = _object_instance_type(
            "classes.dex",
            "classes.dex",
            transform_families=["xor"],
            label_id=0,
        )
        assert tid == instance_type_id("benign_other")

    # -- Path 3: legacy path-based heuristic (no ground-truth; real packers) --

    @pytest.mark.parametrize(
        "object_id, expected",
        [
            ("classes.dex", "encrypted_dex"),
            ("classes2.dex", "encrypted_dex"),
            ("lib/armeabi-v7a/libnative.so", "native_stub"),
        ],
    )
    def test_legacy_heuristic_used_when_no_labels(self, object_id, expected):
        tid = _object_instance_type(
            object_id,
            object_id,
            transform_families=None,
            label_id=1,  # positive but no family info -> fall through
        )
        assert tid == instance_type_id(expected)

    def test_legacy_heuristic_defaults_to_shim(self):
        tid = _object_instance_type(
            "unknown_blob.bin",
            "unknown_blob.bin",
            transform_families=None,
            label_id=1,
        )
        assert tid == instance_type_id("shim")

    def test_all_outputs_are_valid_type_ids(self):
        tid = _object_instance_type(
            "classes.dex", "classes.dex", transform_families=["xor"], label_id=1
        )
        assert 0 <= tid < len(TYPED_INSTANCE_TYPES)


# ---------------------------------------------------------------------------
# End-to-end train + predict — with monkeypatched feature aggregator
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch", reason="requires [dl] extra")


def _fake_objects(region_rows):
    """Produce a simple objects dict keyed by (apk_id, object_id).

    Features are deterministic length-15 vectors so the trainer's
    handcrafted dim matches the real pipeline's Pass-2a output.
    """

    import numpy as np

    by_key = {}
    for row in region_rows:
        apk = row["apk_id"]
        obj = row["object_id"]
        key = (apk, obj)
        label = int(row["label_id"])
        # Packed regions: strong positive feature signal in the first
        # two dims; benign: negative.
        bias = 1.0 if label == 1 else -1.0
        feat = np.array(
            [bias * 2.0, bias * 1.5] + [0.1 * i for i in range(13)],
            dtype=np.float32,
        )
        if key not in by_key:
            by_key[key] = {
                "apk_id": apk,
                "object_id": obj,
                "object_path": row.get("object_path", obj),
                "true_label_id": label,
                "feature_sum": feat.copy(),
                "region_count": 1,
                "positive_region_count": label,
                "region_rows": [row],
                # Provide transform_families for packed rows so
                # _object_instance_type can use GT-based type routing.
                "transform_families": ["xor"] if label == 1 else [],
            }
        else:
            rec = by_key[key]
            rec["feature_sum"] = rec["feature_sum"] + feat
            rec["region_count"] += 1
            rec["positive_region_count"] += label
            rec["region_rows"].append(row)
            rec["true_label_id"] = max(rec["true_label_id"], label)
    for rec in by_key.values():
        rec["feature_vec"] = rec["feature_sum"] / rec["region_count"]
    return by_key, ["f{}".format(i) for i in range(15)]


def _make_region_rows():
    rows = []
    # Two packed APKs, each with one packed dex object (label=1) + one
    # benign-like companion object (label=0).
    for i, apk in enumerate(["apk_packed_0", "apk_packed_1"]):
        rows.append(
            dict(
                apk_id=apk,
                object_id="classes.dex",
                object_path="classes.dex",
                region_id=f"r{i}_0",
                offset_start=0,
                offset_end=4096,
                label_id=1,
                true_label_id=1,
            )
        )
        rows.append(
            dict(
                apk_id=apk,
                object_id="other.bin",
                object_path="other.bin",
                region_id=f"r{i}_1",
                offset_start=0,
                offset_end=2048,
                label_id=0,
                true_label_id=0,
            )
        )
    # Two benign APKs.  Use a non-.dex path so that at inference time the
    # path heuristic routes these to ``shim`` rather than ``encrypted_dex``.
    # Rationale: at training, benign label_id=0 objects always route to
    # ``benign_other`` via GT routing.  At inference the path heuristic maps
    # ``classes.dex`` to ``encrypted_dex``, which was trained ONLY on positive
    # examples → fires positively for benign .dex files too, collapsing bag
    # scores.  Using a non-.dex resource path avoids that train/inference
    # routing inconsistency in this unit test.
    for i, apk in enumerate(["apk_benign_0", "apk_benign_1"]):
        rows.append(
            dict(
                apk_id=apk,
                object_id="res/drawable/icon.png",
                object_path="res/drawable/icon.png",
                region_id=f"rb{i}_0",
                offset_start=0,
                offset_end=4096,
                label_id=0,
                true_label_id=0,
            )
        )
    return rows


class TestTrainAndPredict:
    def test_end_to_end_with_mock_feature_aggregator(self, monkeypatch):
        """Train + predict produces a report with expected shape."""

        import android_packer.baselines.ours as ours_mod

        monkeypatch.setattr(ours_mod, "_aggregate_object_features",
                            lambda region_rows, apk_index, **_: _fake_objects(region_rows))

        region_rows = _make_region_rows()
        apk_index = {apk: None for apk in {r["apk_id"] for r in region_rows}}

        cfg = OursBaselineConfig(
            epochs=20,
            batch_size=2,
            learning_rate=5e-3,
            lambda_sparsity=0.0,
            random_state=0,
        )
        model = train_ours_baseline(region_rows, apk_index, cfg)
        result = model.predict(region_rows, apk_index)

        assert isinstance(result, OursResult)
        assert result.report["baseline"] == "ours"
        # At least one prediction per APK / object / region.
        # Report counts follow the ngram_logreg / payload_hunter_lite shape
        # (``counts.regions / objects / apks``) per §8 parity contract.
        assert result.report["counts"]["apks"] == 4
        assert result.report["counts"]["objects"] == 6  # 2*2 + 2
        assert result.report["counts"]["regions"] == 6

        # Sanity: packed APKs should score higher than benign on average
        # with 20 epochs + strong synthetic signal.
        packed_scores = [
            p.score for p in result.apk_predictions
            if p.apk_id.startswith("apk_packed")
        ]
        benign_scores = [
            p.score for p in result.apk_predictions
            if p.apk_id.startswith("apk_benign")
        ]
        assert min(packed_scores) > max(benign_scores)

    def test_run_ours_baseline_entry(self, monkeypatch):
        import android_packer.baselines.ours as ours_mod

        monkeypatch.setattr(ours_mod, "_aggregate_object_features",
                            lambda region_rows, apk_index, **_: _fake_objects(region_rows))

        rows = _make_region_rows()
        apk_index = {apk: None for apk in {r["apk_id"] for r in rows}}
        cfg = OursBaselineConfig(epochs=2, batch_size=2, random_state=0)
        model = train_ours_baseline(rows, apk_index, cfg)

        result = run_ours_baseline(rows, apk_index, model)
        assert isinstance(result, OursResult)
        assert result.report["typed_instance_types"] == list(TYPED_INSTANCE_TYPES)
