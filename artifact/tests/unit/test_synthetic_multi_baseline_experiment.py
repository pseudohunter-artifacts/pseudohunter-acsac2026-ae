"""Integration test for the multi-baseline experiment runner."""

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

from android_packer.cli import run_synthetic_multi_baseline as runner  # noqa: E402


def _write_seed_apk(path: Path) -> None:
    """Build a tiny but realistic-looking APK for the synthetic packer.

    ``classes.dex`` has the DEX magic so iter_apk_objects recognises it
    as a code object; ``assets/readme.txt`` gives the packer a benign
    file to coexist with. Sizes are intentionally small so the mini
    regioning config (64-byte windows) still produces several regions.
    """

    dex_payload = b"dex\n035\x00" + (b"payload bytes " * 64)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.dex", dex_payload)
        archive.writestr("assets/readme.txt", b"benign text " * 16)


def _write_seed_manifest(root: Path, seeds: list[tuple[str, int, str]]) -> Path:
    seed_dir = root / "seeds"
    seed_dir.mkdir()
    entries = []
    for package_name, version_code, filename in seeds:
        apk_path = seed_dir / filename
        _write_seed_apk(apk_path)
        entries.append(
            {
                "package_name": package_name,
                "app_name": package_name.rsplit(".", 1)[-1],
                "version_name": "1.0",
                "version_code": version_code,
                "local_path": str(apk_path),
            }
        )
    manifest_path = root / "seed_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "entries": entries}),
        encoding="utf-8",
    )
    return manifest_path


def _write_config(root: Path, seed_manifest: Path, baselines: list[str]) -> Path:
    config = {
        "input": {"seed_manifest": str(seed_manifest)},
        "synthetic": {
            "transform_families": ["xor"],
            "asset_prefix": "assets/synthetic",
            "rng_seed": 0,
            "split_count": 2,
            # B1: opt out of payload-size-range floor so the test
            # fixture's sub-64KiB placeholder DEX still runs through.
            "enforce_payload_size_range": False,
        },
        "regioning": {
            "window_size": 64,
            "stride": 32,
            "min_region_size": 1,
            "include_tail": True,
            "max_depth": 1,
            "max_member_bytes": None,
        },
        "labeling": {
            "min_overlap_bytes": 1,
            "min_overlap_ratio": 0.0,
        },
        "baselines": {
            "enabled": baselines,
            "entropy": {
                "entropy_threshold": 1.0,
                "entropy_weight": 1.0,
                "nonprintable_weight": 0.0,
            },
            "sanity_rules": {"threshold": 1.0},
            "ngram_logreg": {
                "train_mode": "same_set",
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
                "random_state": 0,
                "threshold": 0.5,
                "bigram_hash_dim": 64,
            },
            "byte_cnn": {
                "train_mode": "same_set",
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 5e-3,
                "positive_class_weight": 2.0,
                "random_state": 0,
                "threshold": 0.5,
                "max_length": 64,
                "embedding_dim": 8,
                "conv_channels": 8,
                "kernel_sizes": [3, 5],
                "hidden_dim": 16,
                "dropout": 0.0,
            },
            "apkid": {
                "min_hits": 1,
                "include_aux_categories": False,
                "timeout_seconds": 30.0,
            },
            "ours": {
                # Tiny config so CI can finish in a few seconds.  The
                # Tier-A defaults (bag supervision + holdout_transform)
                # are still the production choice.  See
                # ``docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md``
                # §7.3 / §7.4.
                "train_mode": "same_set",
                "supervision_mode": "bag",
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 5e-3,
                "lambda_diff_pseudo": 0.0,
                "lambda_sparsity": 0.0,
                "train_max_bag_size": 16,
                "train_min_positive_fraction": 0.05,
                "random_state": 0,
            },
            "mil_byte_cnn_fusion": {
                "mil_weight": 0.5,
                "byte_cnn_weight": 0.5,
                "threshold": 0.5,
                "score_transform": "identity",
            },
        },
        "outputs": {
            "output_root": str(root / "outputs"),
            "synthetic_generated_dir": str(root / "generated_apks"),
            "synthetic_manifest_dir": str(root / "synthetic_manifests"),
            "synthetic_label_dir": str(root / "synthetic_labels"),
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class MultiBaselineRunnerTests(unittest.TestCase):
    def test_runs_three_baselines_and_writes_combined_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [("org.example.one", 1, "one.apk")],
            )
            # Run xor + base64 on one seed APK with entropy +
            # sanity_rules + ngram_logreg. APKiD intentionally excluded
            # so the test does not depend on optional deps.
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "sanity_rules", "ngram_logreg"],
            )
            rc = runner.main(
                [
                    "--config",
                    str(config_path),
                    "--transforms",
                    "xor",
                    "base64",
                ]
            )
            self.assertEqual(rc, 0)

            out_root = tmp / "outputs"
            summary = json.loads((out_root / "summary.json").read_text("utf-8"))
            manifest_out = json.loads(
                (out_root / "experiment_manifest.json").read_text("utf-8")
            )

            self.assertEqual(manifest_out["failure_count"], 0)
            self.assertEqual(manifest_out["task_count"], 2)
            self.assertEqual(
                sorted(summary["baselines"]),
                ["entropy", "ngram_logreg", "sanity_rules"],
            )

            for baseline_name in ("entropy", "sanity_rules", "ngram_logreg"):
                block = summary["baselines"][baseline_name]
                self.assertEqual(block["successful_task_count"], 2)
                # Overall + per-transform aggregates exist.
                self.assertIn("overall", block)
                self.assertEqual(
                    sorted(block["by_transform"]),
                    ["base64", "xor"],
                )
                overall = block["overall"]
                for level in ("region", "object", "apk"):
                    level_metrics = overall["metrics"][level]
                    for key in ("precision", "recall", "f1", "auroc", "auprc"):
                        self.assertIn(key, level_metrics)
                self.assertIn("ranking", overall)
                self.assertIn("localization", overall)

            # Ngram warning must appear so readers know results are in-sample.
            warnings = summary.get("warnings", [])
            self.assertTrue(
                any("same_set" in w for w in warnings),
                f"expected same_set warning in {warnings}",
            )

            # Cross-baseline artefact paths all land under each task dir.
            for task in manifest_out["tasks"]:
                task_baselines = task.get("baselines", {})
                self.assertEqual(
                    sorted(task_baselines),
                    ["entropy", "ngram_logreg", "sanity_rules"],
                )
                for baseline_name in task_baselines:
                    self.assertEqual(task_baselines[baseline_name]["status"], "ok")

    def test_include_benign_apks_makes_apk_auroc_defined(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.benign_a", 1, "a.apk"),
                    ("org.example.benign_b", 1, "b.apk"),
                ],
            )
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "sanity_rules"],
            )
            rc = runner.main(
                [
                    "--config",
                    str(config_path),
                    "--transforms",
                    "xor",
                    "--include-benign-apks",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)

            out_root = tmp / "outputs"
            summary = json.loads((out_root / "summary.json").read_text("utf-8"))
            manifest_out = json.loads(
                (out_root / "experiment_manifest.json").read_text("utf-8")
            )

            self.assertEqual(manifest_out["benign_control_count"], 1)
            self.assertEqual(manifest_out["task_count"], 3)
            benign_tasks = [
                task
                for task in manifest_out["tasks"]
                if task.get("task_kind") == "benign_control"
            ]
            self.assertEqual(len(benign_tasks), 1)
            self.assertTrue(benign_tasks[0]["evaluation_only"])
            self.assertEqual(benign_tasks[0]["baselines"]["entropy"]["status"], "ok")

            for baseline_name in ("entropy", "sanity_rules"):
                apk_metrics = summary["baselines"][baseline_name]["overall"]["metrics"]["apk"]
                self.assertEqual(apk_metrics["positives"], 2)
                self.assertEqual(apk_metrics["support"], 3)
                self.assertIsNotNone(apk_metrics["auroc"])
                self.assertIsNotNone(apk_metrics["auprc"])

    def test_ngram_prepare_skips_when_train_rows_exceed_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = {
                "status": "ok",
                "task_name": "task_a",
                "generated_apk_path": str(tmp / "a.apk"),
                "transform_family": "xor",
                "_region_label_rows": [
                    {
                        "apk_id": "seed_a",
                        "object_path": "classes.dex",
                        "object_id": "classes.dex",
                        "offset_start": 0,
                        "offset_end": 1,
                        "label_id": 0,
                    }
                ],
            }
            warnings: list[str] = []

            prepared = runner._prepare_ngram_model(
                task_states=[state],
                config={"max_train_rows": 0},
                output_root=tmp,
                warnings=warnings,
            )

            self.assertTrue(prepared["skipped"])
            self.assertEqual(prepared["skip_reason"], "max_train_rows_exceeded")
            self.assertTrue(any("ngram_logreg skipped" in w for w in warnings))

    def test_lite_prepare_skips_when_train_rows_exceed_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = {
                "status": "ok",
                "task_name": "task_a",
                "generated_apk_path": str(tmp / "a.apk"),
                "transform_family": "xor",
                "_region_label_rows": [
                    {
                        "apk_id": "seed_a",
                        "object_path": "classes.dex",
                        "object_id": "classes.dex",
                        "offset_start": 0,
                        "offset_end": 1,
                        "label_id": 0,
                    }
                ],
            }
            warnings: list[str] = []

            prepared = runner._prepare_payload_hunter_lite_model(
                task_states=[state],
                config={"max_train_rows": 0},
                output_root=tmp,
                warnings=warnings,
            )

            self.assertTrue(prepared["skipped"])
            self.assertEqual(prepared["skip_reason"], "max_train_rows_exceeded")
            self.assertTrue(
                any("payload_hunter_lite skipped" in w for w in warnings)
            )

    def test_byte_cnn_train_row_sampling_is_deterministic_and_stratified(self):
        rows = []
        for family in ("xor", "base64", "split_xor"):
            for label in (0, 1):
                for i in range(8):
                    rows.append(
                        {
                            "apk_id": f"apk_{family}",
                            "object_path": "classes.dex",
                            "object_id": "classes.dex",
                            "region_id": f"{family}_{label}_{i}",
                            "offset_start": i * 64,
                            "offset_end": i * 64 + 64,
                            "transform_family": family,
                            "label_id": label,
                        }
                    )

        first = runner._sample_byte_cnn_train_rows(
            rows,
            max_rows=12,
            min_positive_rows=5,
            group_key="transform_family",
            random_state=17,
        )
        second = runner._sample_byte_cnn_train_rows(
            rows,
            max_rows=12,
            min_positive_rows=5,
            group_key="transform_family",
            random_state=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertGreaterEqual(sum(row["label_id"] for row in first), 5)
        self.assertEqual({row["label_id"] for row in first}, {0, 1})
        self.assertGreaterEqual(
            len({row["transform_family"] for row in first}),
            2,
        )

    def test_byte_cnn_calibration_split_is_deterministic_and_non_empty(self):
        rows = []
        for family in ("xor", "base64"):
            for label in (0, 1):
                for i in range(12):
                    rows.append(
                        {
                            "apk_id": f"apk_{family}",
                            "object_path": "classes.dex",
                            "object_id": "classes.dex",
                            "region_id": f"{family}_{label}_{i}",
                            "offset_start": i * 64,
                            "offset_end": i * 64 + 64,
                            "transform_family": family,
                            "label_id": label,
                        }
                    )

        fit_a, cal_a = runner._split_byte_cnn_fit_and_calibration_rows(
            rows,
            validation_fraction=0.25,
            min_validation_rows=8,
            min_positive_rows=3,
            group_key="transform_family",
            random_state=19,
        )
        fit_b, cal_b = runner._split_byte_cnn_fit_and_calibration_rows(
            rows,
            validation_fraction=0.25,
            min_validation_rows=8,
            min_positive_rows=3,
            group_key="transform_family",
            random_state=19,
        )

        self.assertEqual(fit_a, fit_b)
        self.assertEqual(cal_a, cal_b)
        self.assertGreater(len(fit_a), 0)
        self.assertGreater(len(cal_a), 0)
        self.assertEqual(len(fit_a) + len(cal_a), len(rows))
        self.assertEqual({row["label_id"] for row in fit_a}, {0, 1})
        self.assertEqual({row["label_id"] for row in cal_a}, {0, 1})

    def test_apkid_is_skipped_when_package_missing(self):
        # Explicitly simulate an environment where APKiD cannot be
        # used, by patching ``_default_scan_fn`` to raise
        # ``ApkidNotInstalledError``. This keeps the test deterministic
        # regardless of whether apkid is actually pip-installed in the
        # test environment (F0c installed apkid 3.1.0, so the previous
        # "hope the package is missing" assumption no longer holds).
        from unittest.mock import patch
        from android_packer.baselines import ApkidNotInstalledError

        def _raise_not_installed() -> None:
            raise ApkidNotInstalledError("apkid not installed (simulated)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [("org.example.only", 1, "only.apk")],
            )
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "apkid"],
            )
            with patch(
                "android_packer.baselines.apkid._default_scan_fn",
                side_effect=_raise_not_installed,
            ):
                rc = runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--transforms",
                        "xor",
                    ]
                )
            self.assertEqual(rc, 0)

            out_root = tmp / "outputs"
            summary = json.loads((out_root / "summary.json").read_text("utf-8"))
            manifest_out = json.loads(
                (out_root / "experiment_manifest.json").read_text("utf-8")
            )

            # Entropy is healthy.
            self.assertEqual(
                summary["baselines"]["entropy"]["successful_task_count"], 1
            )
            # APKiD appears as a baseline key but contributed zero tasks.
            self.assertIn("apkid", summary["baselines"])
            # Warning mentions the skip reason.
            self.assertTrue(
                any("apkid skipped" in w for w in summary.get("warnings", [])),
                f"expected apkid-skip warning in {summary.get('warnings', [])}",
            )
            # Per-task status for apkid is 'skipped'.
            task = manifest_out["tasks"][0]
            self.assertEqual(task["baselines"]["apkid"]["status"], "skipped")

    def test_ours_dispatch_same_set_mode(self):
        """L43 + L45 runner wiring: Ours baseline runs end-to-end under
        same_set, honours ``supervision_mode="bag"``, and writes the
        three prediction tables to disk.  Exercises the
        ``_prepare_ours_model`` → scoring → summary path."""

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed; ours requires [dl] extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.ours_a", 1, "a.apk"),
                    ("org.example.ours_b", 1, "b.apk"),
                ],
            )
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "ours"],
            )
            rc = runner.main(
                ["--config", str(config_path), "--no-skip-existing"]
            )
            self.assertEqual(rc, 0)

            summary_path = tmp / "outputs" / "summary.json"
            manifest_path = tmp / "outputs" / "experiment_manifest.json"
            summary = json.loads(summary_path.read_text("utf-8"))
            manifest_out = json.loads(manifest_path.read_text("utf-8"))

            self.assertIn("ours", summary["baselines"])
            self.assertEqual(
                summary["baselines"]["ours"]["successful_task_count"],
                manifest_out["task_count"],
            )

            # Each task has per-baseline 'ok' status + the three
            # prediction JSONLs on disk.
            for task in manifest_out["tasks"]:
                self.assertEqual(task["baselines"]["ours"]["status"], "ok")
                for suffix in (
                    "ours_region_predictions_path",
                    "ours_object_predictions_path",
                    "ours_apk_predictions_path",
                ):
                    p = Path(task[suffix])
                    self.assertTrue(
                        p.exists(),
                        f"missing ours prediction artefact {p}",
                    )

    def test_byte_cnn_dispatch_same_set_mode(self):
        """byte_cnn runs end-to-end through the multi-baseline runner and
        writes region/object/APK prediction artefacts."""

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed; byte_cnn requires [dl] extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.byte_a", 1, "a.apk"),
                    ("org.example.byte_b", 1, "b.apk"),
                ],
            )
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "byte_cnn"],
            )
            config = json.loads(config_path.read_text("utf-8"))
            config["baselines"]["byte_cnn"].update(
                {
                    "calibration_mode": "fold_local_best_f1",
                    "calibration_target": "object",
                    "calibration_validation_fraction": 0.25,
                    "calibration_min_validation_rows": 2,
                    "calibration_min_positive_rows": 1,
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            rc = runner.main(
                ["--config", str(config_path), "--no-skip-existing"]
            )
            self.assertEqual(rc, 0)

            summary_path = tmp / "outputs" / "summary.json"
            manifest_path = tmp / "outputs" / "experiment_manifest.json"
            summary = json.loads(summary_path.read_text("utf-8"))
            manifest_out = json.loads(manifest_path.read_text("utf-8"))

            self.assertIn("byte_cnn", summary["baselines"])
            self.assertEqual(
                summary["baselines"]["byte_cnn"]["successful_task_count"],
                manifest_out["task_count"],
            )

            for task in manifest_out["tasks"]:
                self.assertEqual(task["baselines"]["byte_cnn"]["status"], "ok")
                report_path = Path(task["byte_cnn_report_path"])
                report = json.loads(report_path.read_text("utf-8"))
                self.assertEqual(report["calibration_mode"], "fold_local_best_f1")
                self.assertEqual(report["calibration_target"], "object")
                self.assertIn("applied_threshold", report)
                for suffix in (
                    "byte_cnn_region_predictions_path",
                    "byte_cnn_object_predictions_path",
                    "byte_cnn_apk_predictions_path",
                ):
                    p = Path(task[suffix])
                    self.assertTrue(
                        p.exists(),
                        f"missing byte_cnn prediction artefact {p}",
                    )

    def test_mil_byte_cnn_fusion_dispatch_same_set_mode(self):
        """Fusion can be enabled by itself and still prepares both
        Ours/MIL and byte_cnn component models before writing standard
        three-level prediction artefacts."""

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed; fusion requires [dl] extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.fusion_a", 1, "a.apk"),
                    ("org.example.fusion_b", 1, "b.apk"),
                ],
            )
            config_path = _write_config(
                tmp,
                manifest,
                baselines=["entropy", "mil_byte_cnn_fusion"],
            )
            rc = runner.main(
                ["--config", str(config_path), "--no-skip-existing"]
            )
            self.assertEqual(rc, 0)

            summary_path = tmp / "outputs" / "summary.json"
            manifest_path = tmp / "outputs" / "experiment_manifest.json"
            summary = json.loads(summary_path.read_text("utf-8"))
            manifest_out = json.loads(manifest_path.read_text("utf-8"))

            self.assertIn("mil_byte_cnn_fusion", summary["baselines"])
            self.assertEqual(
                summary["baselines"]["mil_byte_cnn_fusion"]["successful_task_count"],
                manifest_out["task_count"],
            )

            for task in manifest_out["tasks"]:
                self.assertEqual(
                    task["baselines"]["mil_byte_cnn_fusion"]["status"],
                    "ok",
                )
                report_path = Path(task["mil_byte_cnn_fusion_report_path"])
                report = json.loads(report_path.read_text("utf-8"))
                self.assertEqual(report["baseline"], "mil_byte_cnn_fusion")
                self.assertIn("fusion", report)
                self.assertIn("components", report)
                self.assertEqual(report["components"]["mil"]["baseline"], "ours")
                self.assertEqual(
                    report["components"]["byte_cnn"]["baseline"],
                    "byte_cnn",
                )
                for suffix in (
                    "mil_byte_cnn_fusion_region_predictions_path",
                    "mil_byte_cnn_fusion_object_predictions_path",
                    "mil_byte_cnn_fusion_apk_predictions_path",
                ):
                    p = Path(task[suffix])
                    self.assertTrue(
                        p.exists(),
                        f"missing fusion prediction artefact {p}",
                    )


class SkipExistingTests(unittest.TestCase):
    """`--skip-existing` (default on) reuses cached artefacts when the
    fingerprint matches, recomputes when inputs or config change, and can
    be disabled with `--no-skip-existing` / `--force`."""

    def _run(self, config_path: Path, extra: list[str]) -> Path:
        rc = runner.main(
            ["--config", str(config_path), "--transforms", "xor", *extra]
        )
        self.assertEqual(rc, 0)
        return Path(json.loads(config_path.read_text("utf-8"))["outputs"]["output_root"])

    def test_second_run_reuses_all_artefacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(tmp, [("org.example.reuse", 1, "reuse.apk")])
            config_path = _write_config(
                tmp, manifest, baselines=["entropy", "sanity_rules"]
            )

            out_root = self._run(config_path, [])
            manifest_first = json.loads((out_root / "experiment_manifest.json").read_text("utf-8"))
            self.assertEqual(manifest_first["reuse"]["generate_reused_count"], 0)
            self.assertEqual(manifest_first["reuse"]["baseline_reused_count"], 0)

            self._run(config_path, [])
            manifest_second = json.loads((out_root / "experiment_manifest.json").read_text("utf-8"))
            self.assertEqual(manifest_second["reuse"]["generate_reused_count"], 1)
            self.assertEqual(manifest_second["reuse"]["baseline_reused_count"], 2)
            for task in manifest_second["tasks"]:
                self.assertTrue(task.get("generate_reused"))
                for baseline_name, bundle in task.get("baselines", {}).items():
                    self.assertTrue(
                        bundle.get("reused"),
                        f"baseline {baseline_name} expected reused=True; got {bundle}",
                    )

    def test_seed_apk_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(tmp, [("org.example.change", 1, "change.apk")])
            config_path = _write_config(
                tmp, manifest, baselines=["entropy", "sanity_rules"]
            )
            out_root = self._run(config_path, [])

            # Mutate the seed APK content. Same path, different bytes.
            entries = json.loads(manifest.read_text("utf-8"))["entries"]
            seed_path = Path(entries[0]["local_path"])
            with zipfile.ZipFile(seed_path, "w") as archive:
                archive.writestr("classes.dex", b"dex\n035\x00" + (b"DIFFERENT bytes!! " * 40))
                archive.writestr("assets/readme.txt", b"changed text " * 16)

            self._run(config_path, [])
            manifest_after = json.loads((out_root / "experiment_manifest.json").read_text("utf-8"))
            # Generate must be recomputed; the downstream baselines must
            # be recomputed too because the APK they score has changed.
            self.assertEqual(manifest_after["reuse"]["generate_reused_count"], 0)
            self.assertEqual(manifest_after["reuse"]["baseline_reused_count"], 0)

    def test_no_skip_existing_forces_full_recompute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = _write_seed_manifest(tmp, [("org.example.force", 1, "force.apk")])
            config_path = _write_config(
                tmp, manifest, baselines=["entropy", "sanity_rules"]
            )
            out_root = self._run(config_path, [])
            self._run(config_path, ["--no-skip-existing"])
            manifest_after = json.loads((out_root / "experiment_manifest.json").read_text("utf-8"))
            self.assertEqual(manifest_after["reuse"]["generate_reused_count"], 0)
            self.assertEqual(manifest_after["reuse"]["baseline_reused_count"], 0)


if __name__ == "__main__":
    unittest.main()
