import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import unittest
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env() -> dict:
    """Ensure the subprocess can import ``android_packer`` without editable install."""

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + existing if existing else "")
    return env


class SyntheticEntropyExperimentTests(unittest.TestCase):
    def test_cli_runs_mini_batch_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.one", 1, "one.apk"),
                    ("org.example.two", 2, "two.apk"),
                ],
            )
            config = _write_config(tmp, seed_manifest)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/experiments/run_synthetic_entropy_baseline.py",
                    "--config",
                    str(config),
                    "--transforms",
                    "xor",
                    "base64",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            self.assertIn("tasks=4 successes=4 failures=0", completed.stdout)
            summary = json.loads((tmp / "outputs" / "summary.json").read_text())
            manifest = json.loads(
                (tmp / "outputs" / "experiment_manifest.json").read_text()
            )
            self.assertEqual(summary["successful_task_count"], 4)
            self.assertEqual(summary["overall"]["task_count"], 4)
            self.assertEqual(sorted(summary["by_transform"]), ["base64", "xor"])
            self.assertEqual(manifest["task_count"], 4)
            self.assertEqual(manifest["failure_count"], 0)
            for task in manifest["tasks"]:
                self.assertEqual(task["status"], "ok")
                self.assertTrue(Path(task["region_predictions_path"]).exists())
                self.assertTrue(Path(task["baseline_report_path"]).exists())

            # Batch B aggregation: AUROC / AUPRC keys must be present per
            # level, and the summary must also surface ranking + localization
            # macro-averages.
            overall = summary["overall"]
            for level in ("region", "object", "apk"):
                level_metrics = overall["metrics"][level]
                self.assertIn("auroc", level_metrics)
                self.assertIn("auprc", level_metrics)
                self.assertIn("auroc_task_count", level_metrics)
            self.assertIn("ranking", overall)
            self.assertIn("object", overall["ranking"])
            self.assertIn("mrr", overall["ranking"]["object"])
            self.assertIn("top_k_hit", overall["ranking"]["object"])
            self.assertIn("localization", overall)
            self.assertIn("mean_iou", overall["localization"]["object"])
            self.assertIn("mean_boundary_error", overall["localization"]["object"])
            self.assertIn("offset_hit_rate", overall["localization"]["object"])

    def test_cli_records_failed_task_and_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_manifest = _write_seed_manifest(
                tmp,
                [
                    ("org.example.valid", 1, "valid.apk"),
                    ("org.example.missing", 2, "missing.apk"),
                ],
                missing={"missing.apk"},
            )
            config = _write_config(tmp, seed_manifest)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/experiments/run_synthetic_entropy_baseline.py",
                    "--config",
                    str(config),
                    "--transforms",
                    "xor",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tasks=2 successes=1 failures=1", completed.stdout)
            manifest = json.loads(
                (tmp / "outputs" / "experiment_manifest.json").read_text()
            )
            self.assertEqual([task["status"] for task in manifest["tasks"]], ["ok", "failed"])
            self.assertIn("missing.apk", manifest["tasks"][1]["error"])


def _write_seed_manifest(
    root: Path,
    seeds: list[tuple[str, int, str]],
    *,
    missing: Optional[set[str]] = None,
) -> Path:
    missing = missing or set()
    seed_dir = root / "seeds"
    seed_dir.mkdir()
    entries = []
    for package_name, version_code, filename in seeds:
        apk_path = seed_dir / filename
        if filename not in missing:
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


def _write_seed_apk(path: Path) -> None:
    payload = b"dex\n035\x00" + (b"payload bytes " * 64)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.dex", payload)
        archive.writestr("assets/readme.txt", b"benign text")


def _write_config(root: Path, seed_manifest: Path) -> Path:
    config = {
        "input": {
            "seed_manifest": str(seed_manifest),
        },
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
        "baseline": {
            "entropy_threshold": 1.0,
            "entropy_weight": 1.0,
            "nonprintable_weight": 0.0,
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


if __name__ == "__main__":
    unittest.main()
