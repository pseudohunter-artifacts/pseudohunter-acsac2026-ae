"""CLI: run a batch synthetic-packer + entropy-baseline experiment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from android_packer.apkio import iter_apk_objects
from android_packer.apkio.objects import file_sha256
from android_packer.baselines import EntropyBaselineConfig, run_entropy_baseline
from android_packer.experiments import aggregate_reports
from android_packer.labeling import build_training_labels
from android_packer.regioning import iter_regions
from android_packer.synthetic import (
    SUPPORTED_TRANSFORMS,
    build_synthetic_apk,
    derive_task_rng_seed,
)
from android_packer.utils.jsonl import write_jsonl
from android_packer.utils.paths import find_project_root

REPO_ROOT = find_project_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "synthetic_entropy_baseline.json"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic generation, labeling, and entropy baseline in batch."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment config JSON.",
    )
    parser.add_argument("--seed-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--synthetic-generated-dir", type=Path, default=None)
    parser.add_argument("--synthetic-manifest-dir", type=Path, default=None)
    parser.add_argument("--synthetic-label-dir", type=Path, default=None)
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=SUPPORTED_TRANSFORMS,
        default=None,
        help="Transform families to run. Defaults to config.",
    )
    parser.add_argument(
        "--limit-seeds",
        type=int,
        default=None,
        help="Optional cap on seed entries, for smoke tests.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed task instead of continuing the batch.",
    )
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(args.config)
    seed_manifest_path = args.seed_manifest or Path(config["input"]["seed_manifest"])
    output_root = args.output_root or Path(config["outputs"]["output_root"])
    generated_dir = args.synthetic_generated_dir or Path(
        config["outputs"]["synthetic_generated_dir"]
    )
    synthetic_manifest_dir = args.synthetic_manifest_dir or Path(
        config["outputs"]["synthetic_manifest_dir"]
    )
    synthetic_label_dir = args.synthetic_label_dir or Path(
        config["outputs"]["synthetic_label_dir"]
    )
    transforms = list(args.transforms or config["synthetic"]["transform_families"])
    _validate_transforms(transforms)
    if args.limit_seeds is not None and args.limit_seeds < 1:
        raise ValueError("--limit-seeds must be at least 1")

    seed_manifest = _read_json(seed_manifest_path)
    seed_entries = list(seed_manifest.get("entries", []))
    if args.limit_seeds is not None:
        seed_entries = seed_entries[: args.limit_seeds]

    output_root.mkdir(parents=True, exist_ok=True)
    task_root = output_root / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)

    experiment_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "synthetic_entropy_baseline",
        "seed_manifest": _repo_path(seed_manifest_path),
        "seed_count": len(seed_entries),
        "transform_families": transforms,
        "config": {
            "regioning": config["regioning"],
            "labeling": config["labeling"],
            "baseline": config["baseline"],
        },
        "paths": {
            "output_root": _repo_path(output_root),
            "synthetic_generated_dir": _repo_path(generated_dir),
            "synthetic_manifest_dir": _repo_path(synthetic_manifest_dir),
            "synthetic_label_dir": _repo_path(synthetic_label_dir),
        },
        "tasks": [],
    }

    reports: list[dict] = []
    failures = 0
    total_tasks = len(seed_entries) * len(transforms)
    task_index = 0
    for seed_entry in seed_entries:
        for transform_family in transforms:
            task_index += 1
            task = _task_descriptor(
                seed_entry=seed_entry,
                transform_family=transform_family,
                task_root=task_root,
                generated_dir=generated_dir,
                synthetic_manifest_dir=synthetic_manifest_dir,
                synthetic_label_dir=synthetic_label_dir,
            )
            print(f"start task={task_index}/{total_tasks} name={task['task_name']}", flush=True)
            try:
                task_report = _run_task(
                    task=task,
                    config=config,
                    seed_entry=seed_entry,
                    transform_family=transform_family,
                )
                task["status"] = "ok"
                task["metrics"] = task_report["metrics"]
                reports.append(task_report)
                print(
                    " ".join(
                        [
                            f"ok task={task['task_name']}",
                            f"region_f1={task_report['metrics']['region']['f1']}",
                            f"object_f1={task_report['metrics']['object']['f1']}",
                            f"apk_f1={task_report['metrics']['apk']['f1']}",
                        ]
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - batch should record task failures.
                failures += 1
                task["status"] = "failed"
                task["error"] = str(exc)
                task["traceback"] = traceback.format_exc()
                print(f"failed task={task['task_name']} error={exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    experiment_manifest["tasks"].append(task)
                    _write_outputs(output_root, experiment_manifest, reports)
                    raise
            experiment_manifest["tasks"].append(task)

    experiment_manifest["task_count"] = len(experiment_manifest["tasks"])
    experiment_manifest["success_count"] = sum(
        1 for task in experiment_manifest["tasks"] if task["status"] == "ok"
    )
    experiment_manifest["failure_count"] = failures
    summary = _write_outputs(output_root, experiment_manifest, reports)
    print(
        " ".join(
            [
                f"tasks={experiment_manifest['task_count']}",
                f"successes={experiment_manifest['success_count']}",
                f"failures={experiment_manifest['failure_count']}",
                f"summary={_repo_path(output_root / 'summary.json')}",
            ]
        ),
        flush=True,
    )
    return {
        "experiment_manifest": experiment_manifest,
        "summary": summary,
    }


def _run_task(
    *,
    task: Dict[str, Any],
    config: Mapping[str, Any],
    seed_entry: Mapping[str, Any],
    transform_family: str,
) -> dict:
    seed_apk = Path(seed_entry["local_path"])
    regioning = config["regioning"]
    labeling = config["labeling"]
    baseline = config["baseline"]

    # Per-task RNG seed derivation (2026-04-29, B2 follow-up): see
    # ``src/android_packer/synthetic/seed_derivation.py`` docstring
    # and ``docs/method/threat_model.md`` §"B2 实装后遗留".
    base_seed = int(config["synthetic"]["rng_seed"])
    effective_seed = derive_task_rng_seed(
        base_seed=base_seed,
        package_name=str(seed_entry.get("package_name") or ""),
        version_code=str(seed_entry.get("version_code") or ""),
        transform_family=transform_family,
    )
    task["rng_seed_base"] = base_seed
    task["rng_seed"] = effective_seed

    synthetic_result = build_synthetic_apk(
        seed_apk=seed_apk,
        generated_apk_out=Path(task["generated_apk_path"]),
        manifest_out=Path(task["synthetic_manifest_path"]),
        labels_out=Path(task["synthetic_labels_path"]),
        transform_family=transform_family,
        rng_seed=effective_seed,
        asset_prefix=str(config["synthetic"]["asset_prefix"]),
        split_count=int(config["synthetic"]["split_count"]),
        # B1 (2026-04-29): default True enforces per-family payload
        # lower bound; config may opt out in unit-test fixtures with
        # placeholder DEXes below the 64 KiB floor.
        enforce_payload_size_range=bool(
            config["synthetic"].get("enforce_payload_size_range", True)
        ),
    )

    object_rows = []
    region_rows = []
    for metadata, data in iter_apk_objects(
        synthetic_result.generated_apk_path,
        max_depth=int(regioning["max_depth"]),
        max_member_bytes=regioning.get("max_member_bytes"),
    ):
        object_rows.append(metadata.to_dict())
        for region in iter_regions(
            metadata,
            data,
            window_size=int(regioning["window_size"]),
            stride=int(regioning["stride"]),
            min_region_size=int(regioning["min_region_size"]),
            include_tail=bool(regioning["include_tail"]),
        ):
            region_rows.append(region.to_dict())

    write_jsonl(Path(task["objects_path"]), object_rows)
    write_jsonl(Path(task["regions_path"]), region_rows)

    labels = build_training_labels(
        regions=region_rows,
        synthetic_labels=[label.to_dict() for label in synthetic_result.labels],
        min_overlap_bytes=int(labeling["min_overlap_bytes"]),
        min_overlap_ratio=float(labeling["min_overlap_ratio"]),
    )
    write_jsonl(
        Path(task["region_labels_path"]),
        (row.to_dict() for row in labels.region_labels),
    )
    write_jsonl(
        Path(task["object_labels_path"]),
        (row.to_dict() for row in labels.object_labels),
    )
    write_jsonl(
        Path(task["apk_labels_path"]),
        (row.to_dict() for row in labels.apk_labels),
    )

    result = run_entropy_baseline(
        (row.to_dict() for row in labels.region_labels),
        EntropyBaselineConfig(
            entropy_threshold=float(baseline["entropy_threshold"]),
            entropy_weight=float(baseline["entropy_weight"]),
            nonprintable_weight=float(baseline["nonprintable_weight"]),
        ),
    )
    write_jsonl(
        Path(task["region_predictions_path"]),
        (row.to_dict() for row in result.region_predictions),
    )
    write_jsonl(
        Path(task["object_predictions_path"]),
        (row.to_dict() for row in result.object_predictions),
    )
    write_jsonl(
        Path(task["apk_predictions_path"]),
        (row.to_dict() for row in result.apk_predictions),
    )

    task_report = {
        **result.report,
        "task_name": task["task_name"],
        "package_name": seed_entry.get("package_name"),
        "app_name": seed_entry.get("app_name"),
        "version_code": seed_entry.get("version_code"),
        "transform_family": transform_family,
        "paths": {
            "generated_apk": _repo_path(Path(task["generated_apk_path"])),
            "region_labels": _repo_path(Path(task["region_labels_path"])),
            "region_predictions": _repo_path(Path(task["region_predictions_path"])),
        },
    }
    _write_json(Path(task["baseline_report_path"]), task_report)
    return task_report


def _task_descriptor(
    *,
    seed_entry: Mapping[str, Any],
    transform_family: str,
    task_root: Path,
    generated_dir: Path,
    synthetic_manifest_dir: Path,
    synthetic_label_dir: Path,
) -> Dict[str, Any]:
    stem = _task_stem(seed_entry, transform_family)
    task_dir = task_root / stem
    return {
        "task_name": stem,
        "package_name": seed_entry.get("package_name"),
        "app_name": seed_entry.get("app_name"),
        "version_name": seed_entry.get("version_name"),
        "version_code": seed_entry.get("version_code"),
        "transform_family": transform_family,
        "seed_apk_path": seed_entry.get("local_path"),
        "generated_apk_path": _repo_path(generated_dir / f"{stem}.apk"),
        "synthetic_manifest_path": _repo_path(synthetic_manifest_dir / f"{stem}.manifest.json"),
        "synthetic_labels_path": _repo_path(synthetic_label_dir / f"{stem}.labels.jsonl"),
        "objects_path": _repo_path(task_dir / "objects.jsonl"),
        "regions_path": _repo_path(task_dir / "regions.jsonl"),
        "region_labels_path": _repo_path(task_dir / "region_labels.jsonl"),
        "object_labels_path": _repo_path(task_dir / "object_labels.jsonl"),
        "apk_labels_path": _repo_path(task_dir / "apk_labels.jsonl"),
        "region_predictions_path": _repo_path(task_dir / "entropy.region_predictions.jsonl"),
        "object_predictions_path": _repo_path(task_dir / "entropy.object_predictions.jsonl"),
        "apk_predictions_path": _repo_path(task_dir / "entropy.apk_predictions.jsonl"),
        "baseline_report_path": _repo_path(task_dir / "entropy_report.json"),
    }


def _write_outputs(
    output_root: Path,
    experiment_manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> dict:
    summary = _summarize_reports(reports)
    _write_json(output_root / "experiment_manifest.json", dict(experiment_manifest))
    _write_json(output_root / "summary.json", summary)
    return summary


def _summarize_reports(reports: Sequence[Mapping[str, Any]]) -> dict:
    by_transform: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        by_transform.setdefault(str(report["transform_family"]), []).append(report)

    return {
        "experiment": "synthetic_entropy_baseline",
        "successful_task_count": len(reports),
        "overall": aggregate_reports(reports),
        "by_transform": {
            transform: aggregate_reports(items)
            for transform, items in sorted(by_transform.items())
        },
    }


def _task_stem(seed_entry: Mapping[str, Any], transform_family: str) -> str:
    package_name = str(seed_entry["package_name"]).replace(".", "_")
    version_code = str(seed_entry["version_code"])
    seed_path = seed_entry.get("local_path")
    seed_token = _seed_token(seed_path) if seed_path else None
    parts = [package_name, version_code]
    if seed_token:
        parts.append(seed_token)
    parts.append(transform_family)
    return "_".join(parts)


def _seed_token(seed_path: Any) -> Optional[str]:
    """Return a short content token for ``seed_path`` or ``None`` if unreadable.

    Distinct seed APKs that happen to share ``(package_name, version_code)``
    would otherwise collide on the task stem and overwrite each other's
    artefacts. We fall back to ``None`` when the seed is missing so that a
    later failure in ``_run_task`` can still be recorded under a deterministic
    task name.
    """

    try:
        return file_sha256(Path(seed_path))[:8]
    except (OSError, ValueError):
        return None


def _validate_transforms(transforms: Iterable[str]) -> None:
    unsupported = [item for item in transforms if item not in SUPPORTED_TRANSFORMS]
    if unsupported:
        raise ValueError(f"unsupported transform families: {', '.join(unsupported)}")


def _load_config(path: Path) -> dict:
    return _read_json(path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _repo_path(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    result = run_experiment(args)
    return 1 if result["experiment_manifest"]["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
