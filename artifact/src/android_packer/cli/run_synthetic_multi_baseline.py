"""CLI: run a batch synthetic experiment scored by multiple baselines.

This runner generates synthetic packed APKs from a seed manifest and
scores each generated APK with every enabled baseline (entropy,
sanity_rules, ngram_logreg, optionally apkid). The goal is a single
``summary.json`` that can drive the first multi-baseline comparison
table in the paper, without regenerating APKs between baselines.

Supported baselines:

- ``entropy``: stdlib-only entropy + printable-ratio threshold.
- ``sanity_rules``: internal heuristic sanity check.
- ``ngram_logreg``: byte-level LR; requires sklearn. The config must
  provide ``train_mode = "same_set"`` (the only supported mode at the
  moment). In this mode the model is trained on the union of every
  task's region labels and scored in-sample. This overstates the
  true generalisation; the summary's ``warnings`` field documents
  this explicitly so the reader cannot be misled.
- ``apkid``: opt-in; skipped with a recorded note when the optional
  ``apkid`` package is not installed.

Everything else (generation, labelling) is done once per task and
reused across baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from android_packer.apkio import iter_apk_objects
from android_packer.apkio.objects import file_sha256
from android_packer.baselines import (
    ByteCnnBaselineConfig,
    EntropyBaselineConfig,
    MilByteCnnFusionConfig,
    NgramLogRegConfig,
    OursBaselineConfig,
    OursBaselineModel,
    PayloadHunterLiteConfig,
    PayloadHunterLiteModel,
    SanityRulesConfig,
    fuse_mil_and_byte_cnn_results,
    run_entropy_baseline,
    run_sanity_rules_baseline,
    train_byte_cnn,
    train_ngram_logreg,
    train_ours_baseline,
    train_ours_baseline_from_objects,
    train_payload_hunter_lite,
)
from android_packer.baselines.ours import _aggregate_object_features
from android_packer.evaluation.metrics import binary_classification_metrics
from android_packer.experiments import aggregate_reports
from android_packer.features import ByteFeatureConfig, ObjectByteLoader
from android_packer.models import ByteCnnRegionScorerConfig
from android_packer.labeling import build_training_labels
from android_packer.regioning import iter_regions
from android_packer.synthetic import (
    SUPPORTED_TRANSFORMS,
    build_synthetic_apk,
    derive_task_rng_seed,
)
from android_packer.utils.jsonl import read_jsonl, write_jsonl
from android_packer.utils.paths import find_project_root

REPO_ROOT = find_project_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "synthetic_multi_baseline.json"

SUPPORTED_BASELINES = (
    "entropy",
    "sanity_rules",
    "ngram_logreg",
    "byte_cnn",
    "payload_hunter_lite",
    "ours",
    "mil_byte_cnn_fusion",
    "apkid",
)

# Bump these when the on-disk artefact format changes in a way that
# invalidates previously cached results. ``--skip-existing`` compares
# the recorded version against the current one before reusing.
GENERATE_FINGERPRINT_VERSION = "1"
BASELINE_FINGERPRINT_VERSION = "1"

# Phase tokens accepted by ``--force-phase``.
FORCE_PHASE_GENERATE = "generate"
FORCE_PHASE_BASELINE = "baseline"
SUPPORTED_FORCE_PHASES = (FORCE_PHASE_GENERATE, FORCE_PHASE_BASELINE)

# Safety guard for baselines that materialise one feature row per region.
# The v4 LOFO corpus currently has ~2.08M region rows; historical runs on
# this workstation pushed ngram_logreg to ~39 GB RSS and PayloadHunter-Lite
# to >30 GB RSS on smaller folds.  Keep the default conservative; set the
# per-baseline JSON value to null only on a high-memory machine.
DEFAULT_NGRAM_MAX_TRAIN_ROWS = 500_000
DEFAULT_BYTE_CNN_MAX_TRAIN_ROWS = 500_000
DEFAULT_BYTE_CNN_TRAIN_SAMPLE_MIN_POSITIVE_ROWS = 64
DEFAULT_LITE_MAX_TRAIN_ROWS = 500_000


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic generation + region labelling once, then score each "
            "generated APK with every enabled baseline; emit a combined summary."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=SUPPORTED_BASELINES,
        default=None,
        help="Subset of baselines to run. Defaults to config.",
    )
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument(
        "--include-benign-apks",
        type=int,
        default=None,
        help=(
            "Add up to N pure-benign seed APK evaluation-only controls. "
            "They are excluded from model training but scored so APK AUROC/AUPRC "
            "are defined. Defaults to config.synthetic.include_benign_apks or 0."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")

    # --- Skip-existing controls ----------------------------------------
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help=(
            "Reuse cached generate/baseline artefacts when their fingerprint "
            "matches the current input (default)."
        ),
    )
    skip_group.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable fingerprint-based reuse and recompute everything.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Alias for --no-skip-existing; recompute everything.",
    )
    parser.add_argument(
        "--force-phase",
        action="append",
        choices=SUPPORTED_FORCE_PHASES,
        default=None,
        help=(
            "Recompute the named phase even when --skip-existing is on. "
            "May be passed multiple times. Valid values: generate, baseline."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan what would be generated/scored or skipped under the current "
            "fingerprint state without writing any files. Implies a read-only run."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(args.config)
    seed_manifest_path = args.seed_manifest or Path(
        config.get("input", {}).get("seed_manifest") or config.get("seed_manifest")
    )
    output_root = args.output_root or Path(
        config.get("outputs", {}).get("output_root") or config.get("paths", {}).get("output_root")
    )
    generated_dir = args.synthetic_generated_dir or Path(
        config.get("outputs", {}).get("synthetic_generated_dir") or config.get("paths", {}).get("synthetic_generated_dir")
    )
    synthetic_manifest_dir = args.synthetic_manifest_dir or Path(
        config.get("outputs", {}).get("synthetic_manifest_dir") or config.get("paths", {}).get("synthetic_manifest_dir")
    )
    synthetic_label_dir = args.synthetic_label_dir or Path(
        config.get("outputs", {}).get("synthetic_label_dir") or config.get("paths", {}).get("synthetic_label_dir")
    )

    transforms = list(args.transforms or config.get("synthetic", {}).get("transform_families") or config.get("transform_families"))
    _validate_transforms(transforms)
    baselines = list(args.baselines or config["baselines"]["enabled"])
    _validate_baselines(baselines)

    if args.limit_seeds is not None and args.limit_seeds < 1:
        raise ValueError("--limit-seeds must be at least 1")
    include_benign_apks = (
        args.include_benign_apks
        if args.include_benign_apks is not None
        else int(config.get("synthetic", {}).get("include_benign_apks", 0) or 0)
    )
    if include_benign_apks < 0:
        raise ValueError("--include-benign-apks must be non-negative")

    # Resolve skip-existing policy. ``--force`` / ``--no-skip-existing``
    # win over the default; ``--force-phase`` is additive.
    skip_existing = bool(getattr(args, "skip_existing", True))
    if getattr(args, "force", False):
        skip_existing = False
    forced_phases = set(getattr(args, "force_phase", None) or [])
    dry_run = bool(getattr(args, "dry_run", False))

    seed_manifest = _read_json(seed_manifest_path)
    seed_entries = list(seed_manifest.get("entries", []))
    if args.limit_seeds is not None:
        seed_entries = seed_entries[: args.limit_seeds]

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        task_root = output_root / "tasks"
        task_root.mkdir(parents=True, exist_ok=True)
    else:
        task_root = output_root / "tasks"

    # --- Phase 1: generate + label every task once -----------------------
    task_states: list[dict] = []
    failures = 0
    generate_skipped = 0
    benign_entries = seed_entries[:include_benign_apks]
    total_tasks = len(seed_entries) * len(transforms) + len(benign_entries)
    task_index = 0
    skip_generate = skip_existing and FORCE_PHASE_GENERATE not in forced_phases
    for seed_entry in seed_entries:
        for transform_family in transforms:
            task_index += 1
            state = _task_descriptor(
                seed_entry=seed_entry,
                transform_family=transform_family,
                task_root=task_root,
                generated_dir=generated_dir,
                synthetic_manifest_dir=synthetic_manifest_dir,
                synthetic_label_dir=synthetic_label_dir,
            )

            current_fp = _generate_fingerprint(seed_entry, transform_family, config)
            if skip_generate and _generate_artifacts_reusable(state, current_fp):
                if dry_run:
                    print(
                        f"would skip generate task={task_index}/{total_tasks} "
                        f"name={state['task_name']} (fingerprint match)",
                        flush=True,
                    )
                    state["status"] = "ok"
                    state["generate_reused"] = True
                    task_states.append(state)
                    generate_skipped += 1
                    continue
                try:
                    state["_region_label_rows"] = list(
                        read_jsonl(_resolve_state_path(state["region_labels_path"]))
                    )
                    state["status"] = "ok"
                    state["generate_reused"] = True
                    generate_skipped += 1
                    print(
                        f"skip generate task={task_index}/{total_tasks} "
                        f"name={state['task_name']} (fingerprint match)",
                        flush=True,
                    )
                    task_states.append(state)
                    continue
                except Exception as exc:  # noqa: BLE001 - cache load failure → recompute
                    print(
                        f"warning: cache load failed for {state['task_name']}: {exc}; recomputing",
                        file=sys.stderr,
                        flush=True,
                    )

            if dry_run:
                print(
                    f"would generate task={task_index}/{total_tasks} "
                    f"name={state['task_name']}",
                    flush=True,
                )
                state["status"] = "ok"
                state["generate_reused"] = False
                task_states.append(state)
                continue

            print(
                f"generate task={task_index}/{total_tasks} name={state['task_name']}",
                flush=True,
            )
            try:
                _generate_task_artifacts(
                    state=state,
                    config=config,
                    seed_entry=seed_entry,
                    transform_family=transform_family,
                )
                state["status"] = "ok"
                state["generate_reused"] = False
                _write_fingerprint(_generate_fingerprint_path(state), current_fp)
            except Exception as exc:  # noqa: BLE001 - batch continues on task failure.
                failures += 1
                state["status"] = "failed"
                state["error"] = str(exc)
                state["traceback"] = traceback.format_exc()
                print(
                    f"failed task={state['task_name']} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    experiment_manifest = _build_experiment_manifest(
                        config,
                        seed_manifest_path,
                        seed_entries,
                        transforms,
                        baselines,
                        output_root,
                        generated_dir,
                        synthetic_manifest_dir,
                        synthetic_label_dir,
                        task_states + [state],
                        failures,
                        {},
                    )
                    _write_outputs(output_root, experiment_manifest, {})
                    raise
            task_states.append(state)

    for seed_entry in benign_entries:
        task_index += 1
        state = _benign_task_descriptor(
            seed_entry=seed_entry,
            task_root=task_root,
            generated_dir=generated_dir,
            synthetic_manifest_dir=synthetic_manifest_dir,
            synthetic_label_dir=synthetic_label_dir,
        )
        current_fp = _generate_fingerprint(seed_entry, state["transform_family"], config)
        if skip_generate and _generate_artifacts_reusable(state, current_fp):
            if dry_run:
                print(
                    f"would skip generate task={task_index}/{total_tasks} "
                    f"name={state['task_name']} (fingerprint match)",
                    flush=True,
                )
                state["status"] = "ok"
                state["generate_reused"] = True
                task_states.append(state)
                generate_skipped += 1
                continue
            try:
                state["_region_label_rows"] = list(
                    read_jsonl(_resolve_state_path(state["region_labels_path"]))
                )
                state["status"] = "ok"
                state["generate_reused"] = True
                generate_skipped += 1
                print(
                    f"skip generate task={task_index}/{total_tasks} "
                    f"name={state['task_name']} (fingerprint match)",
                    flush=True,
                )
                task_states.append(state)
                continue
            except Exception as exc:  # noqa: BLE001 - cache load failure → recompute
                print(
                    f"warning: cache load failed for {state['task_name']}: {exc}; recomputing",
                    file=sys.stderr,
                    flush=True,
                )

        if dry_run:
            print(
                f"would generate task={task_index}/{total_tasks} "
                f"name={state['task_name']}",
                flush=True,
            )
            state["status"] = "ok"
            state["generate_reused"] = False
            task_states.append(state)
            continue

        print(
            f"generate task={task_index}/{total_tasks} name={state['task_name']}",
            flush=True,
        )
        try:
            _generate_benign_task_artifacts(state=state, config=config, seed_entry=seed_entry)
            state["status"] = "ok"
            state["generate_reused"] = False
            _write_fingerprint(_generate_fingerprint_path(state), current_fp)
        except Exception as exc:  # noqa: BLE001 - batch continues on task failure.
            failures += 1
            state["status"] = "failed"
            state["error"] = str(exc)
            state["traceback"] = traceback.format_exc()
            print(
                f"failed task={state['task_name']} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                experiment_manifest = _build_experiment_manifest(
                    config,
                    seed_manifest_path,
                    seed_entries,
                    transforms,
                    baselines,
                    output_root,
                    generated_dir,
                    synthetic_manifest_dir,
                    synthetic_label_dir,
                    task_states + [state],
                    failures,
                    {},
                )
                _write_outputs(output_root, experiment_manifest, {})
                raise
        task_states.append(state)

    # --- Phase 2: train/score per baseline ------------------------------
    warnings: list[str] = []
    reports_by_baseline: dict[str, list[dict]] = {name: [] for name in baselines}
    skip_baseline = skip_existing and FORCE_PHASE_BASELINE not in forced_phases
    baseline_skipped = 0

    # APKiD availability check up-front so that we record a clean skip
    # reason in the summary instead of failing the whole run.
    apkid_skip_reason: Optional[str] = None
    if "apkid" in baselines:
        try:
            # Import lazily; the attempt itself verifies optional deps.
            from android_packer.baselines.apkid import _default_scan_fn  # type: ignore
            from android_packer.baselines import ApkidNotInstalledError
            try:
                _default_scan_fn()
            except ApkidNotInstalledError as exc:
                apkid_skip_reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            apkid_skip_reason = f"apkid unavailable: {exc}"
        if apkid_skip_reason:
            warnings.append(f"apkid skipped: {apkid_skip_reason}")

    # n-gram LR needs a trained model. Build it once from the union of
    # all successful tasks' region labels ("same_set" mode).
    # In dry-run we cannot train (no disk writes allowed), so leave it empty.
    ngram_state: Dict[str, Any] = {}
    if "ngram_logreg" in baselines and not dry_run:
        # Only train when we will actually score with it; if every task
        # reuses cached ngram results we still need the model for any
        # task whose fingerprint changed. Cheaper to always train when
        # at least one task is going to (re)score.
        ngram_state = _prepare_ngram_model(
            task_states=task_states,
    config=config.get("baselines", {}).get("ngram_logreg", {}) if isinstance(config.get("baselines"), dict) else {},
            output_root=output_root,
            warnings=warnings,
        )

    # Byte-CNN is the lightweight byte-only neural baseline. It is
    # trained in the same same_set / holdout modes as PayloadHunter-Lite,
    # but streams raw region bytes by batch instead of materialising a
    # handcrafted feature matrix.
    byte_cnn_state: Dict[str, Any] = {}
    needs_byte_cnn = any(
        name in baselines for name in ("byte_cnn", "mil_byte_cnn_fusion")
    )
    if needs_byte_cnn and not dry_run:
        byte_cnn_state = _prepare_byte_cnn_model(
            task_states=task_states,
            config=config.get("baselines", {}).get("byte_cnn", {}) if isinstance(config.get("baselines"), dict) else {},
            output_root=output_root,
            warnings=warnings,
        )

    # PayloadHunter-Lite is trained identically to ngram_logreg on the
    # union of every task's labels; the F-Lite-e batch will promote
    # ``train_mode`` to a first-class flag and replace this with a
    # holdout-aware loop.
    lite_state: Dict[str, Any] = {}
    if "payload_hunter_lite" in baselines and not dry_run:
        lite_state = _prepare_payload_hunter_lite_model(
            task_states=task_states,
    config=config.get("baselines", {}).get("payload_hunter_lite", {}) if isinstance(config.get("baselines"), dict) else {},
            output_root=output_root,
            warnings=warnings,
        )

    # Ours (Typed-Instance MIL) — trained in the same same_set /
    # holdout_transform / holdout_package modes as payload_hunter_lite.
    # The paper-quotable Tier-A configuration is
    # ``supervision_mode="bag"`` and ``train_mode="holdout_transform"``
    # (LOFO-by-family).  See
    # ``docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md``
    # §7.3 (L43 integrity contract) and §7.4 (L45 LOFO split).
    ours_state: Dict[str, Any] = {}
    needs_ours = any(
        name in baselines for name in ("ours", "mil_byte_cnn_fusion")
    )
    if needs_ours and not dry_run:
        ours_state = _prepare_ours_model(
            task_states=task_states,
            config=config.get("baselines", {}).get("ours", {}) if isinstance(config.get("baselines"), dict) else {},
            output_root=output_root,
            warnings=warnings,
        )

    for state in task_states:
        if state.get("status") != "ok":
            continue

        task_report_bundle: Dict[str, Any] = state.setdefault("baselines", {})
        for baseline_name in baselines:
            if baseline_name == "apkid" and apkid_skip_reason is not None:
                task_report_bundle[baseline_name] = {"status": "skipped", "reason": apkid_skip_reason}
                continue
            if baseline_name == "ngram_logreg" and ngram_state.get("skipped"):
                task_report_bundle[baseline_name] = {
                    "status": "skipped",
                    "reason": ngram_state.get("skip_reason", "not_prepared"),
                }
                continue
            if baseline_name == "byte_cnn" and byte_cnn_state.get("skipped"):
                task_report_bundle[baseline_name] = {
                    "status": "skipped",
                    "reason": byte_cnn_state.get("skip_reason", "not_prepared"),
                }
                continue
            if baseline_name == "payload_hunter_lite" and lite_state.get("skipped"):
                task_report_bundle[baseline_name] = {
                    "status": "skipped",
                    "reason": lite_state.get("skip_reason", "not_prepared"),
                }
                continue
            if baseline_name == "ours" and ours_state.get("skipped"):
                task_report_bundle[baseline_name] = {
                    "status": "skipped",
                    "reason": ours_state.get("skip_reason", "not_prepared"),
                }
                continue
            if baseline_name == "mil_byte_cnn_fusion" and (
                byte_cnn_state.get("skipped") or ours_state.get("skipped")
            ):
                task_report_bundle[baseline_name] = {
                    "status": "skipped",
                    "reason": "component_not_prepared",
                    "byte_cnn_reason": byte_cnn_state.get("skip_reason"),
                    "ours_reason": ours_state.get("skip_reason"),
                }
                continue

            current_bp = _baseline_fingerprint(baseline_name, state, config)
            if skip_baseline and _baseline_artifacts_reusable(state, baseline_name, current_bp):
                cached_report_path = _resolve_state_path(state[f"{baseline_name}_report_path"])
                try:
                    cached_report = _read_json(cached_report_path)
                except Exception as exc:  # noqa: BLE001
                    cached_report = None
                    print(
                        f"warning: cached report unreadable for "
                        f"{baseline_name}/{state['task_name']}: {exc}; rescoring",
                        file=sys.stderr,
                        flush=True,
                    )
                if cached_report is not None:
                    if dry_run:
                        print(
                            f"would skip score baseline={baseline_name} "
                            f"task={state['task_name']} (fingerprint match)",
                            flush=True,
                        )
                    reports_by_baseline[baseline_name].append(cached_report)
                    task_report_bundle[baseline_name] = {
                        "status": "ok",
                        "metrics": cached_report.get("metrics"),
                        "baseline_report_path": state.get(
                            f"{baseline_name}_report_path"
                        ),
                        "reused": True,
                    }
                    baseline_skipped += 1
                    continue

            if dry_run:
                print(
                    f"would score baseline={baseline_name} task={state['task_name']}",
                    flush=True,
                )
                task_report_bundle[baseline_name] = {"status": "planned"}
                continue

            try:
                report = _score_task_with_baseline(
                    baseline_name=baseline_name,
                    state=state,
                    config=config,
                    ngram_state=ngram_state,
                    byte_cnn_state=byte_cnn_state,
                    lite_state=lite_state,
                    ours_state=ours_state,
                )
                reports_by_baseline[baseline_name].append(report)
                task_report_bundle[baseline_name] = {
                    "status": "ok",
                    "metrics": report.get("metrics"),
                    "baseline_report_path": state.get(
                        f"{baseline_name}_report_path"
                    ),
                    "reused": False,
                }
                _write_fingerprint(
                    _baseline_fingerprint_path(state, baseline_name), current_bp
                )
            except Exception as exc:  # noqa: BLE001 - record per-baseline failure.
                task_report_bundle[baseline_name] = {
                    "status": "failed",
                    "error": str(exc),
                }
                print(
                    f"failed baseline={baseline_name} task={state['task_name']} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise

    if dry_run:
        # Summarise the plan and bail out without writing anything.
        plan = {
            "dry_run": True,
            "task_count": len(task_states),
            "success_count": len(task_states),
            "failure_count": 0,
            "generate_would_skip": generate_skipped,
            "baseline_would_skip": baseline_skipped,
            "baselines": list(baselines),
        }
        print(
            " ".join(
                [
                    f"plan tasks={plan['task_count']}",
                    f"generate_skipped={plan['generate_would_skip']}",
                    f"baseline_skipped={plan['baseline_would_skip']}",
                    f"baselines={','.join(baselines)}",
                ]
            ),
            flush=True,
        )
        return {"experiment_manifest": plan, "summary": plan}

    experiment_manifest = _build_experiment_manifest(
        config,
        seed_manifest_path,
        seed_entries,
        transforms,
        baselines,
        output_root,
        generated_dir,
        synthetic_manifest_dir,
        synthetic_label_dir,
        task_states,
        failures,
        ngram_state,
    )
    experiment_manifest["warnings"] = warnings
    experiment_manifest["reuse"] = {
        "skip_existing": skip_existing,
        "forced_phases": sorted(forced_phases),
        "generate_reused_count": generate_skipped,
        "baseline_reused_count": baseline_skipped,
    }
    summary = _write_outputs(output_root, experiment_manifest, reports_by_baseline)

    print(
        " ".join(
            [
                f"tasks={experiment_manifest['task_count']}",
                f"successes={experiment_manifest['success_count']}",
                f"failures={experiment_manifest['failure_count']}",
                f"baselines={','.join(baselines)}",
                f"generate_reused={generate_skipped}",
                f"baseline_reused={baseline_skipped}",
            ]
        ),
        flush=True,
    )
    return {"experiment_manifest": experiment_manifest, "summary": summary}


# ---------------------------------------------------------------------------
# Per-task helpers
# ---------------------------------------------------------------------------


def _generate_task_artifacts(
    *,
    state: Dict[str, Any],
    config: Mapping[str, Any],
    seed_entry: Mapping[str, Any],
    transform_family: str,
) -> None:
    seed_apk = Path(seed_entry["local_path"])
    regioning = config["regioning"]
    labeling = config["labeling"]

    # Per-task RNG seed derivation (2026-04-29, B2 follow-up): replace
    # the global ``synthetic.rng_seed`` with a task-specific value so
    # two tasks that share a seed APK or transform family still get
    # uncorrelated RNG states. The config-level seed remains the
    # reproducibility anchor; the derived value is what actually feeds
    # :func:`random.Random` inside :func:`build_synthetic_apk`.
    # See ``docs/method/threat_model.md`` §"B2 实装后遗留：task 间
    # layout 一致性" for the problem statement and the formula.
    base_seed = int(config["synthetic"]["rng_seed"])
    effective_seed = derive_task_rng_seed(
        base_seed=base_seed,
        package_name=str(seed_entry.get("package_name") or ""),
        version_code=str(seed_entry.get("version_code") or ""),
        transform_family=transform_family,
    )
    # Record both values in the task state so they flow into the
    # experiment_manifest for end-to-end reproducibility.
    state["rng_seed_base"] = base_seed
    state["rng_seed"] = effective_seed

    synthetic_result = build_synthetic_apk(
        seed_apk=seed_apk,
        generated_apk_out=Path(state["generated_apk_path"]),
        manifest_out=Path(state["synthetic_manifest_path"]),
        labels_out=Path(state["synthetic_labels_path"]),
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

    write_jsonl(Path(state["objects_path"]), object_rows)
    write_jsonl(Path(state["regions_path"]), region_rows)

    labels = build_training_labels(
        regions=region_rows,
        synthetic_labels=[label.to_dict() for label in synthetic_result.labels],
        min_overlap_bytes=int(labeling["min_overlap_bytes"]),
        min_overlap_ratio=float(labeling["min_overlap_ratio"]),
    )
    write_jsonl(
        Path(state["region_labels_path"]),
        (row.to_dict() for row in labels.region_labels),
    )
    write_jsonl(
        Path(state["object_labels_path"]),
        (row.to_dict() for row in labels.object_labels),
    )
    write_jsonl(
        Path(state["apk_labels_path"]),
        (row.to_dict() for row in labels.apk_labels),
    )

    # Cache the in-memory region labels so we don't re-read from disk
    # for each baseline. The lists are small compared to the APK bytes.
    state["_region_label_rows"] = [row.to_dict() for row in labels.region_labels]


def _generate_benign_task_artifacts(
    *,
    state: Dict[str, Any],
    config: Mapping[str, Any],
    seed_entry: Mapping[str, Any],
) -> None:
    """Create an evaluation-only pure-benign control task.

    The generated APK is just a byte-for-byte copy of the seed APK,
    and training labels are built with an empty synthetic label list.
    This yields region/object/APK ``label_id=0`` rows while still
    exercising the same object extraction, regioning, and baseline
    scoring paths as packed synthetic tasks.
    """

    seed_apk = Path(seed_entry["local_path"])
    regioning = config["regioning"]
    labeling = config["labeling"]

    generated_apk = Path(state["generated_apk_path"])
    generated_apk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_apk, generated_apk)
    _write_json(
        Path(state["synthetic_manifest_path"]),
        {
            "schema_version": 1,
            "task_name": state["task_name"],
            "kind": "benign_control",
            "seed_apk_path": str(seed_apk),
            "generated_apk_path": str(generated_apk),
            "package_name": seed_entry.get("package_name"),
            "version_code": seed_entry.get("version_code"),
        },
    )
    write_jsonl(Path(state["synthetic_labels_path"]), [])

    object_rows = []
    region_rows = []
    for metadata, data in iter_apk_objects(
        generated_apk,
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

    write_jsonl(Path(state["objects_path"]), object_rows)
    write_jsonl(Path(state["regions_path"]), region_rows)

    labels = build_training_labels(
        regions=region_rows,
        synthetic_labels=[],
        min_overlap_bytes=int(labeling["min_overlap_bytes"]),
        min_overlap_ratio=float(labeling["min_overlap_ratio"]),
    )
    write_jsonl(
        Path(state["region_labels_path"]),
        (row.to_dict() for row in labels.region_labels),
    )
    write_jsonl(
        Path(state["object_labels_path"]),
        (row.to_dict() for row in labels.object_labels),
    )
    write_jsonl(
        Path(state["apk_labels_path"]),
        (row.to_dict() for row in labels.apk_labels),
    )
    state["_region_label_rows"] = [row.to_dict() for row in labels.region_labels]


def _select_byte_cnn_model_for_task(
    byte_cnn_state: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, Optional[float], Optional[str]]:
    group_key = byte_cnn_state.get("group_key")
    thresholds_by_fold = byte_cnn_state.get("thresholds_by_fold", {})
    threshold: Optional[float] = None
    held_key: Optional[str] = None
    if group_key is None:
        model = byte_cnn_state.get("model")
        if model is None:
            raise RuntimeError("byte_cnn model was not prepared")
        if "same_set" in thresholds_by_fold:
            threshold = float(thresholds_by_fold["same_set"])
    else:
        if state.get("evaluation_only"):
            model = byte_cnn_state.get("model")
            if model is None:
                raise RuntimeError("byte_cnn primary model was not prepared")
            if thresholds_by_fold:
                primary_threshold_key = sorted(thresholds_by_fold)[0]
                threshold = float(thresholds_by_fold[primary_threshold_key])
                held_key = primary_threshold_key
        else:
            if group_key == "transform_family":
                held_key = str(state.get("transform_family", ""))
            else:
                held_key = str(state.get("package_name", ""))
            models_by_fold = byte_cnn_state.get("models_by_fold", {})
            model = models_by_fold.get(held_key)
            if model is None:
                raise RuntimeError(
                    f"byte_cnn fold not trained for {group_key}={held_key!r}; "
                    f"available folds={sorted(models_by_fold)}"
                )
            if held_key in thresholds_by_fold:
                threshold = float(thresholds_by_fold[held_key])
    return model, threshold, held_key


def _select_ours_model_for_task(
    ours_state: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Any:
    group_key = ours_state.get("group_key")
    if group_key is None:
        model = ours_state.get("model")
        if model is None:
            raise RuntimeError("ours model was not prepared")
        return model
    if state.get("evaluation_only"):
        model = ours_state.get("model")
        if model is None:
            raise RuntimeError("ours primary model was not prepared")
        return model
    if group_key == "transform_family":
        held_key = str(state.get("transform_family", ""))
    else:
        held_key = str(state.get("package_name", ""))
    models_by_fold = ours_state.get("models_by_fold", {})
    model = models_by_fold.get(held_key)
    if model is None:
        raise RuntimeError(
            f"ours fold not trained for {group_key}={held_key!r}; "
            f"available folds={sorted(models_by_fold)}"
        )
    return model


def _score_task_with_baseline(
    *,
    baseline_name: str,
    state: Dict[str, Any],
    config: Mapping[str, Any],
    ngram_state: Mapping[str, Any],
    byte_cnn_state: Mapping[str, Any],
    lite_state: Mapping[str, Any],
    ours_state: Mapping[str, Any] = {},
) -> dict:
    region_rows = state["_region_label_rows"]
    baseline_cfg = config.get("baselines", {}).get(baseline_name, {}) if isinstance(config.get("baselines"), dict) else {}

    if baseline_name == "entropy":
        entropy_cfg = EntropyBaselineConfig(
            entropy_threshold=float(baseline_cfg.get("entropy_threshold", 7.0)),
            entropy_weight=float(baseline_cfg.get("entropy_weight", 0.7)),
            nonprintable_weight=float(baseline_cfg.get("nonprintable_weight", 0.3)),
        )
        result = run_entropy_baseline(region_rows, entropy_cfg)
        _write_baseline_outputs(state, "entropy", result)
        return _finalise_task_report(result.report, state, "entropy")

    if baseline_name == "sanity_rules":
        sr_cfg = SanityRulesConfig(**_filter_dataclass_kwargs(
            SanityRulesConfig, baseline_cfg
        ))
        result = run_sanity_rules_baseline(region_rows, sr_cfg)
        _write_baseline_outputs(state, "sanity_rules", result)
        return _finalise_task_report(result.report, state, "sanity_rules")

    if baseline_name == "ngram_logreg":
        model = ngram_state.get("model")
        apk_index = ngram_state.get("apk_index", {})
        if model is None:
            raise RuntimeError("ngram_logreg model was not prepared")
        result = model.predict(region_rows, apk_index)
        _write_baseline_outputs(state, "ngram_logreg", result)
        return _finalise_task_report(result.report, state, "ngram_logreg")

    if baseline_name == "byte_cnn":
        apk_index = byte_cnn_state.get("apk_index", {})
        model, threshold, held_key = _select_byte_cnn_model_for_task(
            byte_cnn_state,
            state,
        )
        result = model.predict(region_rows, apk_index, threshold=threshold)
        if threshold is not None:
            result.report["applied_threshold"] = threshold
            if held_key is not None:
                result.report["calibration_fold"] = held_key
            result.report["calibration_mode"] = byte_cnn_state.get("calibration_mode")
            result.report["calibration_target"] = byte_cnn_state.get("calibration_target")
        _write_baseline_outputs(state, "byte_cnn", result)
        return _finalise_task_report(result.report, state, "byte_cnn")

    if baseline_name == "payload_hunter_lite":
        apk_index = lite_state.get("apk_index", {})
        group_key = lite_state.get("group_key")
        if group_key is None:
            # same_set: single primary model scores every task.
            model = lite_state.get("model")
            if model is None:
                raise RuntimeError("payload_hunter_lite model was not prepared")
        else:
            if state.get("evaluation_only"):
                model = lite_state.get("model")
                if model is None:
                    raise RuntimeError("payload_hunter_lite primary model was not prepared")
            else:
                # Holdout mode: route to the fold whose held-out group
                # matches this task. For holdout_transform we trust
                # state["transform_family"]; for holdout_package we derive
                # from state["package_name"] (set by the seed loader).
                if group_key == "transform_family":
                    held_key = str(state.get("transform_family", ""))
                else:
                    held_key = str(state.get("package_name", ""))
                models_by_fold = lite_state.get("models_by_fold", {})
                model = models_by_fold.get(held_key)
                if model is None:
                    raise RuntimeError(
                        f"payload_hunter_lite fold not trained for "
                        f"{group_key}={held_key!r}; available folds={sorted(models_by_fold)}"
                    )
        result = model.predict(region_rows, apk_index)
        _write_baseline_outputs(state, "payload_hunter_lite", result)
        return _finalise_task_report(
            result.report, state, "payload_hunter_lite"
        )

    if baseline_name == "ours":
        apk_index = ours_state.get("apk_index", {})
        model = _select_ours_model_for_task(ours_state, state)
        result = model.predict(region_rows, apk_index)
        _write_baseline_outputs(state, "ours", result)
        return _finalise_task_report(result.report, state, "ours")

    if baseline_name == "mil_byte_cnn_fusion":
        byte_cnn_model, byte_threshold, held_key = _select_byte_cnn_model_for_task(
            byte_cnn_state,
            state,
        )
        ours_model = _select_ours_model_for_task(ours_state, state)
        byte_result = byte_cnn_model.predict(
            region_rows,
            byte_cnn_state.get("apk_index", {}),
            threshold=byte_threshold,
        )
        mil_result = ours_model.predict(region_rows, ours_state.get("apk_index", {}))
        fusion_cfg = MilByteCnnFusionConfig(
            mil_weight=float(baseline_cfg.get("mil_weight", 0.5)),
            byte_cnn_weight=float(baseline_cfg.get("byte_cnn_weight", 0.5)),
            threshold=float(baseline_cfg.get("threshold", 0.5)),
            score_transform=str(baseline_cfg.get("score_transform", "identity")),
        )
        result = fuse_mil_and_byte_cnn_results(
            mil_result=mil_result,
            byte_cnn_result=byte_result,
            config=fusion_cfg,
        )
        result.report["components"] = {
            "mil": {
                "baseline": "ours",
                "train_mode": ours_state.get("train_mode"),
            },
            "byte_cnn": {
                "baseline": "byte_cnn",
                "train_mode": byte_cnn_state.get("train_mode"),
                "applied_threshold": byte_threshold,
                "calibration_fold": held_key,
                "calibration_mode": byte_cnn_state.get("calibration_mode"),
                "calibration_target": byte_cnn_state.get("calibration_target"),
            },
        }
        _write_baseline_outputs(state, "mil_byte_cnn_fusion", result)
        return _finalise_task_report(
            result.report,
            state,
            "mil_byte_cnn_fusion",
        )

    if baseline_name == "apkid":
        from android_packer.baselines import (
            ApkidBaselineConfig,
            run_apkid_baseline,
        )

        apk_label_rows = list(read_jsonl(Path(state["apk_labels_path"])))
        true_label_id = int(apk_label_rows[0].get("label_id", 0)) if apk_label_rows else 0
        entries = [
            {
                "apk_id": state["task_name"],
                "apk_path": state["generated_apk_path"],
                "true_label_id": true_label_id,
            }
        ]
        apkid_cfg = ApkidBaselineConfig(
            include_aux_categories=bool(baseline_cfg.get("include_aux_categories", False)),
            min_hits=int(baseline_cfg.get("min_hits", 1)),
            timeout_seconds=float(baseline_cfg.get("timeout_seconds", 120.0)),
        )
        result = run_apkid_baseline(entries, config=apkid_cfg)
        # APKiD only emits apk-level predictions.
        write_jsonl(
            Path(state["apkid_apk_predictions_path"]),
            (row.to_dict() for row in result.apk_predictions),
        )
        return _finalise_task_report(result.report, state, "apkid")

    raise ValueError(f"unsupported baseline name: {baseline_name}")


def _write_baseline_outputs(state: Dict[str, Any], baseline_name: str, result) -> None:
    write_jsonl(
        Path(state[f"{baseline_name}_region_predictions_path"]),
        (row.to_dict() for row in result.region_predictions),
    )
    write_jsonl(
        Path(state[f"{baseline_name}_object_predictions_path"]),
        (row.to_dict() for row in result.object_predictions),
    )
    write_jsonl(
        Path(state[f"{baseline_name}_apk_predictions_path"]),
        (row.to_dict() for row in result.apk_predictions),
    )


def _finalise_task_report(
    report: dict,
    state: Dict[str, Any],
    baseline_name: str,
) -> dict:
    final = {
        **report,
        "task_name": state["task_name"],
        "transform_family": state["transform_family"],
    }
    report_path = Path(state[f"{baseline_name}_report_path"])
    _write_json(report_path, final)
    return final


def _prepare_ngram_model(
    *,
    task_states: Sequence[Mapping],
    config: Mapping[str, Any],
    output_root: Path,
    warnings: list,
) -> Dict[str, Any]:
    """Train a single n-gram LR model on the union of every task's labels.

    This is the ``same_set`` training mode mentioned in the module
    docstring: the model sees every task at train time and is then
    scored on each task, which means the numbers reported for this
    baseline are in-sample. We pin the caveat in ``warnings`` so it
    appears in the experiment manifest.
    """

    train_mode = str(config.get("train_mode", "same_set"))
    if train_mode != "same_set":
        raise ValueError(
            f"ngram_logreg train_mode={train_mode!r} is not yet supported; "
            "use 'same_set' until holdout modes land."
        )
    warnings.append(
        "ngram_logreg train_mode=same_set: numbers are in-sample and "
        "overstate generalisation; see docs/progress for the roadmap "
        "to holdout_transform / holdout_package modes."
    )

    rows: list[dict] = []
    apk_index: Dict[str, Path] = {}
    for state in task_states:
        if state.get("status") != "ok":
            continue
        apk_index[state["task_name"]] = Path(state["generated_apk_path"])
        if state.get("evaluation_only"):
            continue
        # Training rows must carry an apk_id that matches the index;
        # the region labels produced by build_training_labels carry
        # seed-derived apk_id values, so remap them to task_name.
        for row in state.get("_region_label_rows", []):
            remapped = dict(row)
            remapped["apk_id"] = state["task_name"]
            rows.append(remapped)

    # Mirror the remapping into the per-task region_label_rows so the
    # inference phase also sees apk_id == task_name (necessary for
    # ObjectByteLoader to hit the APK on disk).
    for state in task_states:
        if state.get("status") != "ok":
            continue
        state["_region_label_rows"] = [
            {**row, "apk_id": state["task_name"]}
            for row in state.get("_region_label_rows", [])
        ]

    max_train_rows_raw = config.get(
        "max_train_rows", DEFAULT_NGRAM_MAX_TRAIN_ROWS
    )
    if max_train_rows_raw is not None:
        max_train_rows = int(max_train_rows_raw)
        if len(rows) > max_train_rows:
            warnings.append(
                "ngram_logreg skipped: train_row_count="
                f"{len(rows)} exceeds max_train_rows={max_train_rows}. "
                "This guard prevents workstation OOM; set "
                "baselines.ngram_logreg.max_train_rows=null only on a "
                "high-memory machine."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "max_train_rows_exceeded",
                "max_train_rows": max_train_rows,
            }

    feature_config = ByteFeatureConfig(
        include_unigram=bool(config.get("include_unigram", True)),
        include_bigram=bool(config.get("include_bigram", True)),
        include_scalars=bool(config.get("include_scalars", True)),
        bigram_hash_dim=int(config.get("bigram_hash_dim", 1024)),
    )
    cfg = NgramLogRegConfig(
        feature_config=feature_config,
        C=float(config.get("C", 1.0)),
        max_iter=int(config.get("max_iter", 1000)),
        class_weight=(
            "balanced"
            if str(config.get("class_weight", "balanced")) == "balanced"
            else None  # type: ignore[arg-type]
        ),
        random_state=int(config.get("random_state", 0)),
        threshold=float(config.get("threshold", 0.5)),
        loader_cache_size=int(config.get("loader_cache_size", 64)),
        use_hashing_vectorizer=bool(config.get("use_hashing_vectorizer", True)),
        hashing_n_features=int(config.get("hashing_n_features", 262144)),
    )

    try:
        model = train_ngram_logreg(rows, apk_index, config=cfg)
    except ValueError as exc:
        warnings.append(f"ngram_logreg skipped: {exc}")
        return {}

    model_path = output_root / "models" / "ngram_logreg.pkl"
    model.save(model_path)
    return {
        "model": model,
        "model_path": str(model_path),
        "apk_index": apk_index,
        "train_mode": train_mode,
        "train_row_count": len(rows),
    }


def _package_name_from_task(
    apk_id: str,
    task_states: Sequence[Mapping[str, Any]],
) -> str:
    """Resolve the package_name of a task by its apk_id / task_name.

    Used as a last-resort fallback when the region_training_label row
    does not carry ``package_name`` (e.g. older label schemas). Scans
    ``task_states`` which the multi-baseline runner already keeps in
    memory. Returns an empty string on miss so the caller can still
    partition rows deterministically (with one synthetic "" group).
    """
    for ts in task_states:
        if str(ts.get("task_name")) == apk_id or str(ts.get("apk_id")) == apk_id:
            pn = ts.get("package_name")
            if pn:
                return str(pn)
    return ""


def _stable_row_key(row: Mapping[str, Any]) -> str:
    """Stable identity used for deterministic sampled training subsets."""

    return "|".join(
        str(row.get(key, ""))
        for key in (
            "apk_id",
            "object_path",
            "object_id",
            "region_id",
            "offset_start",
            "offset_end",
            "transform_family",
            "package_name",
            "label_id",
        )
    )


def _sample_byte_cnn_train_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_rows: Optional[int],
    min_positive_rows: int,
    group_key: Optional[str],
    random_state: int,
) -> List[Dict[str, Any]]:
    """Deterministically downsample byte-CNN training rows.

    Sampling is training-only: held-out evaluation rows are never
    sampled. Rows are stratified by the fold grouping key and label so
    that larger all-family runs can stay under the workstation safety
    guard without narrowing the held-out protocol itself.
    """

    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)

    target = max(1, int(max_rows))
    rng = random.Random(int(random_state))
    indexed = list(enumerate(rows))
    selected: set[int] = set()

    positive = [(idx, row) for idx, row in indexed if int(row.get("label_id", 0)) == 1]
    if positive:
        keep_pos = min(len(positive), max(target // 2, int(min_positive_rows)))
        keep_pos = min(keep_pos, target)
        selected.update(idx for idx, _ in rng.sample(positive, keep_pos))

    strata: Dict[tuple, List[tuple[int, Dict[str, Any]]]] = {}
    for idx, row in indexed:
        if idx in selected:
            continue
        group_value = str(row.get(group_key, "")) if group_key else ""
        label = int(row.get("label_id", 0))
        strata.setdefault((group_value, label), []).append((idx, row))

    while len(selected) < target and strata:
        progressed = False
        for key in sorted(strata):
            bucket = strata[key]
            if not bucket:
                continue
            pick_pos = rng.randrange(len(bucket))
            idx, _ = bucket.pop(pick_pos)
            selected.add(idx)
            progressed = True
            if len(selected) >= target:
                break
        strata = {key: bucket for key, bucket in strata.items() if bucket}
        if not progressed:
            break

    sampled = [rows[idx] for idx in selected]
    sampled.sort(key=_stable_row_key)
    return sampled


def _split_byte_cnn_fit_and_calibration_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    validation_fraction: float,
    min_validation_rows: int,
    min_positive_rows: int,
    group_key: Optional[str],
    random_state: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split fold-training rows into model-fit rows and threshold-validation rows."""

    rows_list = list(rows)
    if validation_fraction <= 0.0 or len(rows_list) < 2:
        return rows_list, []

    target = int(round(len(rows_list) * validation_fraction))
    target = max(1, int(min_validation_rows), target)
    target = min(target, len(rows_list) - 1)
    if target <= 0:
        return rows_list, []

    rng = random.Random(int(random_state))
    indexed = list(enumerate(rows_list))
    positives = [(idx, row) for idx, row in indexed if int(row.get("label_id", 0)) == 1]
    negatives = [(idx, row) for idx, row in indexed if int(row.get("label_id", 0)) == 0]
    selected: set[int] = set()

    if positives and target > 0:
        max_positive_pick = max(0, len(positives) - 1)
        keep_pos = min(max_positive_pick, target, max(1, int(min_positive_rows)))
        if keep_pos:
            selected.update(idx for idx, _ in rng.sample(positives, keep_pos))
    if negatives and len(selected) < target:
        max_negative_pick = max(0, len(negatives) - 1)
        keep_neg = min(max_negative_pick, target - len(selected), max(1, target // 2))
        if keep_neg:
            selected.update(idx for idx, _ in rng.sample(negatives, keep_neg))

    strata: Dict[tuple, List[tuple[int, Dict[str, Any]]]] = {}
    for idx, row in indexed:
        if idx in selected:
            continue
        group_value = str(row.get(group_key, "")) if group_key else ""
        label = int(row.get("label_id", 0))
        strata.setdefault((group_value, label), []).append((idx, row))

    fit_label_counts: Dict[int, int] = {}
    for idx, row in indexed:
        if idx in selected:
            continue
        label = int(row.get("label_id", 0))
        fit_label_counts[label] = fit_label_counts.get(label, 0) + 1
    required_fit_label_count = min(2, len({int(row.get("label_id", 0)) for _, row in indexed}))

    while len(selected) < target and strata:
        progressed = False
        for key in sorted(strata):
            bucket = strata[key]
            if not bucket:
                continue
            pick_pos = rng.randrange(len(bucket))
            idx, row = bucket.pop(pick_pos)
            label = int(row.get("label_id", 0))
            remaining_count = fit_label_counts.get(label, 0) - 1
            surviving_label_count = sum(
                1
                for candidate_label, count in fit_label_counts.items()
                if count > 0 and (candidate_label != label or remaining_count > 0)
            )
            if surviving_label_count < required_fit_label_count:
                continue
            selected.add(idx)
            if remaining_count > 0:
                fit_label_counts[label] = remaining_count
            else:
                fit_label_counts.pop(label, None)
            progressed = True
            if len(selected) >= target:
                break
        strata = {key: bucket for key, bucket in strata.items() if bucket}
        if not progressed:
            break

    calibration_rows = [rows_list[idx] for idx in selected]
    fit_rows = [row for idx, row in enumerate(rows_list) if idx not in selected]
    calibration_rows.sort(key=_stable_row_key)
    fit_rows.sort(key=_stable_row_key)
    return fit_rows, calibration_rows


def _byte_cnn_calibrated_threshold_from_report(
    report: Mapping[str, Any],
    *,
    target: str,
    default_threshold: float,
) -> float:
    calibration = report.get("calibration", {}) if isinstance(report, Mapping) else {}
    block = calibration.get(target, {}) if isinstance(calibration, Mapping) else {}
    best = block.get("best_f1") if isinstance(block, Mapping) else None
    if isinstance(best, Mapping) and best.get("threshold") is not None:
        return float(best["threshold"])
    return float(default_threshold)


def _prepare_byte_cnn_model(
    *,
    task_states: Sequence[Mapping],
    config: Mapping[str, Any],
    output_root: Path,
    warnings: list,
) -> Dict[str, Any]:
    """Train byte-CNN under same_set / holdout_transform / holdout_package.

    The model streams raw region bytes from ``ObjectByteLoader`` inside
    its training loop, so this prepare step only builds the row union,
    remaps ``apk_id`` to the generated task name, partitions holdout
    folds, and persists the primary checkpoint for provenance.
    """

    train_mode = str(config.get("train_mode", "same_set"))
    if train_mode not in ("same_set", "holdout_transform", "holdout_package"):
        raise ValueError(
            f"byte_cnn train_mode={train_mode!r} is not one of "
            "'same_set', 'holdout_transform', 'holdout_package'."
        )
    if train_mode == "same_set":
        warnings.append(
            "byte_cnn train_mode=same_set: numbers are in-sample; set "
            "train_mode=holdout_transform for honest out-of-distribution numbers."
        )
    else:
        warnings.append(
            f"byte_cnn train_mode={train_mode}: per-task reports come from "
            "the fold where that task was the held-out group; aggregate "
            "across tasks for a macro metric."
        )

    rows: List[Dict[str, Any]] = []
    apk_index: Dict[str, Path] = {}
    for state in task_states:
        if state.get("status") != "ok":
            continue
        apk_index[state["task_name"]] = Path(state["generated_apk_path"])
        if state.get("evaluation_only"):
            continue
        for row in state.get("_region_label_rows", []):
            remapped = dict(row)
            if remapped.get("apk_id") != state["task_name"]:
                remapped["apk_id"] = state["task_name"]
            remapped.setdefault("transform_family", state["transform_family"])
            rows.append(remapped)
    for state in task_states:
        if state.get("status") != "ok":
            continue
        state["_region_label_rows"] = [
            {
                **row,
                "apk_id": state["task_name"],
                "transform_family": row.get(
                    "transform_family", state["transform_family"]
                ),
            }
            for row in state.get("_region_label_rows", [])
        ]

    max_train_rows_raw = config.get(
        "max_train_rows", DEFAULT_BYTE_CNN_MAX_TRAIN_ROWS
    )
    max_train_rows = (
        None if max_train_rows_raw is None else int(max_train_rows_raw)
    )
    train_sample_rows_raw = config.get("train_sample_rows")
    train_sample_rows = (
        None if train_sample_rows_raw is None else int(train_sample_rows_raw)
    )
    train_sample_min_positive_rows = int(
        config.get(
            "train_sample_min_positive_rows",
            DEFAULT_BYTE_CNN_TRAIN_SAMPLE_MIN_POSITIVE_ROWS,
        )
    )
    calibration_mode = str(config.get("calibration_mode", "fixed_threshold"))
    if calibration_mode not in ("fixed_threshold", "fold_local_best_f1"):
        raise ValueError(
            "byte_cnn calibration_mode must be 'fixed_threshold' or "
            f"'fold_local_best_f1'; got {calibration_mode!r}."
        )
    calibration_target = str(config.get("calibration_target", "object"))
    if calibration_target not in ("region", "object", "apk"):
        raise ValueError(
            "byte_cnn calibration_target must be one of 'region', 'object', or 'apk'; "
            f"got {calibration_target!r}."
        )
    validation_fraction = float(config.get("calibration_validation_fraction", 0.1))
    validation_min_rows = int(config.get("calibration_min_validation_rows", 256))
    validation_min_positive_rows = int(config.get("calibration_min_positive_rows", 32))
    if max_train_rows is not None and train_sample_rows is None:
        if len(rows) > max_train_rows:
            warnings.append(
                "byte_cnn skipped: train_row_count="
                f"{len(rows)} exceeds max_train_rows={max_train_rows}. "
                "This guard prevents workstation OOM; set "
                "baselines.byte_cnn.max_train_rows=null only on a "
                "high-memory machine, or set train_sample_rows to train "
                "on a deterministic fold-local subset."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "max_train_rows_exceeded",
                "max_train_rows": max_train_rows,
            }

    model_keys = {
        "max_length",
        "embedding_dim",
        "conv_channels",
        "kernel_sizes",
        "hidden_dim",
        "dropout",
        "pad_token_id",
        "activation",
    }
    model_kwargs = {key: config[key] for key in model_keys if key in config}
    if "kernel_sizes" in model_kwargs:
        model_kwargs["kernel_sizes"] = tuple(int(k) for k in model_kwargs["kernel_sizes"])
    model_config = ByteCnnRegionScorerConfig(**model_kwargs)

    cfg_kwargs: Dict[str, Any] = {
        "train_mode": "same_set",
        "model_config": model_config,
    }
    base_cfg = ByteCnnBaselineConfig()
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "positive_class_weight",
        "random_state",
        "verbose",
        "threshold",
        "loader_cache_size",
        "device",
    ):
        if key in config:
            cfg_kwargs[key] = type(getattr(base_cfg, key))(config[key])
    inner_cfg = ByteCnnBaselineConfig(**cfg_kwargs)

    models_by_fold: Dict[str, Any] = {}
    primary_model: Any = None
    fold_train_row_counts: Dict[str, int] = {}
    fold_sampled_train_row_counts: Dict[str, int] = {}
    thresholds_by_fold: Dict[str, float] = {}
    calibration_validation_row_counts: Dict[str, int] = {}

    if train_mode == "same_set":
        train_rows = _sample_byte_cnn_train_rows(
            rows,
            max_rows=train_sample_rows,
            min_positive_rows=train_sample_min_positive_rows,
            group_key=None,
            random_state=int(inner_cfg.random_state),
        )
        fold_train_row_counts["same_set"] = len(rows)
        fold_sampled_train_row_counts["same_set"] = len(train_rows)
        calibration_rows: List[Dict[str, Any]] = []
        if calibration_mode == "fold_local_best_f1":
            train_rows, calibration_rows = _split_byte_cnn_fit_and_calibration_rows(
                train_rows,
                validation_fraction=validation_fraction,
                min_validation_rows=validation_min_rows,
                min_positive_rows=validation_min_positive_rows,
                group_key=None,
                random_state=int(inner_cfg.random_state) * 1009 + 17,
            )
            calibration_validation_row_counts["same_set"] = len(calibration_rows)
        if max_train_rows is not None and len(train_rows) > max_train_rows:
            warnings.append(
                "byte_cnn skipped: train_row_count="
                f"{len(train_rows)} exceeds max_train_rows={max_train_rows}. "
                "Set train_sample_rows <= max_train_rows for sampled runs."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "sampled_train_row_count": len(train_rows),
                "skipped": True,
                "skip_reason": "max_train_rows_exceeded",
                "max_train_rows": max_train_rows,
                "train_sample_rows": train_sample_rows,
            }
        if len(train_rows) < len(rows):
            warnings.append(
                "byte_cnn training sampled: train_row_count="
                f"{len(rows)} sampled_train_row_count={len(train_rows)}. "
                "Evaluation still uses full task rows."
            )
        try:
            primary_model = train_byte_cnn(train_rows, apk_index, inner_cfg)
            if calibration_rows:
                calibration_result = primary_model.predict(calibration_rows, apk_index)
                thresholds_by_fold["same_set"] = _byte_cnn_calibrated_threshold_from_report(
                    calibration_result.report,
                    target=calibration_target,
                    default_threshold=inner_cfg.threshold,
                )
        except ValueError as exc:
            warnings.append(f"byte_cnn skipped: {exc}")
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "sampled_train_row_count": len(train_rows),
                "skipped": True,
                "skip_reason": str(exc),
            }
        except ImportError as exc:
            warnings.append(
                f"byte_cnn skipped: torch not installed ({exc}). "
                "Run ``pip install -e .[dl]`` to enable this baseline."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "torch_not_installed",
            }
    else:
        group_key = (
            "transform_family" if train_mode == "holdout_transform" else "package_name"
        )
        if group_key == "package_name":
            for row in rows:
                if "package_name" not in row:
                    row["package_name"] = _package_name_from_task(
                        str(row.get("apk_id", "")), task_states
                    )

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)
        if len(groups) < 2:
            warnings.append(
                f"byte_cnn skipped: train_mode={train_mode!r} needs >= 2 "
                f"{group_key!r} values; got {sorted(groups)}."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "insufficient_groups",
                "group_key": group_key,
            }

        held_values = sorted(groups)
        for fold_idx, held in enumerate(held_values, start=1):
            full_train_rows = [
                row for group, items in groups.items() if group != held for row in items
            ]
            sample_seed = int(inner_cfg.random_state) * 1009 + fold_idx
            train_rows = _sample_byte_cnn_train_rows(
                full_train_rows,
                max_rows=train_sample_rows,
                min_positive_rows=train_sample_min_positive_rows,
                group_key=group_key,
                random_state=sample_seed,
            )
            fold_train_row_counts[held] = len(full_train_rows)
            fold_sampled_train_row_counts[held] = len(train_rows)
            calibration_rows: List[Dict[str, Any]] = []
            if calibration_mode == "fold_local_best_f1":
                train_rows, calibration_rows = _split_byte_cnn_fit_and_calibration_rows(
                    train_rows,
                    validation_fraction=validation_fraction,
                    min_validation_rows=validation_min_rows,
                    min_positive_rows=validation_min_positive_rows,
                    group_key=group_key,
                    random_state=sample_seed * 1009 + 17,
                )
                calibration_validation_row_counts[held] = len(calibration_rows)
            if max_train_rows is not None and len(train_rows) > max_train_rows:
                warnings.append(
                    "byte_cnn fold "
                    f"{group_key}={held!r} skipped: train_row_count="
                    f"{len(train_rows)} exceeds max_train_rows={max_train_rows}. "
                    "Set train_sample_rows <= max_train_rows for sampled runs."
                )
                continue
            print(
                "train byte_cnn fold="
                f"{fold_idx}/{len(held_values)} {group_key}={held!r} "
                f"train_rows={len(train_rows)} full_train_rows={len(full_train_rows)} "
                f"heldout_rows={len(groups[held])}",
                flush=True,
            )
            try:
                fold_model = train_byte_cnn(train_rows, apk_index, inner_cfg)
                if calibration_rows:
                    calibration_result = fold_model.predict(calibration_rows, apk_index)
                    thresholds_by_fold[held] = _byte_cnn_calibrated_threshold_from_report(
                        calibration_result.report,
                        target=calibration_target,
                        default_threshold=inner_cfg.threshold,
                    )
            except ValueError as exc:
                warnings.append(
                    f"byte_cnn fold {group_key}={held!r} skipped: {exc}"
                )
                continue
            except ImportError as exc:
                warnings.append(
                    f"byte_cnn skipped: torch not installed ({exc}). "
                    "Run ``pip install -e .[dl]`` to enable this baseline."
                )
                return {
                    "train_mode": train_mode,
                    "train_row_count": len(rows),
                    "skipped": True,
                    "skip_reason": "torch_not_installed",
                    "group_key": group_key,
                }
            print(
                "finished byte_cnn fold="
                f"{fold_idx}/{len(held_values)} {group_key}={held!r}",
                flush=True,
            )
            models_by_fold[held] = fold_model
            if primary_model is None:
                primary_model = fold_model

        if not models_by_fold:
            warnings.append("byte_cnn skipped: no folds produced a trained model.")
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "no_trained_folds",
                "group_key": group_key,
                "fold_train_row_counts": fold_train_row_counts,
                "fold_sampled_train_row_counts": fold_sampled_train_row_counts,
            }

    if train_sample_rows is not None:
        sampled_counts = sorted(set(fold_sampled_train_row_counts.values()))
        warnings.append(
            "byte_cnn training sampled: train_sample_rows="
            f"{train_sample_rows} fold_sampled_train_row_counts={sampled_counts}. "
            "Held-out evaluation rows are not sampled."
        )
    if calibration_mode == "fold_local_best_f1":
        warnings.append(
            "byte_cnn fold-local calibration enabled: target="
            f"{calibration_target} thresholds_by_fold={thresholds_by_fold}. "
            "Validation rows are drawn only from each training complement."
        )

    model_path = output_root / "models" / "byte_cnn.pt"
    if primary_model is not None:
        try:
            primary_model.save(model_path)
        except ImportError:
            pass

    return {
        "model": primary_model,
        "models_by_fold": models_by_fold,
        "model_path": str(model_path),
        "apk_index": apk_index,
        "train_mode": train_mode,
        "train_row_count": len(rows),
        "train_sample_rows": train_sample_rows,
        "train_sample_min_positive_rows": train_sample_min_positive_rows,
        "fold_train_row_counts": fold_train_row_counts,
        "fold_sampled_train_row_counts": fold_sampled_train_row_counts,
        "calibration_mode": calibration_mode,
        "calibration_target": calibration_target,
        "thresholds_by_fold": thresholds_by_fold,
        "calibration_validation_row_counts": calibration_validation_row_counts,
        "group_key": (
            "transform_family"
            if train_mode == "holdout_transform"
            else ("package_name" if train_mode == "holdout_package" else None)
        ),
    }



def _prepare_payload_hunter_lite_model(
    *,
    task_states: Sequence[Mapping],
    config: Mapping[str, Any],
    output_root: Path,
    warnings: list,
) -> Dict[str, Any]:
    """Train a single PayloadHunter-Lite model on the union of every task.

    Mirrors :func:`_prepare_ngram_model`: `same_set` mode trains on the
    full region-label union and scores each task in-sample. Numbers
    are in-sample and a warning is appended to the experiment manifest
    so downstream readers don't mistake them for generalisation.

    ``apk_id -> task_name`` remapping is already done by
    :func:`_prepare_ngram_model` for ``ngram_logreg``; we reuse the
    remapped rows and the same ``apk_index`` if available to avoid
    touching state twice. When ``ngram_logreg`` is not enabled we
    perform the remap here inline.
    """

    train_mode = str(config.get("train_mode", "same_set"))
    if train_mode not in ("same_set", "holdout_transform", "holdout_package"):
        raise ValueError(
            f"payload_hunter_lite train_mode={train_mode!r} is not one of "
            "'same_set', 'holdout_transform', 'holdout_package'."
        )
    if train_mode == "same_set":
        warnings.append(
            "payload_hunter_lite train_mode=same_set: numbers are in-sample; "
            "run scripts/sweep_baselines.py or set train_mode=holdout_transform "
            "for honest out-of-distribution numbers."
        )
    else:
        # Holdout modes: numbers are honest OOD, but per-task reports
        # reflect the fold's test split. Record the hold-out mode in
        # the manifest warnings so downstream aggregators know which
        # fold contributed each row.
        warnings.append(
            f"payload_hunter_lite train_mode={train_mode}: per-task reports "
            "come from the fold where that task was the held-out group; "
            "aggregate across tasks for a macro metric."
        )

    # Remap apk_id -> task_name if the ngram prepare step didn't run.
    rows: List[Dict[str, Any]] = []
    apk_index: Dict[str, Path] = {}
    for state in task_states:
        if state.get("status") != "ok":
            continue
        apk_index[state["task_name"]] = Path(state["generated_apk_path"])
        for row in state.get("_region_label_rows", []):
            remapped = dict(row)
            if remapped.get("apk_id") != state["task_name"]:
                remapped["apk_id"] = state["task_name"]
            # Propagate transform_family so a future holdout_transform
            # wiring has the right group key on every row without
            # re-reading manifests.
            remapped.setdefault("transform_family", state["transform_family"])
            rows.append(remapped)
    # Also sync the per-task rows so inference sees the same apk_id
    # mapping (idempotent if ngram already did it).
    for state in task_states:
        if state.get("status") != "ok":
            continue
        state["_region_label_rows"] = [
            {
                **row,
                "apk_id": state["task_name"],
                "transform_family": row.get(
                    "transform_family", state["transform_family"]
                ),
            }
            for row in state.get("_region_label_rows", [])
        ]

    max_train_rows_raw = config.get(
        "max_train_rows", DEFAULT_LITE_MAX_TRAIN_ROWS
    )
    if max_train_rows_raw is not None:
        max_train_rows = int(max_train_rows_raw)
        if len(rows) > max_train_rows:
            warnings.append(
                "payload_hunter_lite skipped: train_row_count="
                f"{len(rows)} exceeds max_train_rows={max_train_rows}. "
                "This guard prevents workstation OOM; set "
                "baselines.payload_hunter_lite.max_train_rows=null only "
                "on a high-memory machine."
            )
            return {
                "train_mode": train_mode,
                "train_row_count": len(rows),
                "skipped": True,
                "skip_reason": "max_train_rows_exceeded",
                "max_train_rows": max_train_rows,
            }

    # Build the PayloadHunterLiteConfig from the JSON sub-block. Only
    # the flat top-level scalars are honoured here; nested
    # feature/model configs keep their defaults. Ablation-level knobs
    # will be wired via F-Lite-e.
    cfg_kwargs: Dict[str, Any] = {
        "train_mode": train_mode,
    }
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "object_loss_weight",
        "positive_class_weight",
        "random_state",
        "threshold",
        "loader_cache_size",
        "device",
    ):
        if key in config:
            cfg_kwargs[key] = type(getattr(PayloadHunterLiteConfig(), key))(config[key])
    # Training helpers always use same_set under the hood; the fold
    # selection is done here.
    inner_cfg = PayloadHunterLiteConfig(**{**cfg_kwargs, "train_mode": "same_set"})

    models_by_fold: Dict[str, Any] = {}
    primary_model: Any = None

    if train_mode == "same_set":
        try:
            primary_model = train_payload_hunter_lite(rows, apk_index, inner_cfg)
        except ValueError as exc:
            warnings.append(f"payload_hunter_lite skipped: {exc}")
            return {}
        except ImportError as exc:
            warnings.append(
                f"payload_hunter_lite skipped: torch not installed ({exc}). "
                "Run ``pip install -e .[dl]`` to enable this baseline."
            )
            return {}
    else:
        # holdout_transform -> group by transform_family.
        # holdout_package  -> group by package_name; derive from
        #   state["package_name"] which multi_baseline populates.
        group_key = (
            "transform_family" if train_mode == "holdout_transform" else "package_name"
        )
        # Partition rows by group key. Rows already carry the group
        # key (we propagate transform_family above; package_name is
        # derived below when missing).
        if group_key == "package_name":
            for row in rows:
                if "package_name" not in row:
                    # Prefer the state-level package_name when available
                    # via the task-to-package index built below.
                    row["package_name"] = _package_name_from_task(
                        str(row.get("apk_id", "")), task_states
                    )

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)
        if len(groups) < 2:
            warnings.append(
                f"payload_hunter_lite skipped: train_mode={train_mode!r} "
                f"needs >= 2 {group_key!r} values; got {sorted(groups)}."
            )
            return {}

        for held in sorted(groups):
            train_rows = [
                r for g, items in groups.items() if g != held for r in items
            ]
            try:
                fold_model = train_payload_hunter_lite(
                    train_rows, apk_index, inner_cfg
                )
            except ValueError as exc:
                warnings.append(
                    f"payload_hunter_lite fold {group_key}={held!r} skipped: {exc}"
                )
                continue
            except ImportError as exc:
                warnings.append(
                    f"payload_hunter_lite skipped: torch not installed ({exc}). "
                    "Run ``pip install -e .[dl]`` to enable this baseline."
                )
                return {}
            models_by_fold[held] = fold_model
            if primary_model is None:
                primary_model = fold_model

        if not models_by_fold:
            warnings.append(
                "payload_hunter_lite skipped: no folds produced a trained model."
            )
            return {}

    # Persist the primary model for provenance. Per-fold models are
    # kept in-memory only; if the user wants them on disk they can
    # subclass this prep function, but the sweep script (which does
    # fold-granular analysis) writes its own fold artefacts.
    model_path = output_root / "models" / "payload_hunter_lite.pt"
    if primary_model is not None:
        try:
            primary_model.save(model_path)
        except ImportError:
            pass

    return {
        "model": primary_model,
        "models_by_fold": models_by_fold,
        "model_path": str(model_path),
        "apk_index": apk_index,
        "train_mode": train_mode,
        "train_row_count": len(rows),
        "group_key": (
            "transform_family"
            if train_mode == "holdout_transform"
            else ("package_name" if train_mode == "holdout_package" else None)
        ),
    }


def _prepare_ours_model(
    *,
    task_states: Sequence[Mapping],
    config: Mapping[str, Any],
    output_root: Path,
    warnings: list,
) -> Dict[str, Any]:
    """Train an Ours (Typed-Instance MIL) model under same_set /
    holdout_transform / holdout_package train modes.

    Mirrors :func:`_prepare_payload_hunter_lite_model` one-to-one so
    downstream scoring can dispatch through the same ``group_key``
    routing.  Paper-integrity notes (2026-05-07):

    - ``supervision_mode`` defaults to ``"bag"`` here (Tier-A quotable
      regime).  ``instance_aided`` is allowed for diagnostic runs but
      emits a warning so the experiment manifest is self-describing.
      See ``docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md``
      §7.3 (L43).
    - The ``holdout_transform`` mode here is the LOFO-by-family
      evaluation that §7.4 (L45) calls for.  ``data/synthetic/splits_v4_lofo/``
      remains the external, deterministic split artefact for sweeps
      that need fold-level provenance; this runner does the split
      in-memory for speed.
    """

    train_mode = str(config.get("train_mode", "same_set"))
    if train_mode not in ("same_set", "holdout_transform", "holdout_package"):
        raise ValueError(
            f"ours train_mode={train_mode!r} is not one of "
            "'same_set', 'holdout_transform', 'holdout_package'."
        )
    supervision_mode = str(config.get("supervision_mode", "bag"))
    if supervision_mode not in ("bag", "instance_aided"):
        raise ValueError(
            f"ours supervision_mode={supervision_mode!r} is not one of "
            "'bag', 'instance_aided'."
        )
    if supervision_mode == "instance_aided":
        warnings.append(
            "ours supervision_mode=instance_aided: numbers include "
            "per-instance BCE; NOT the weakly-supervised claim. Do not "
            "quote these as Tier-A in the paper — see AGENTS.md §8.1."
        )
    if train_mode == "same_set":
        warnings.append(
            "ours train_mode=same_set: numbers are in-sample; set "
            "train_mode=holdout_transform for honest LOFO numbers."
        )

    # Build (rows, apk_index) -- identical shape to payload_hunter_lite
    # prepare.  Remap apk_id -> task_name so ObjectByteLoader hits.
    rows: List[Dict[str, Any]] = []
    apk_index: Dict[str, Path] = {}
    for state in task_states:
        if state.get("status") != "ok":
            continue
        apk_index[state["task_name"]] = Path(state["generated_apk_path"])
        if state.get("evaluation_only"):
            continue
        for row in state.get("_region_label_rows", []):
            remapped = dict(row)
            if remapped.get("apk_id") != state["task_name"]:
                remapped["apk_id"] = state["task_name"]
            remapped.setdefault("transform_family", state["transform_family"])
            rows.append(remapped)
    for state in task_states:
        if state.get("status") != "ok":
            continue
        state["_region_label_rows"] = [
            {
                **row,
                "apk_id": state["task_name"],
                "transform_family": row.get(
                    "transform_family", state["transform_family"]
                ),
            }
            for row in state.get("_region_label_rows", [])
        ]

    # Honour flat scalar knobs mirrored from OursBaselineConfig.  We
    # deliberately avoid nesting the sub-configs here; ablation-level
    # knobs (typed encoder width, MIL pooling choice, etc.) will flow
    # through a dedicated ablation CLI.
    cfg_kwargs: Dict[str, Any] = {"supervision_mode": supervision_mode}
    base_cfg = OursBaselineConfig()
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "lambda_diff_pseudo",
        "lambda_sparsity",
        "bag_pos_weight",
        "random_state",
        "verbose",
        "train_max_bag_size",
        "train_min_positive_fraction",
        "threshold",
        "loader_cache_size",
        "scoring_mode",
    ):
        if key in config:
            caster = type(getattr(base_cfg, key))
            # Preserve Optional[int] for train_max_bag_size: allow null.
            if key == "train_max_bag_size" and config[key] is None:
                cfg_kwargs[key] = None
            else:
                cfg_kwargs[key] = caster(config[key])

    # L46 (2026-05-11): ablation-level sub-config knobs.
    # mil_pooling / n_types can override the default OursConfig to
    # enable flat-pool and no-type-embedding ablations via JSON config.
    ours_config_override: Dict[str, Any] = {}
    if "mil_pooling" in config:
        ours_config_override["mil_pooling"] = str(config["mil_pooling"])
    if "topk_k_ratio" in config:
        from android_packer.models.mil_head import TopKPoolingConfig
        ours_config_override["topk"] = TopKPoolingConfig(
            k_ratio=float(config["topk_k_ratio"])
        )
    typed_override: Dict[str, Any] = {}
    if "n_types" in config:
        typed_override["n_types"] = int(config["n_types"])
    if typed_override:
        from android_packer.models.typed_encoder import TypedEncoderConfig
        ours_config_override["typed"] = TypedEncoderConfig(**typed_override)
    if ours_config_override:
        from android_packer.models.ours import OursConfig as _OursConfig
        cfg_kwargs["ours_config"] = _OursConfig(**ours_config_override)

    # Handcrafted feature sub-config overrides (Tier 1A, improvement_plan_L47.md).
    # Use dataclasses.replace() rather than asdict()+reconstruct to avoid the
    # nested-dataclass-to-dict issue (EntropyDeltaConfig becomes a plain dict
    # when asdict() is called, breaking HandcraftedFeatureConfig(**...)).
    if "include_dex_structure" in config:
        from dataclasses import replace as _dc_replace
        cfg_kwargs["handcrafted_config"] = _dc_replace(
            base_cfg.handcrafted_config,
            include_dex_structure=bool(config["include_dex_structure"]),
        )

    inner_cfg = OursBaselineConfig(**cfg_kwargs)

    models_by_fold: Dict[str, Any] = {}
    primary_model: Any = None

    if train_mode == "same_set":
        try:
            primary_model = train_ours_baseline(rows, apk_index, inner_cfg)
        except ValueError as exc:
            warnings.append(f"ours skipped: {exc}")
            return {}
        except ImportError as exc:
            warnings.append(
                f"ours skipped: torch not installed ({exc}). "
                "Run ``pip install -e .[dl]`` to enable this baseline."
            )
            return {}
    else:
        group_key = (
            "transform_family" if train_mode == "holdout_transform" else "package_name"
        )
        if group_key == "package_name":
            for row in rows:
                if "package_name" not in row:
                    row["package_name"] = _package_name_from_task(
                        str(row.get("apk_id", "")), task_states
                    )

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)
        if len(groups) < 2:
            warnings.append(
                f"ours skipped: train_mode={train_mode!r} needs >= 2 "
                f"{group_key!r} values; got {sorted(groups)}."
            )
            return {}

        print(
            "aggregate ours object features "
            f"rows={len(rows)} groups={len(groups)}",
            flush=True,
        )
        loader = ObjectByteLoader(cache_size=inner_cfg.loader_cache_size)
        all_objects, _feat_names = _aggregate_object_features(
            rows,
            apk_index,
            handcrafted_config=inner_cfg.handcrafted_config,
            loader=loader,
        )
        object_groups: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        for key, obj in all_objects.items():
            obj_group_values = set()
            for row in obj.get("region_rows", []):
                obj_group_values.add(str(row[group_key]))
            if not obj_group_values:
                obj_group_values.add(str(obj.get(group_key, "")))
            compact_obj = {
                k: v for k, v in obj.items() if k != "region_rows"
            }
            for group_value in obj_group_values:
                object_groups.setdefault(group_value, {})[key] = compact_obj
        print(
            "aggregate ours object features done "
            f"objects={len(all_objects)}",
            flush=True,
        )

        held_values = sorted(groups)
        for fold_idx, held in enumerate(held_values, start=1):
            train_objects = {
                key: obj
                for group_value, items in object_groups.items()
                if group_value != held
                for key, obj in items.items()
            }
            print(
                "train ours fold="
                f"{fold_idx}/{len(held_values)} {group_key}={held!r} "
                f"train_objects={len(train_objects)} "
                f"heldout_rows={len(groups[held])}",
                flush=True,
            )
            try:
                fold_model = train_ours_baseline_from_objects(
                    train_objects, inner_cfg
                )
            except ValueError as exc:
                warnings.append(
                    f"ours fold {group_key}={held!r} skipped: {exc}"
                )
                continue
            except ImportError as exc:
                warnings.append(
                    f"ours skipped: torch not installed ({exc}). "
                    "Run ``pip install -e .[dl]`` to enable this baseline."
                )
                return {}
            print(
                "finished ours fold="
                f"{fold_idx}/{len(held_values)} {group_key}={held!r}",
                flush=True,
            )
            models_by_fold[held] = fold_model
            if primary_model is None:
                primary_model = fold_model

        if not models_by_fold:
            warnings.append("ours skipped: no folds produced a trained model.")
            return {}

    model_path = output_root / "models" / "ours.pt"
    if primary_model is not None:
        try:
            primary_model.save(model_path)
        except ImportError:
            pass

    return {
        "model": primary_model,
        "models_by_fold": models_by_fold,
        "model_path": str(model_path),
        "apk_index": apk_index,
        "train_mode": train_mode,
        "supervision_mode": supervision_mode,
        "train_row_count": len(rows),
        "group_key": (
            "transform_family"
            if train_mode == "holdout_transform"
            else ("package_name" if train_mode == "holdout_package" else None)
        ),
    }


# ---------------------------------------------------------------------------
# Manifest + summary
# ---------------------------------------------------------------------------


def _task_descriptor(
    *,
    seed_entry: Mapping[str, Any],
    transform_family: str,
    task_root: Path,
    generated_dir: Path,
    synthetic_manifest_dir: Path,
    synthetic_label_dir: Path,
    task_kind: str = "packed",
    evaluation_only: bool = False,
) -> Dict[str, Any]:
    stem = _task_stem(seed_entry, transform_family)
    task_dir = task_root / stem
    state = {
        "task_name": stem,
        "task_kind": task_kind,
        "evaluation_only": evaluation_only,
        "package_name": seed_entry.get("package_name"),
        "app_name": seed_entry.get("app_name"),
        "version_name": seed_entry.get("version_name"),
        "version_code": seed_entry.get("version_code"),
        "transform_family": transform_family,
        "seed_apk_path": seed_entry.get("local_path"),
        "generated_apk_path": _repo_path(generated_dir / f"{stem}.apk"),
        "synthetic_manifest_path": _repo_path(
            synthetic_manifest_dir / f"{stem}.manifest.json"
        ),
        "synthetic_labels_path": _repo_path(
            synthetic_label_dir / f"{stem}.labels.jsonl"
        ),
        "objects_path": _repo_path(task_dir / "objects.jsonl"),
        "regions_path": _repo_path(task_dir / "regions.jsonl"),
        "region_labels_path": _repo_path(task_dir / "region_labels.jsonl"),
        "object_labels_path": _repo_path(task_dir / "object_labels.jsonl"),
        "apk_labels_path": _repo_path(task_dir / "apk_labels.jsonl"),
    }
    # Per-baseline output paths. Kept here so the descriptor is the
    # single source of truth for where each artefact lives.
    for baseline_name in SUPPORTED_BASELINES:
        base_stem = f"{baseline_name}"
        state[f"{baseline_name}_region_predictions_path"] = _repo_path(
            task_dir / f"{base_stem}.region_predictions.jsonl"
        )
        state[f"{baseline_name}_object_predictions_path"] = _repo_path(
            task_dir / f"{base_stem}.object_predictions.jsonl"
        )
        state[f"{baseline_name}_apk_predictions_path"] = _repo_path(
            task_dir / f"{base_stem}.apk_predictions.jsonl"
        )
        state[f"{baseline_name}_report_path"] = _repo_path(
            task_dir / f"{base_stem}_report.json"
        )
    # APKiD uses its own APK-only predictions file name for clarity.
    state["apkid_apk_predictions_path"] = _repo_path(
        task_dir / "apkid.apk_predictions.jsonl"
    )
    return state


def _benign_task_descriptor(
    *,
    seed_entry: Mapping[str, Any],
    task_root: Path,
    generated_dir: Path,
    synthetic_manifest_dir: Path,
    synthetic_label_dir: Path,
) -> Dict[str, Any]:
    return _task_descriptor(
        seed_entry=seed_entry,
        transform_family="benign_control",
        task_root=task_root,
        generated_dir=generated_dir,
        synthetic_manifest_dir=synthetic_manifest_dir,
        synthetic_label_dir=synthetic_label_dir,
        task_kind="benign_control",
        evaluation_only=True,
    )


def _build_experiment_manifest(
    config: Mapping[str, Any],
    seed_manifest_path: Path,
    seed_entries: Sequence[Mapping],
    transforms: Sequence[str],
    baselines: Sequence[str],
    output_root: Path,
    generated_dir: Path,
    synthetic_manifest_dir: Path,
    synthetic_label_dir: Path,
    task_states: Sequence[Mapping],
    failures: int,
    ngram_state: Mapping[str, Any],
) -> Dict[str, Any]:
    # task_states may include non-serialisable entries (the cached
    # region_label_rows). Strip them before emitting the manifest.
    public_tasks = [_public_task_state(state) for state in task_states]
    return {
        "schema_version": 1,
        "experiment": "synthetic_multi_baseline",
        "seed_manifest": _repo_path(seed_manifest_path),
        "seed_count": len(seed_entries),
        "benign_control_count": sum(
            1 for state in public_tasks if state.get("task_kind") == "benign_control"
        ),
        "transform_families": list(transforms),
        "baselines": list(baselines),
        "config": {
            "regioning": config.get("regioning", {}),
            "labeling": config.get("labeling", {}),
            "synthetic": config.get("synthetic", {}),
            "baselines": config.get("baselines", {}) if isinstance(config.get("baselines"), dict) else [],
        },
        "paths": {
            "output_root": _repo_path(output_root),
            "synthetic_generated_dir": _repo_path(generated_dir),
            "synthetic_manifest_dir": _repo_path(synthetic_manifest_dir),
            "synthetic_label_dir": _repo_path(synthetic_label_dir),
        },
        "task_count": len(public_tasks),
        "success_count": sum(1 for t in public_tasks if t.get("status") == "ok"),
        "failure_count": failures,
        "tasks": public_tasks,
        "ngram_state": {
            k: v
            for k, v in ngram_state.items()
            # Exclude the fitted model (not JSON serialisable).
            if k != "model" and k != "apk_index"
        }
        if ngram_state
        else {},
    }


def _public_task_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in state.items() if not k.startswith("_")}


def _write_outputs(
    output_root: Path,
    experiment_manifest: Mapping[str, Any],
    reports_by_baseline: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict:
    _write_json(output_root / "experiment_manifest.json", dict(experiment_manifest))
    summary = _summarise(experiment_manifest, reports_by_baseline)
    _write_json(output_root / "summary.json", summary)
    return summary


def _summarise(
    experiment_manifest: Mapping[str, Any],
    reports_by_baseline: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict:
    summary: Dict[str, Any] = {
        "experiment": "synthetic_multi_baseline",
        "success_count": experiment_manifest["success_count"],
        "failure_count": experiment_manifest["failure_count"],
        "baselines": {},
        "warnings": experiment_manifest.get("warnings", []),
    }
    for baseline_name in experiment_manifest["baselines"]:
        reports = list(reports_by_baseline.get(baseline_name, []))
        by_transform: dict[str, list[Mapping[str, Any]]] = {}
        for report in reports:
            by_transform.setdefault(
                str(report.get("transform_family")), []
            ).append(report)
        overall = aggregate_reports(reports)
        pooled_apk = _pooled_apk_metrics(experiment_manifest, baseline_name)
        if pooled_apk is not None:
            overall["metrics"]["apk"] = pooled_apk
        summary["baselines"][baseline_name] = {
            "successful_task_count": len(reports),
            "overall": overall,
            "by_transform": {
                transform: aggregate_reports(items)
                for transform, items in sorted(by_transform.items())
            },
        }
    return summary


def _pooled_apk_metrics(
    experiment_manifest: Mapping[str, Any], baseline_name: str
) -> Optional[dict]:
    truth: List[int] = []
    predictions: List[int] = []
    scores: List[float] = []
    for task in experiment_manifest.get("tasks", []):
        bundle = task.get("baselines", {}).get(baseline_name, {})
        if bundle.get("status") != "ok":
            continue
        key = "apkid_apk_predictions_path" if baseline_name == "apkid" else f"{baseline_name}_apk_predictions_path"
        path_value = task.get(key)
        if not path_value:
            continue
        try:
            rows = list(read_jsonl(_resolve_state_path(path_value)))
        except OSError:
            continue
        for row in rows:
            if "true_label_id" not in row or "predicted_label_id" not in row or "score" not in row:
                continue
            truth.append(int(row["true_label_id"]))
            predictions.append(int(row["predicted_label_id"]))
            scores.append(float(row["score"]))
    if not truth:
        return None
    return binary_classification_metrics(
        truth=truth,
        predictions=predictions,
        scores=scores,
    ).to_dict()


# ---------------------------------------------------------------------------
# Skip-existing fingerprints
# ---------------------------------------------------------------------------


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """SHA256 of the JSON-canonical form of ``payload``."""

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_file_sha256(path: Path) -> Optional[str]:
    try:
        return file_sha256(Path(path))
    except (OSError, ValueError):
        return None


def _resolve_state_path(state_path: str) -> Path:
    """``state[*_path]`` may be relative (to repo root) or absolute.

    Returns an absolute :class:`Path` either way so callers can ``exists()``
    or read it without caring about which form was stored.
    """

    p = Path(state_path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _generate_fingerprint(
    seed_entry: Mapping[str, Any],
    transform_family: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inputs that, taken together, fully determine the generate-phase output."""

    seed_path = seed_entry.get("local_path")
    seed_sha = _safe_file_sha256(Path(seed_path)) if seed_path else None
    relevant_config = {
        "synthetic": config.get("synthetic", {}),
        "regioning": config.get("regioning", {}),
        "labeling": config.get("labeling", {}),
    }
    return {
        "version": GENERATE_FINGERPRINT_VERSION,
        "kind": "generate",
        "transform_family": transform_family,
        "seed_apk_sha256": seed_sha,
        "seed_apk_path": str(seed_path) if seed_path else None,
        "package_name": seed_entry.get("package_name"),
        "version_code": seed_entry.get("version_code"),
        "config_hash": _stable_hash(relevant_config),
    }


def _baseline_fingerprint(
    baseline_name: str,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inputs that determine a single (task, baseline) score output."""

    baseline_cfg = config.get("baselines", {}).get(baseline_name, {})
    generated_apk = _resolve_state_path(state["generated_apk_path"])
    region_labels = _resolve_state_path(state["region_labels_path"])
    return {
        "version": BASELINE_FINGERPRINT_VERSION,
        "kind": "baseline",
        "baseline": baseline_name,
        "task_name": state["task_name"],
        "generated_apk_sha256": _safe_file_sha256(generated_apk),
        "region_labels_sha256": _safe_file_sha256(region_labels),
        "baseline_config_hash": _stable_hash(dict(baseline_cfg)),
    }


def _generate_fingerprint_path(state: Mapping[str, Any]) -> Path:
    """`<task_dir>/.generate.fingerprint.json`. Recomputed from regions_path."""

    task_dir = _resolve_state_path(state["regions_path"]).parent
    return task_dir / ".generate.fingerprint.json"


def _baseline_fingerprint_path(state: Mapping[str, Any], baseline_name: str) -> Path:
    task_dir = _resolve_state_path(state["regions_path"]).parent
    return task_dir / f".{baseline_name}.fingerprint.json"


def _write_fingerprint(path: Path, fingerprint: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(fingerprint), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _load_fingerprint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fingerprints_match(
    expected: Mapping[str, Any], stored: Optional[Mapping[str, Any]]
) -> bool:
    if stored is None:
        return False
    for key, want in expected.items():
        if stored.get(key) != want:
            return False
    return True


def _generate_artifacts_present(state: Mapping[str, Any]) -> bool:
    keys = (
        "generated_apk_path",
        "synthetic_manifest_path",
        "synthetic_labels_path",
        "objects_path",
        "regions_path",
        "region_labels_path",
        "object_labels_path",
        "apk_labels_path",
    )
    return all(_resolve_state_path(state[k]).exists() for k in keys)


def _generate_artifacts_reusable(
    state: Mapping[str, Any], current_fp: Mapping[str, Any]
) -> bool:
    if not _generate_artifacts_present(state):
        return False
    stored = _load_fingerprint(_generate_fingerprint_path(state))
    return _fingerprints_match(current_fp, stored)


def _baseline_artifacts_present(
    state: Mapping[str, Any], baseline_name: str
) -> bool:
    if baseline_name == "apkid":
        keys = ("apkid_apk_predictions_path", "apkid_report_path")
    else:
        keys = (
            f"{baseline_name}_region_predictions_path",
            f"{baseline_name}_object_predictions_path",
            f"{baseline_name}_apk_predictions_path",
            f"{baseline_name}_report_path",
        )
    return all(_resolve_state_path(state[k]).exists() for k in keys)


def _baseline_artifacts_reusable(
    state: Mapping[str, Any],
    baseline_name: str,
    current_fp: Mapping[str, Any],
) -> bool:
    if not _baseline_artifacts_present(state, baseline_name):
        return False
    stored = _load_fingerprint(_baseline_fingerprint_path(state, baseline_name))
    return _fingerprints_match(current_fp, stored)


# ---------------------------------------------------------------------------
# Small helpers (shared shape with the entropy runner)
# ---------------------------------------------------------------------------


def _filter_dataclass_kwargs(cls, mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only mapping keys that correspond to dataclass field names."""

    try:
        field_names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    except AttributeError:
        return dict(mapping)
    return {k: v for k, v in mapping.items() if k in field_names}


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
    try:
        return file_sha256(Path(seed_path))[:8]
    except (OSError, ValueError):
        return None


def _validate_transforms(transforms: Sequence[str]) -> None:
    unsupported = [item for item in transforms if item not in SUPPORTED_TRANSFORMS]
    if unsupported:
        raise ValueError(f"unsupported transform families: {', '.join(unsupported)}")


def _validate_baselines(baselines: Sequence[str]) -> None:
    unsupported = [item for item in baselines if item not in SUPPORTED_BASELINES]
    if unsupported:
        raise ValueError(f"unsupported baselines: {', '.join(unsupported)}")


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
