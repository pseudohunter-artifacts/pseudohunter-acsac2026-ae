"""Track B corpus pipeline: F-Droid origin -> open-source packer -> diff labels.

This is the one-shot orchestrator that:

  1. Reads the PackerGrind app inventory
     (``data/real_world/packergrind/origins_manifest.json`` from
     ``fetch_fdroid_origins.py``) to know which F-Droid APKs are on disk.

  2. For each (origin APK, packer) combination, invokes
     ``scripts/build_track_b_corpus.py`` (single-packer / single-benign mode)
     to produce a packed APK.

  3. For each produced (origin, packed) pair, invokes the diff-based
     labeler (``scripts/labeling/build_training_labels.py`` which wraps
     ``src/android_packer/labeling/diff_alignment.py``) to emit
     region-level ground-truth labels.

  4. Writes a single ``corpus_manifest.json`` that downstream training /
     evaluation code can consume as a "Track B v2" corpus.

This script is STATE-FREE w.r.t. subprocesses: every external command is
re-discoverable from the emitted plan. In ``--dry-run`` (default) it only
prints the plan; in ``--execute`` it runs the commands sequentially,
continuing past individual failures so one missing packer doesn't abort
the batch.

Usage::

    # Plan only
    python scripts/data/build_track_b_from_fdroid.py

    # Plan with only open-source packers (skips CS1-CS5 which need humans)
    python scripts/data/build_track_b_from_fdroid.py --packers S3,S5,S6

    # Execute for one app x one packer smoke test
    python scripts/data/build_track_b_from_fdroid.py \
        --packers S3 --only-apps 2048 --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ORIGINS_MANIFEST = REPO_ROOT / "data" / "real_world" / "packergrind" / "origins_manifest.json"
DEFAULT_PACKERS_YAML = REPO_ROOT / "configs" / "data" / "track_b_packers.yaml"
DEFAULT_CORPUS_OUT = REPO_ROOT / "data" / "real_world" / "track_b_v2"
DEFAULT_PLAN_OUT = REPO_ROOT / "outputs" / "experiments" / "track_b_v2" / "build_plan.json"
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "data" / "real_world" / "track_b_v2" / "corpus_manifest.json"

# Registered packer ids (memoized from the project's naming convention).
# Open-source packers produce bytes deterministically -> good for an automated
# pipeline. Commercial packers require manual vendor interaction -> surfaced
# in the plan as ``manual_required`` entries.
OPEN_SOURCE_PACKERS = {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
COMMERCIAL_PACKERS = {"CS1", "CS2", "CS3", "CS4", "CS5"}

# Short code (what our memory-of-naming + the CLI accepts) -> real packer_id
# registered under configs/data/track_b_packers.yaml. The right-hand side is
# what ``scripts/build_track_b_corpus.py --only <id>`` expects.
SHORT_TO_PACKER_ID = {
    "S1": "s1_cvvt_apkprotect",
    "S2": "s2_ijiami_apkprotect",
    "S3": "s3_oncealong_apk_dex_shell",
    "S4": "s4_huyehan_third_generation_shell",
    "S5": "s5_timscriptov_apkprotector_multiplatform",
    "S6": "s6_dpt_shell_luoyesiqiu",
    "S7": "s7_bangcle_oss",
    "CS1": "cs1_360_jiagu",
    "CS2": "cs2_ijiami_com",
    "CS3": "cs3_bangcle",
    "CS4": "cs4_legu",
    "CS5": "cs5_dexprotector",
}
PACKER_ID_TO_SHORT = {v: k for k, v in SHORT_TO_PACKER_ID.items()}


@dataclass
class PipelineTask:
    app: str                     # PackerGrind canonical app name
    package_name: str
    origin_apk: str              # path to the origin APK on disk
    packer_id: str               # e.g. "S3 . oncealong"
    source_repo: Optional[str] = None   # "main" | "archive" (from origins_manifest)
    packed_apk: Optional[str] = None
    labels_jsonl: Optional[str] = None

    # Step status: planned | skipped | packed_ok | pack_failed | label_ok | label_failed | manual_required
    status: str = "planned"
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_origins(manifest_path: Path, *, include_planned: bool = False) -> List[Dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    # In real pipelines we only trust on-disk APKs (downloaded / already-ok).
    # In dry-run mode the reviewer may want to walk the plan before ever
    # calling fetch_fdroid_origins.py --execute, so ``planned`` is permitted
    # when ``include_planned`` is set.
    accept = {"skipped_already_ok", "downloaded"}
    if include_planned:
        accept |= {"planned"}
    return [r for r in results if r.get("status") in accept and r.get("local_path")]


def _load_packer_ids(packers_yaml: Path, only: Optional[List[str]]) -> List[str]:
    """Load packer IDs from configs/data/track_b_packers.yaml.

    Falls back to the static OPEN_SOURCE_PACKERS + COMMERCIAL_PACKERS sets
    when the YAML isn't available (e.g. running the script on a leaner tree).
    """
    chosen: List[str] = []
    if packers_yaml.exists():
        try:
            import yaml  # noqa: WPS433

            data = yaml.safe_load(packers_yaml.read_text(encoding="utf-8"))
            packers = data.get("packers") if isinstance(data, dict) else None
            if isinstance(packers, list):
                chosen = [
                    str(p.get("id") or p.get("packer_id") or "")
                    for p in packers
                    if isinstance(p, dict)
                ]
                chosen = [c for c in chosen if c]
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] failed to parse {packers_yaml}: {exc}", file=sys.stderr)
    if not chosen:
        # Conservative default: the four open-source packers we have patches for.
        chosen = ["S1", "S2", "S3", "S4", "S5", "S6"]

    if only:
        only_set = {x.strip() for x in only if x and x.strip()}
        chosen = [c for c in chosen if c in only_set or c.split(" ")[0] in only_set]
    # Preserve order but dedup.
    seen = set()
    result = []
    for c in chosen:
        if c not in seen:
            result.append(c)
            seen.add(c)
    return result


def _plan_tasks(
    origins: List[Dict[str, Any]],
    packer_ids: List[str],
    corpus_dir: Path,
    only_apps: Optional[List[str]],
) -> List[PipelineTask]:
    tasks: List[PipelineTask] = []
    app_whitelist = {a for a in (only_apps or []) if a}
    for origin in origins:
        app = origin.get("app")
        if app_whitelist and app not in app_whitelist:
            continue
        apk_path = origin.get("local_path")
        if not apk_path:
            continue
        package_name = origin.get("package_name") or "?"
        src_repo = (origin.get("extra") or {}).get("source_repo")

        for pid in packer_ids:
            pid_key = pid.split(" ")[0]  # "S3 . oncealong" -> "S3"
            packed_name = f"{package_name}.{pid_key}.packed.apk"
            labels_name = f"{package_name}.{pid_key}.labels.jsonl"
            task = PipelineTask(
                app=app,
                package_name=package_name,
                origin_apk=apk_path,
                packer_id=pid,
                source_repo=src_repo,
                packed_apk=str(corpus_dir / "packed" / packed_name),
                labels_jsonl=str(corpus_dir / "labels" / labels_name),
            )
            if pid_key in COMMERCIAL_PACKERS:
                task.status = "manual_required"
                task.message = (
                    f"{pid} is a commercial packer; upload {apk_path} to the "
                    f"vendor console and drop the output at {task.packed_apk}"
                )
            tasks.append(task)
    return tasks


def _run_pack_batch(
    tasks: List[PipelineTask],
    *,
    corpus_dir: Path,
    origins_dir: Path,
    dry_run: bool,
) -> None:
    """Batch-delegate packing to ``scripts/build_track_b_corpus.py``.

    ``build_track_b_corpus.py`` is registry-driven: it scans every APK in
    ``--benign-dir`` and packs each with every packer in ``--only``. So we
    invoke it once per packer-set (not once per task), which matches the
    upstream tool's intended usage and lets it stage clones / toolchains once.

    Task status is reconciled post-hoc from the upstream plan's ``reports``
    array (each report carries ``packed_apk`` + ``ok`` + per-step details).
    Commercial packers are skipped (their upstream report marks them
    ``manual_upload`` and yields no bytes).
    """
    # Partition: commercial tasks -> skipped immediately.
    batch_tasks = [t for t in tasks if t.packer_id.split(" ")[0] not in COMMERCIAL_PACKERS]
    for t in tasks:
        if t.packer_id.split(" ")[0] in COMMERCIAL_PACKERS:
            t.status = "manual_required"
            t.message = (
                f"{t.packer_id} is a commercial packer; upload {t.origin_apk} "
                "to the vendor console and drop the output into packed_dir manually"
            )
    if not batch_tasks:
        return

    # Collect the unique set of upstream packer_ids to restrict build_track_b_corpus.py to.
    upstream_ids = sorted({
        SHORT_TO_PACKER_ID.get(t.packer_id.split(" ")[0], t.packer_id.split(" ")[0])
        for t in batch_tasks
    })

    upstream_packed_dir = corpus_dir / "packed"
    upstream_plan = corpus_dir / "build_track_b_corpus.plan.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_track_b_corpus.py"),
        "--benign-dir", str(origins_dir),
        "--packed-dir", str(upstream_packed_dir),
        "--out-summary", str(upstream_plan),
        "--only", *upstream_ids,
    ]
    if not dry_run:
        cmd.append("--execute")

    if dry_run:
        for t in batch_tasks:
            t.extra["pack_cmd"] = cmd
            t.status = "planned"
            t.message = f"dry-run; would batch-delegate to {upstream_ids}"
        return

    upstream_packed_dir.mkdir(parents=True, exist_ok=True)
    upstream_plan.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        # Upstream plan still gets written on partial failure, so continue to
        # reconciliation; mark every batch task as failed as a floor, the
        # reports loop below will upgrade any that actually succeeded.
        for t in batch_tasks:
            t.status = "pack_failed"
            t.message = f"build_track_b_corpus.py exited {exc.returncode}"
    except FileNotFoundError as exc:
        for t in batch_tasks:
            t.status = "pack_failed"
            t.message = f"missing tool: {exc}"
        return

    # Reconcile: read upstream plan and map (packer_id, benign_apk) -> report.
    if not upstream_plan.exists():
        return
    try:
        upstream = json.loads(upstream_plan.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        for t in batch_tasks:
            if t.status in {"planned", "pack_failed"}:
                t.message = f"{t.message} (also: failed to parse upstream plan: {exc})"
        return
    report_by_key: Dict[tuple, Dict[str, Any]] = {}
    for rep in upstream.get("reports", []):
        key = (rep.get("packer_id"), Path(str(rep.get("benign_apk", ""))).name)
        report_by_key[key] = rep

    for t in batch_tasks:
        short = t.packer_id.split(" ")[0]
        upstream_id = SHORT_TO_PACKER_ID.get(short, short)
        benign_name = Path(t.origin_apk).name
        rep = report_by_key.get((upstream_id, benign_name))
        if rep is None:
            # Upstream didn't see this benign-apk (e.g. benign-dir was pruned).
            if t.status == "pack_failed":
                continue
            t.status = "pack_failed"
            t.message = "no upstream report for this (packer_id, benign_apk)"
            continue
        upstream_packed = rep.get("packed_apk")
        if upstream_packed:
            t.packed_apk = upstream_packed  # pin to the real produced path
        t.extra["upstream_packer_id"] = upstream_id
        t.extra["upstream_inject_labels"] = rep.get("inject_labels_jsonl")
        step_summary = [
            {"name": s.get("name"), "ok": bool(s.get("ok"))}
            for s in rep.get("steps", [])
        ]
        t.extra["upstream_steps"] = step_summary
        if rep.get("ok") and upstream_packed and Path(upstream_packed).exists():
            t.status = "packed_ok"
            t.message = "packed via build_track_b_corpus.py"
        else:
            t.status = "pack_failed"
            bad_step = next((s["name"] for s in step_summary if not s["ok"]), "?")
            t.message = f"upstream step failed: {bad_step}"


def _run_pack(task: PipelineTask, *, dry_run: bool) -> None:
    """Legacy single-task pack entry point (kept for completeness).

    New callers should prefer :func:`_run_pack_batch`, which batches every
    task in a single ``build_track_b_corpus.py`` invocation. This single-shot
    form is retained only for tooling that wants to pack one (origin, packer)
    pair without the orchestrator's batching semantics.
    """
    pid_key = task.packer_id.split(" ")[0]
    if pid_key in COMMERCIAL_PACKERS:
        task.status = "manual_required"
        return
    upstream_id = SHORT_TO_PACKER_ID.get(pid_key, pid_key)
    origin_parent = Path(task.origin_apk).parent
    upstream_plan = Path(task.packed_apk).parent.parent / "build_track_b_corpus.plan.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_track_b_corpus.py"),
        "--benign-dir", str(origin_parent),
        "--packed-dir", str(Path(task.packed_apk).parent),
        "--out-summary", str(upstream_plan),
        "--only", upstream_id,
    ]
    if not dry_run:
        cmd.append("--execute")
    if dry_run:
        task.extra["pack_cmd"] = cmd
        task.status = "planned"
        task.message = "dry-run; pack command recorded"
        return
    Path(task.packed_apk).parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        task.status = "pack_failed"
        task.message = f"build_track_b_corpus.py exited {exc.returncode}"
        return
    except FileNotFoundError as exc:
        task.status = "pack_failed"
        task.message = f"missing tool: {exc}"
        return
    # Single-mode does not rewrite task.packed_apk from the report; the caller
    # is responsible for collecting from upstream_packed_dir.
    task.status = "packed_ok"
    task.message = "packed via build_track_b_corpus.py (single)"


def _run_label(task: PipelineTask, *, dry_run: bool) -> None:
    """Invoke the diff-based labeler on (origin, packed) to emit labels."""
    if task.status not in {"packed_ok", "planned"}:  # planned => dry-run
        return
    # The labeler is exposed through scripts/labeling/build_training_labels.py,
    # which in turn imports src/android_packer/labeling/diff_alignment.py.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "labeling" / "build_training_labels.py"),
        "--origin", task.origin_apk,
        "--packed", task.packed_apk,
        "--out", task.labels_jsonl,
    ]
    if dry_run:
        task.extra["label_cmd"] = cmd
        return

    Path(task.labels_jsonl).parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        task.status = "label_failed"
        task.message = f"labeler exited {exc.returncode}"
        return
    except FileNotFoundError as exc:
        task.status = "label_failed"
        task.message = f"missing tool: {exc}"
        return
    if not Path(task.labels_jsonl).exists():
        task.status = "label_failed"
        task.message = f"labeler did not produce {task.labels_jsonl}"
        return
    task.status = "label_ok"
    task.message = "packed+labeled"


def _write_plan_json(path: Path, tasks: List[PipelineTask], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "mode": mode,
        "tasks": [asdict(t) for t in tasks],
        "counts": _counts(tasks),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _counts(tasks: List[PipelineTask]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in tasks:
        out[t.status] = out.get(t.status, 0) + 1
    out["total"] = len(tasks)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins-manifest", type=Path, default=DEFAULT_ORIGINS_MANIFEST)
    parser.add_argument("--packers-yaml", type=Path, default=DEFAULT_PACKERS_YAML)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_OUT)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument(
        "--packers",
        type=str,
        default=None,
        help="Comma-separated packer IDs to include, e.g. 'S3,S5' or 'S3 . oncealong'.",
    )
    parser.add_argument(
        "--only-apps",
        type=str,
        default=None,
        help="Comma-separated PackerGrind app names to process (subset).",
    )
    parser.add_argument(
        "--skip-commercial",
        action="store_true",
        default=True,
        help="Skip commercial packers (default: true; they require manual uploads).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    dry_run = not args.execute

    if not args.origins_manifest.exists():
        print(f"[ABORT] origins manifest not found: {args.origins_manifest}", file=sys.stderr)
        print(
            "        run scripts/data/fetch_fdroid_origins.py --execute first.",
            file=sys.stderr,
        )
        return 2

    origins = _load_origins(args.origins_manifest, include_planned=dry_run)
    if not origins:
        if dry_run:
            print(
                f"[ABORT] origins manifest {args.origins_manifest} has no "
                f"usable entries (no 'planned' / 'downloaded' / 'skipped_already_ok').",
                file=sys.stderr,
            )
        else:
            print(
                f"[ABORT] origins manifest {args.origins_manifest} has no "
                f"on-disk APKs; run fetch_fdroid_origins.py --execute first.",
                file=sys.stderr,
            )
        return 2

    packer_list = None
    if args.packers:
        packer_list = [x.strip() for x in args.packers.split(",") if x.strip()]
    packer_ids = _load_packer_ids(args.packers_yaml, packer_list)
    if args.skip_commercial:
        packer_ids = [p for p in packer_ids if p.split(" ")[0] not in COMMERCIAL_PACKERS]
    if not packer_ids:
        print(f"[ABORT] no packers selected after filters", file=sys.stderr)
        return 2

    only_apps = None
    if args.only_apps:
        only_apps = [x.strip() for x in args.only_apps.split(",") if x.strip()]

    corpus_dir = args.corpus_dir.resolve()
    tasks = _plan_tasks(origins, packer_ids, corpus_dir, only_apps)
    if not tasks:
        print("[ABORT] no (origin, packer) tasks to run after --only-apps filter", file=sys.stderr)
        return 2

    # Derive the origins directory from the manifest -- every origin we actually
    # have on disk must share the same parent (that's how fetch_fdroid_origins.py
    # lays them out). We feed this directory straight to build_track_b_corpus.py.
    origins_parents = {Path(o["local_path"]).parent for o in origins if o.get("local_path")}
    if len(origins_parents) != 1:
        print(
            f"[WARN] origins span multiple parents ({origins_parents}); "
            f"falling back to the first one for batch packing",
            file=sys.stderr,
        )
    origins_dir = next(iter(origins_parents))

    # Step 1: batch-pack via build_track_b_corpus.py (one call for all tasks).
    _run_pack_batch(tasks, corpus_dir=corpus_dir, origins_dir=origins_dir, dry_run=dry_run)
    # Step 2: label (only for tasks that actually produced packed bytes, or in dry-run we record the cmd)
    for t in tasks:
        if dry_run or t.status == "packed_ok":
            _run_label(t, dry_run=dry_run)

    # Write plan + corpus manifest.
    _write_plan_json(args.plan_out, tasks, "dry-run" if dry_run else "execute")

    corpus_manifest = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "mode": "dry-run" if dry_run else "execute",
        "corpus_dir": str(corpus_dir),
        "counts": _counts(tasks),
        "tasks": [asdict(t) for t in tasks],
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(corpus_manifest, indent=2), encoding="utf-8"
    )

    mode_tag = "DRY-RUN" if dry_run else "EXECUTE"
    c = _counts(tasks)
    print(f"[{mode_tag}] plan={args.plan_out}")
    print(f"          manifest={args.manifest_out}")
    print("  counts:", json.dumps(c))
    # Compact per-task preview (truncated).
    preview_n = min(20, len(tasks))
    for t in tasks[:preview_n]:
        pid_key = t.packer_id.split(" ")[0]
        print(
            f"  [{t.status:20}] {t.app:22} x {pid_key:5s} "
            f"(origin={Path(t.origin_apk).name}) {t.message}"
        )
    if len(tasks) > preview_n:
        print(f"  ... ({len(tasks) - preview_n} more tasks in the plan JSON)")

    hard_failures = c.get("pack_failed", 0) + c.get("label_failed", 0)
    return 0 if hard_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
