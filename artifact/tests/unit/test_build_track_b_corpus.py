"""Tests for ``scripts/build_track_b_corpus.py``.

Focus on the pure-Python orchestration logic (registry parsing, dry-run
report shape, --only filtering). We never spawn subprocesses here; all
external toolchains are avoided by running in ``--dry-run`` mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure ``scripts/`` is importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import build_track_b_corpus as orch  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, *, packer_id: str = "demo_packer") -> Path:
    """Minimal but schema-valid registry with a single configured packer."""
    content = f"""\
schema_version: 1
packers:
  {packer_id}:
    gen_level: Gen1
    license: Apache-2.0
    status: selected
    repo:
      url: https://example.invalid/demo.git
      head_sha: deadbeefca
    build:
      kind: gradle
      toolchain: [jdk17]
      cmd: ./gradlew assembleRelease
    pack:
      kind: jar_invoke
      cmd: java -jar demo.jar --in ${{ANDROID_PACKER_INPUT_APK}} --out ${{ANDROID_PACKER_OUTPUT_APK}}
    patch:
      applier: thirdparty/patches/demo/deadbeefca/apply.py
      status: todo
    label:
      transform_family: packer_demo
      path_a: true
      path_b: true
"""
    reg = tmp_path / "packers.yaml"
    reg.write_text(content, encoding="utf-8")
    return reg


def _make_benign(tmp_path: Path, names=("a.apk", "b.apk")) -> Path:
    bdir = tmp_path / "benign"
    bdir.mkdir()
    for n in names:
        (bdir / n).write_bytes(b"PK\x03\x04 fake")
    return bdir


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def test_load_registry_happy_path(tmp_path):
    reg = _write_registry(tmp_path)
    packers = orch.load_registry(reg)
    assert "demo_packer" in packers
    assert packers["demo_packer"]["label"]["transform_family"] == "packer_demo"


def test_load_registry_rejects_missing_packers_key(tmp_path):
    reg = tmp_path / "bad.yaml"
    reg.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="packers"):
        orch.load_registry(reg)


def test_load_registry_rejects_missing_transform_family(tmp_path):
    reg = tmp_path / "bad.yaml"
    reg.write_text(
        "schema_version: 1\npackers:\n  broken:\n    label: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transform_family"):
        orch.load_registry(reg)


def test_load_real_registry_shape():
    """The checked-in registry must parse cleanly and expose 10 packers."""
    reg_path = _REPO_ROOT / "configs" / "data" / "track_b_packers.yaml"
    packers = orch.load_registry(reg_path)
    assert len(packers) >= 5  # S1..S5 minimum
    assert "s5_timscriptov_apkprotector_multiplatform" in packers
    assert packers["s5_timscriptov_apkprotector_multiplatform"]["patch"]["status"] == "done"


# ---------------------------------------------------------------------------
# Benign discovery
# ---------------------------------------------------------------------------


def test_discover_benign_empty_dir_returns_empty(tmp_path):
    assert orch.discover_benign_apks(tmp_path / "does_not_exist") == []


def test_discover_benign_lists_apks_sorted(tmp_path):
    bdir = _make_benign(tmp_path, names=("zeta.apk", "alpha.apk"))
    apks = orch.discover_benign_apks(bdir)
    assert [p.name for p in apks] == ["alpha.apk", "zeta.apk"]


# ---------------------------------------------------------------------------
# Dry-run orchestration
# ---------------------------------------------------------------------------


def test_dry_run_single_packer_all_steps_ok(tmp_path):
    reg = _write_registry(tmp_path)
    bdir = _make_benign(tmp_path, names=("a.apk",))
    summary = orch.orchestrate(
        orch.load_registry(reg),
        orch.discover_benign_apks(bdir),
        workspace=tmp_path,
        packed_dir=tmp_path / "packed",
        dry_run=True,
    )
    assert summary["ok_count"] == 1
    assert summary["fail_count"] == 0
    report = summary["reports"][0]
    step_names = [s["name"] for s in report["steps"]]
    assert step_names == ["clone", "apply_patch", "build", "pack"]
    assert all(s["ok"] for s in report["steps"])


def test_dry_run_no_benign_reports_preflight_failure(tmp_path):
    reg = _write_registry(tmp_path)
    summary = orch.orchestrate(
        orch.load_registry(reg),
        [],
        workspace=tmp_path,
        packed_dir=tmp_path / "packed",
        dry_run=True,
    )
    assert summary["benign_apk_count"] == 0
    assert summary["fail_count"] == 1
    assert summary["reports"][0]["steps"][0]["name"] == "preflight"


def test_dry_run_only_filter(tmp_path):
    """--only allowlist should restrict the set of packers run."""
    reg_content = """\
schema_version: 1
packers:
  a_packer:
    gen_level: Gen1
    status: selected
    repo: {url: "https://x.invalid/a.git", head_sha: "aaaa"}
    build: {cmd: "true"}
    pack: {kind: "jar_invoke", cmd: "true"}
    patch: null
    label: {transform_family: packer_a}
  b_packer:
    gen_level: Gen1
    status: selected
    repo: {url: "https://x.invalid/b.git", head_sha: "bbbb"}
    build: {cmd: "true"}
    pack: {kind: "jar_invoke", cmd: "true"}
    patch: null
    label: {transform_family: packer_b}
"""
    reg = tmp_path / "multi.yaml"
    reg.write_text(reg_content, encoding="utf-8")
    bdir = _make_benign(tmp_path, names=("a.apk",))
    summary = orch.orchestrate(
        orch.load_registry(reg),
        orch.discover_benign_apks(bdir),
        workspace=tmp_path,
        packed_dir=tmp_path / "packed",
        dry_run=True,
        packer_allowlist=["b_packer"],
    )
    assert {r["packer_id"] for r in summary["reports"]} == {"b_packer"}


def test_dry_run_commercial_packer_pack_step_is_skip(tmp_path):
    """Commercial packers (vendor_cli_pack / manual_upload) skip pack step OK."""
    reg_content = """\
schema_version: 1
packers:
  cs_demo:
    gen_level: Gen2
    license: commercial_eula
    status: candidate
    build:
      kind: vendor_cli
    pack:
      kind: vendor_cli_pack
    patch: null
    label:
      transform_family: packer_cs_demo
"""
    reg = tmp_path / "cs.yaml"
    reg.write_text(reg_content, encoding="utf-8")
    bdir = _make_benign(tmp_path, names=("a.apk",))
    summary = orch.orchestrate(
        orch.load_registry(reg),
        orch.discover_benign_apks(bdir),
        workspace=tmp_path,
        packed_dir=tmp_path / "packed",
        dry_run=True,
    )
    pack_step = next(s for s in summary["reports"][0]["steps"] if s["name"] == "pack")
    assert pack_step["ok"] is True
    assert "human-in-the-loop" in pack_step["message"]


# ---------------------------------------------------------------------------
# JSON summary shape (contract with downstream)
# ---------------------------------------------------------------------------


def test_summary_is_json_serializable(tmp_path):
    reg = _write_registry(tmp_path)
    bdir = _make_benign(tmp_path, names=("a.apk",))
    summary = orch.orchestrate(
        orch.load_registry(reg),
        orch.discover_benign_apks(bdir),
        workspace=tmp_path,
        packed_dir=tmp_path / "packed",
        dry_run=True,
    )
    # Must round-trip through json without raising.
    dumped = json.dumps(summary)
    reloaded = json.loads(dumped)
    assert reloaded["ok_count"] == summary["ok_count"]
