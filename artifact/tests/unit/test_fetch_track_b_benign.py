"""Tests for ``scripts/fetch_track_b_benign.py``.

All tests avoid network: we stub ``urllib.request.urlopen`` where needed and
rely on local file fixtures otherwise.
"""

from __future__ import annotations

import hashlib
import io
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import fetch_track_b_benign as fetcher  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_manifest(tmp_path: Path, *, sha256: str | None = None) -> Path:
    sha_line = f'sha256: "{sha256}"' if sha256 else "sha256: null"
    content = textwrap.dedent(
        f"""\
        schema_version: 1
        benign_dir: {tmp_path.as_posix()}/benign
        apks:
          - package_name: com.example.demo
            apk_name: com.example.demo.apk
            download_url: https://example.invalid/com.example.demo.apk
            origin: supplemental_for_b
            {sha_line}
        """
    )
    p = tmp_path / "manifest.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def test_load_manifest_happy_path(tmp_path):
    manifest_path = _minimal_manifest(tmp_path, sha256="a" * 64)
    data = fetcher.load_manifest(manifest_path)
    assert data["apks"][0]["package_name"] == "com.example.demo"


def test_load_manifest_rejects_missing_apks_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="apks"):
        fetcher.load_manifest(p)


def test_load_manifest_rejects_entry_missing_required_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            schema_version: 1
            apks:
              - package_name: x
                apk_name: x.apk
                origin: supplemental_for_b
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="download_url"):
        fetcher.load_manifest(p)


def test_real_manifest_parses():
    """The checked-in Track B manifest must parse cleanly and list 10 APKs."""
    real = _REPO_ROOT / "configs" / "data" / "track_b_benign_manifest.yaml"
    data = fetcher.load_manifest(real)
    assert len(data["apks"]) == 10
    packages = [a["package_name"] for a in data["apks"]]
    # Sanity-check overlap with Track A (should contain org.fdroid.fdroid etc.)
    assert "org.fdroid.fdroid" in packages
    assert "com.fsck.k9" in packages
    origins = {a["origin"] for a in data["apks"]}
    assert origins == {"reused_from_track_a", "supplemental_for_b"}


def test_real_manifest_intersection_count():
    """Verify the 7/3 reused/supplemental split documented in the manifest."""
    real = _REPO_ROOT / "configs" / "data" / "track_b_benign_manifest.yaml"
    data = fetcher.load_manifest(real)
    reused = sum(1 for a in data["apks"] if a["origin"] == "reused_from_track_a")
    supplemental = sum(1 for a in data["apks"] if a["origin"] == "supplemental_for_b")
    assert reused == 7
    assert supplemental == 3
    assert data["counts"]["intersection_with_track_a"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Dry-run outcomes (no network)
# ---------------------------------------------------------------------------


def test_dry_run_plans_download(tmp_path):
    manifest_path = _minimal_manifest(tmp_path, sha256="a" * 64)
    rc = fetcher.main([
        "--manifest",
        str(manifest_path),
        # --dry-run is default
    ])
    assert rc == 0


def test_only_filter_marks_other_entries(tmp_path, capsys):
    """--only=foo should mark non-matching entries as filtered_out."""
    content = textwrap.dedent(
        f"""\
        schema_version: 1
        benign_dir: {tmp_path.as_posix()}/benign
        apks:
          - package_name: com.wanted
            apk_name: com.wanted.apk
            download_url: https://example.invalid/w.apk
            origin: supplemental_for_b
            sha256: null
          - package_name: com.ignored
            apk_name: com.ignored.apk
            download_url: https://example.invalid/i.apk
            origin: supplemental_for_b
            sha256: null
        """
    )
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(content, encoding="utf-8")
    summary_path = tmp_path / "out.json"
    rc = fetcher.main([
        "--manifest", str(manifest_path),
        "--only", "com.wanted",
        "--json-summary", str(summary_path),
    ])
    assert rc == 0
    import json as _json
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    statuses = {o["package_name"]: o["status"] for o in summary["outcomes"]}
    assert statuses["com.wanted"] == "planned"
    assert statuses["com.ignored"] == "filtered_out"


# ---------------------------------------------------------------------------
# sha256 comparison (no network, stub downloads)
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    payload = b"hello world" * 100
    p.write_bytes(payload)
    assert fetcher.sha256_of(p) == hashlib.sha256(payload).hexdigest()


def test_existing_file_matching_sha256_is_skipped(tmp_path):
    payload = b"packaged apk bytes"
    expected = hashlib.sha256(payload).hexdigest()
    manifest_path = _minimal_manifest(tmp_path, sha256=expected)
    # Pre-place the "downloaded" APK so the fetcher skips it.
    benign_dir = tmp_path / "benign"
    benign_dir.mkdir()
    (benign_dir / "com.example.demo.apk").write_bytes(payload)

    summary_path = tmp_path / "out.json"
    rc = fetcher.main([
        "--manifest", str(manifest_path),
        "--execute",
        "--json-summary", str(summary_path),
    ])
    assert rc == 0
    import json as _json
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["outcomes"][0]["status"] == "skipped_already_ok"


def test_existing_file_mismatched_sha256_reports_failure(tmp_path):
    expected = "b" * 64  # not the real sha of the file contents
    manifest_path = _minimal_manifest(tmp_path, sha256=expected)
    benign_dir = tmp_path / "benign"
    benign_dir.mkdir()
    (benign_dir / "com.example.demo.apk").write_bytes(b"wrong bytes")

    summary_path = tmp_path / "out.json"
    rc = fetcher.main([
        "--manifest", str(manifest_path),
        "--execute",
        "--json-summary", str(summary_path),
    ])
    assert rc == 1  # failed
    import json as _json
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["outcomes"][0]["status"] == "sha256_mismatch"


# ---------------------------------------------------------------------------
# Stubbed download exercising the happy path + lock-sha256
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._buf.close()
        return False


def test_execute_downloads_then_locks_sha256(tmp_path, monkeypatch):
    manifest_path = _minimal_manifest(tmp_path, sha256=None)
    payload = b"fake apk bytes: track-b smoke test"
    expected = hashlib.sha256(payload).hexdigest()

    def _fake_urlopen(request, timeout=180):  # noqa: ARG001
        return _FakeResponse(payload)

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", _fake_urlopen)

    rc = fetcher.main([
        "--manifest", str(manifest_path),
        "--execute",
        "--lock-sha256",
    ])
    assert rc == 0

    # APK was written + sha256 locked back into manifest.
    written = tmp_path / "benign" / "com.example.demo.apk"
    assert written.exists()
    assert fetcher.sha256_of(written) == expected

    locked_text = manifest_path.read_text(encoding="utf-8")
    assert f"sha256: {expected}" in locked_text
    assert "sha256: null" not in locked_text
