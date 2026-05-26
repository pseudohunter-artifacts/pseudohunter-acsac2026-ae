from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "data" / "build_packed_pair_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_packed_pair_manifest_for_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_packed_apk_hierarchical_layout(tmp_path: Path) -> None:
    module = _load_module()
    unpacked = tmp_path / "benign" / "app.apk"
    unpacked.parent.mkdir()
    unpacked.write_bytes(b"APK")
    packed = tmp_path / "packed" / "s6_dpt" / "app" / "packed.apk"
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b"PACKED")

    found = module.find_packed_apk(tmp_path / "packed", "s6_dpt", unpacked)

    assert found == packed


def test_find_packed_apk_flat_layout(tmp_path: Path) -> None:
    module = _load_module()
    unpacked = tmp_path / "benign" / "app.apk"
    unpacked.parent.mkdir()
    unpacked.write_bytes(b"APK")
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    packed = packed_dir / "s5_apkprotector__app.apk"
    packed.write_bytes(b"PACKED")

    found = module.find_packed_apk(packed_dir, "s5_apkprotector", unpacked)

    assert found == packed


def test_find_packed_apk_matches_sha_prefix_layout(tmp_path: Path) -> None:
    module = _load_module()
    sha = "0008C3A85769C8082AF4E009A7B37A411F956D4503E38EF7E2B23B10B2CA75A1"
    unpacked = tmp_path / "benign" / f"{sha}.apk"
    unpacked.parent.mkdir()
    unpacked.write_bytes(b"APK")
    packed = tmp_path / "packed" / "dpt_shell" / f"dpt__{sha[:24]}.apk"
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b"PACKED")

    found = module.find_packed_apk(tmp_path / "packed", "dpt_shell", unpacked)

    assert found == packed


def test_write_jsonl_serializes_records(tmp_path: Path) -> None:
    module = _load_module()
    row = module.PairRecord(
        pair_id="s6__app",
        packer_id="s6",
        source="fdroid",
        package_hint="app",
        unpacked_apk="app.apk",
        packed_apk="packed.apk",
        unpacked_sha256="0" * 64,
        packed_sha256="1" * 64,
        status="paired",
        apkid_unpacked_clean=True,
        apkid_packed_has_packer=True,
        apkid_unpacked=None,
        apkid_packed=None,
        notes=[],
    )
    out = tmp_path / "pairs.jsonl"

    module.write_jsonl(out, [row])

    text = out.read_text(encoding="utf-8")
    assert '"pair_id": "s6__app"' in text
    assert '"status": "paired"' in text
