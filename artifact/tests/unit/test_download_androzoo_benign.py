from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "data" / "download_androzoo_benign.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_androzoo_benign_for_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(seed: str) -> str:
    return seed * 64


def _write_csv(path: Path, rows: list[dict]) -> None:
    header = [
        "sha256",
        "sha1",
        "md5",
        "dex_date",
        "apk_size",
        "pkg_name",
        "vercode",
        "vt_detection",
        "vt_scan_date",
        "dex_size",
        "markets",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in header) + "\n")


def test_select_candidates_prefers_modern_clean_play_rows(tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "latest.csv.gz"
    mb = 1024 * 1024
    _write_csv(
        csv_path,
        [
            {
                "sha256": _sha("a"),
                "dex_date": "2025-02-01",
                "apk_size": 5 * mb,
                "pkg_name": "com.example.side",
                "vt_detection": "0",
                "markets": "fdroid",
            },
            {
                "sha256": _sha("b"),
                "dex_date": "2024-01-01",
                "apk_size": 6 * mb,
                "pkg_name": "com.example.play",
                "vt_detection": "0",
                "markets": "play",
            },
            {
                "sha256": _sha("c"),
                "dex_date": "2026-01-01",
                "apk_size": 4 * mb,
                "pkg_name": "com.example.dirty",
                "vt_detection": "3",
                "markets": "play",
            },
            {
                "sha256": _sha("d"),
                "dex_date": "2019-12-31",
                "apk_size": 4 * mb,
                "pkg_name": "com.example.old",
                "vt_detection": "0",
                "markets": "play",
            },
        ],
    )

    selected = module.select_candidates(
        csv_path,
        target=2,
        min_dex_date="2020",
        max_size_mb=30,
        preferred_market="play",
        candidate_multiplier=1,
    )

    assert [row["pkg_name"] for row in selected] == [
        "com.example.play",
        "com.example.side",
    ]


def test_write_candidate_files_outputs_sha_and_jsonl(tmp_path: Path) -> None:
    module = _load_module()
    candidates = [
        {
            "sha256": _sha("e"),
            "pkg_name": "com.example.app",
            "apk_size": 123,
            "dex_date": "2025-01-01",
            "markets": "play",
            "vt_detection": 0,
        }
    ]
    sha_file = tmp_path / "candidates.txt"
    jsonl_file = tmp_path / "candidates.jsonl"

    module.write_candidate_files(candidates, sha_file, jsonl_file)

    assert sha_file.read_text(encoding="utf-8").strip() == _sha("e")
    assert '"pkg_name": "com.example.app"' in jsonl_file.read_text(encoding="utf-8")
