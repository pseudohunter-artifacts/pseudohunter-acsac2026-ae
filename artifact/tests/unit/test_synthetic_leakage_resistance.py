"""Regression tests for synthetic-corpus leakage fixes (A-v3, 2026-05-07).

Each test pins a *single* leakage vector identified by
``scripts/diag_synthetic_leakage_v2.py``. If a future refactor reintroduces
the leak, the corresponding test will fail and document why.
"""

from __future__ import annotations

import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.synthetic import build_synthetic_apk
from android_packer.synthetic.packer import build_seed_naming_profile


# Native seed APK with a realistic directory layout — emulates a small
# production APK closely enough that ``SeedNamingProfile`` produces a
# non-degenerate sample. We deliberately use a non-default date_time and
# external_attr so we can verify injected entries inherit them rather
# than the legacy ``(1980, ...)`` / ``0o644`` constants.
_NATIVE_DATE = (2024, 6, 15, 12, 30, 0)
_NATIVE_ATTR = 0o600 << 16


def _write_native_seed(path: Path) -> bytes:
    """Build a seed APK with a varied native directory / extension mix."""

    # 80-byte+ DEX so signature_strip can run too.
    payload = b"dex\n035\x00" + b"benign-dex-body-bytes-" * 8 + b"\x00" * 64

    native_entries = [
        ("classes.dex", payload, zipfile.ZIP_STORED),
        ("AndroidManifest.xml", b"<?xml version='1.0' ?><x/>", zipfile.ZIP_DEFLATED),
        ("res/layout/main.xml", b"<layout/>", zipfile.ZIP_DEFLATED),
        ("res/drawable/icon.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 256, zipfile.ZIP_DEFLATED),
        ("res/raw/data.bin", b"benign-binary-content" * 64, zipfile.ZIP_DEFLATED),
        ("assets/cache/state.dat", b"benign-state" * 32, zipfile.ZIP_DEFLATED),
        ("assets/data/config.json", b'{"key":"value"}', zipfile.ZIP_DEFLATED),
        ("kotlin/internal/module.kotlin_module", b"benign-kotlin", zipfile.ZIP_DEFLATED),
        ("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\r\n", zipfile.ZIP_DEFLATED),
    ]

    with zipfile.ZipFile(path, "w") as archive:
        for name, data, ctype in native_entries:
            info = zipfile.ZipInfo(filename=name, date_time=_NATIVE_DATE)
            info.compress_type = ctype
            info.external_attr = _NATIVE_ATTR
            archive.writestr(info, data)
    return payload


class SyntheticLeakageResistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.seed_apk = self.tmp / "seed.apk"
        _write_native_seed(self.seed_apk)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _generate(self, family: str, rng_seed: int = 17):
        result = build_synthetic_apk(
            seed_apk=self.seed_apk,
            generated_apk_out=self.tmp / f"gen_{family}_{rng_seed}.apk",
            transform_family=family,
            rng_seed=rng_seed,
            xor_key=0x2A,
            split_count=3,
            enforce_payload_size_range=False,
        )
        return result

    # -------------------------------------------------------------- L1
    def test_l1_injection_path_does_not_contain_family_name(self) -> None:
        """Path naming must not embed the transform family name (L1)."""
        for family in ("xor", "base64", "split_xor", "path_randomized",
                       "signature_strip", "embedded_archive"):
            with self.subTest(family=family):
                result = self._generate(family)
                for inj in result.manifest["injected_objects"]:
                    if inj.get("host_object_path"):
                        # sub-range transforms hijack a host entry, so the
                        # path is the host's — irrelevant for L1.
                        continue
                    self.assertNotIn(
                        family, inj["object_path"],
                        f"{family}: injection path leaks family name: "
                        f"{inj['object_path']}",
                    )

    # ------------------------------------------------------- L1 / L7 / L11
    def test_l11_injection_stem_not_pure_hex(self) -> None:
        """Stem must be alphanumeric mixed-case (not pure 12-hex token)."""
        result = self._generate("xor", rng_seed=99)
        whole_object_paths = [
            inj["object_path"]
            for inj in result.manifest["injected_objects"]
            if not inj.get("host_object_path")
        ]
        self.assertTrue(whole_object_paths, "expected at least one whole-object injection")
        for path in whole_object_paths:
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            # The legacy allocator produced 12 hex chars. After the fix
            # it samples ``random_stem_length`` from the seed (here 3..32
            # chars) and uses mixed-case alnum. Verify NOT pure 12-hex.
            self.assertFalse(
                bool(re.fullmatch(r"[0-9a-f]{12}", stem)),
                f"stem {stem!r} matches the legacy 12-hex pattern (L11 leak)",
            )

    # -------------------------------------------------------------- L4
    def test_l4_path_prefix_drawn_from_seed_directories(self) -> None:
        """Injection prefix should be one of the seed APK's directories,
        not the hard-coded ``assets/payload``."""
        seed_dirs = {
            "", "res/layout", "res/drawable", "res/raw",
            "assets/cache", "assets/data", "kotlin/internal",
        }
        result = self._generate("base64", rng_seed=42)
        whole = [
            inj["object_path"] for inj in result.manifest["injected_objects"]
            if not inj.get("host_object_path")
        ]
        for path in whole:
            prefix = path.rsplit("/", 1)[0] if "/" in path else ""
            self.assertIn(
                prefix, seed_dirs,
                f"injection prefix {prefix!r} is not in the seed APK's "
                f"directory pool — L4 leak.",
            )

    # -------------------------------------------------------------- L2-a
    def test_l2a_injected_entry_inherits_seed_date_time(self) -> None:
        """Injected ZipInfo.date_time must match the seed's mode value,
        not the legacy ``(1980, 1, 1, 0, 0, 0)`` constant."""
        result = self._generate("xor", rng_seed=11)
        with zipfile.ZipFile(result.generated_apk_path) as archive:
            injected_paths = {
                inj["object_path"]
                for inj in result.manifest["injected_objects"]
                if not inj.get("host_object_path")
            }
            for info in archive.infolist():
                if info.filename in injected_paths:
                    self.assertEqual(
                        info.date_time, _NATIVE_DATE,
                        f"{info.filename}: inherited date_time={info.date_time}, "
                        f"expected seed mode {_NATIVE_DATE} (L2-a leak)",
                    )

    # -------------------------------------------------------------- L14
    def test_l14_injected_entry_inherits_seed_external_attr(self) -> None:
        """Injected entries must inherit the seed's mode external_attr."""
        result = self._generate("path_randomized", rng_seed=23)
        with zipfile.ZipFile(result.generated_apk_path) as archive:
            injected_paths = {
                inj["object_path"]
                for inj in result.manifest["injected_objects"]
                if not inj.get("host_object_path")
            }
            for info in archive.infolist():
                if info.filename in injected_paths:
                    self.assertEqual(
                        info.external_attr, _NATIVE_ATTR,
                        f"{info.filename}: external_attr={info.external_attr} "
                        f"!= seed mode {_NATIVE_ATTR} (L14 leak)",
                    )

    # -------------------------------------------------------------- L3 / L15
    def test_l3_injected_entry_not_always_at_tail(self) -> None:
        """Injected entries must not deterministically land at the central
        directory tail; with shuffle they should disperse over many runs."""
        tail_hits = 0
        total = 0
        for seed in range(20):
            result = self._generate("xor", rng_seed=seed)
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                names = archive.namelist()
                injected = {
                    inj["object_path"]
                    for inj in result.manifest["injected_objects"]
                    if not inj.get("host_object_path")
                }
                positions = [i for i, n in enumerate(names) if n in injected]
                if not positions:
                    continue
                total += 1
                if max(positions) == len(names) - 1:
                    tail_hits += 1
        # Allow occasional tail landings (rng can pick the last slot ~1/N
        # of the time); reject the *always-at-tail* legacy behaviour
        # (which would be 100%). With 9 native entries, P(tail) ≈ 0.1.
        self.assertLess(
            tail_hits / max(1, total), 0.6,
            f"injected entry lands at the tail in {tail_hits}/{total} runs "
            f"— shuffle is not happening (L3 leak)",
        )

    # ------------------------------------------------------- profile self-check
    def test_seed_profile_excludes_meta_inf(self) -> None:
        profile = build_seed_naming_profile(self.seed_apk)
        for directory in profile.directory_pool:
            self.assertFalse(
                directory.startswith("META-INF"),
                f"naming_profile must not surface META-INF/ as an injection target",
            )

    # -------------------------------------------------------- determinism
    def test_determinism_preserved_under_shuffle(self) -> None:
        """The added shuffle must remain RNG-deterministic given the same
        ``rng_seed``."""
        first = self._generate("split_xor", rng_seed=7)
        second = self._generate("split_xor", rng_seed=7)
        with zipfile.ZipFile(first.generated_apk_path) as a, \
                zipfile.ZipFile(second.generated_apk_path) as b:
            self.assertEqual(a.namelist(), b.namelist())


if __name__ == "__main__":
    unittest.main()
