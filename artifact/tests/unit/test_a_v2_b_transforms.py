"""Unit tests for the three A-v2-b transforms (2026-04-30).

Covers:

* ``multi_dex_shim`` — shim classes.dex overwrite + payload at
  ``assets/<random>``.
* ``embedded_archive`` — nested jar + XOR encryption whole.
* ``dex_string_encrypted`` — in-place encryption of string_data
  spans inside classes.dex.

Each transform has at least (a) a byte-sanity test verifying the
produced records have the structural properties claimed in
``docs/method/threat_model.md`` §"Track A v2", and (b) a label-range
test verifying ``payload_offset_*`` match expectations.

Tests exercise the full ``build_synthetic_apk`` pipeline (E2E) for
parity with ``test_synthetic_transforms.py``; the actual transform
logic is reached indirectly but every label / manifest side-effect
we care about is checked on the real build result.
"""

from __future__ import annotations

import io
import random
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# ``_dex_fixtures`` lives next to this file; extend sys.path the same
# way ``test_synthetic_transforms.py`` does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dex_fixtures import build_minimal_dex  # noqa: E402

from android_packer.synthetic import (  # noqa: E402
    SUPPORTED_TRANSFORMS,
    TRANSFORMS,
    build_synthetic_apk,
)
from android_packer.synthetic.records import SyntheticPackerError  # noqa: E402
from android_packer.synthetic.transforms import xor_bytes  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_seed_apk_with_host(
    path: Path, *, include_classes_dex: bool = True
) -> bytes:
    """Write a seed APK with a minimal parseable classes.dex + asset + so.

    Returns the bytes of the host classes.dex so tests can assert on
    pre-transform content. The seed APK is populated with the same
    three host objects used in ``test_synthetic_transforms.py``
    (DEX + PNG-like asset + ELF .so) so all three new transforms have
    valid hosts available.
    """

    host_dex_bytes, _layout = build_minimal_dex(num_code_items=4000)

    rng = random.Random(0xBEEF)
    asset_bytes = (
        b"license header\n"
        + b"".join(
            bytes(rng.choices(b"ABCDEF0123456789 \n", k=512))
            for _ in range(256)
        )
    )
    assert len(asset_bytes) >= 128 * 1024

    elf_header = (
        b"\x7fELF\x02\x01\x01\x00"
        + b"\x00" * 8
        + b"\x03\x00"
        + b"\xb7\x00"
        + b"\x01\x00\x00\x00"
    )
    so_bytes = elf_header + b"\x00" * (128 * 1024 - len(elf_header))

    with zipfile.ZipFile(path, "w") as archive:
        if include_classes_dex:
            archive.writestr("classes.dex", host_dex_bytes)
        archive.writestr("assets/license.txt", asset_bytes)
        archive.writestr("lib/arm64-v8a/libdemo.so", so_bytes)

    return host_dex_bytes


def _make_payload_file(tmpdir: Path, size: int = 96 * 1024) -> Path:
    """Build a deterministic external payload file.

    Keep >= 64 KiB so the B1 size floor accepts it. Byte pattern is
    deterministic so tests can recover & compare after decryption.
    """

    payload_bytes = bytes((i * 31 + 7) & 0xFF for i in range(size))
    payload_path = tmpdir / "external_payload.bin"
    payload_path.write_bytes(payload_bytes)
    return payload_path


# ---------------------------------------------------------------------------
# multi_dex_shim
# ---------------------------------------------------------------------------


class MultiDexShimTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("multi_dex_shim", TRANSFORMS)
        self.assertIn("multi_dex_shim", SUPPORTED_TRANSFORMS)

    def test_emits_two_records_shim_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="multi_dex_shim",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            injected = result.manifest["injected_objects"]

            self.assertEqual(len(injected), 2)
            shim, asset = injected[0], injected[1]

            # (1) Shim overwrites classes.dex in-place, benign-loader
            # semantic (empty positive range).
            self.assertEqual(shim["object_path"], "classes.dex")
            self.assertEqual(shim["host_object_path"], "classes.dex")
            self.assertEqual(shim["offset_start"], 0)
            self.assertEqual(shim["offset_end"], 0)

            # (2) Asset-level payload injection, whole-object.
            self.assertIsNone(asset.get("host_object_path"))
            # A-v3 leakage fix (2026-05-07): the injection path is now
            # sampled from the seed APK's directory pool (or falls back
            # to ``assets/synthetic`` / ``assets/payload`` when there's
            # no naming profile, e.g. the legacy unit-test seed). The
            # only invariant we still enforce is that the path is
            # distinct from the shim's classes.dex and is non-empty.
            self.assertNotEqual(asset["object_path"], "classes.dex")
            self.assertTrue(asset["object_path"])
            # Family name MUST NOT appear in the path (regression guard
            # for L1 leak — see ``test_synthetic_leakage_resistance.py``).
            self.assertNotIn("multi_dex_shim", asset["object_path"])

            # Verify on-disk APK: classes.dex starts with DEX magic,
            # proving the shim overwrite happened.
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                shim_on_disk = archive.read("classes.dex")
                self.assertTrue(shim_on_disk.startswith(b"dex\n035\x00"))
                # And the shim is >= 64 KiB (pads downstream size floors).
                self.assertGreaterEqual(len(shim_on_disk), 64 * 1024)
                # Payload asset exists.
                self.assertIn(asset["object_path"], set(archive.namelist()))

    def test_payload_recoverable_with_asset_xor_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)
            original_payload = payload_path.read_bytes()

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="multi_dex_shim",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            _shim, asset = result.manifest["injected_objects"]
            self.assertIsNotNone(asset["xor_key"])
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                encrypted = archive.read(asset["object_path"])
            recovered = xor_bytes(encrypted, asset["xor_key"])
            self.assertEqual(recovered, original_payload)

    def test_requires_classes_dex_in_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed, include_classes_dex=False)
            payload_path = _make_payload_file(tmp)
            with self.assertRaises(SyntheticPackerError):
                build_synthetic_apk(
                    seed_apk=seed,
                    generated_apk_out=tmp / "generated.apk",
                    transform_family="multi_dex_shim",
                    rng_seed=0,
                    payload_path=payload_path,
                    enforce_payload_size_range=False,
                )


# ---------------------------------------------------------------------------
# embedded_archive
# ---------------------------------------------------------------------------


class EmbeddedArchiveTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("embedded_archive", TRANSFORMS)

    def test_emits_single_zip_suffixed_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="embedded_archive",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            injected = result.manifest["injected_objects"]
            self.assertEqual(len(injected), 1)
            rec = injected[0]
            self.assertIsNone(rec.get("host_object_path"))
            # A-v3 leakage fix (2026-05-07): with a naming profile derived
            # from the seed APK, the path no longer carries a hard-coded
            # ``.zip`` suffix or the ``embedded_archive`` family token.
            # The remaining invariants are: path is non-empty and does
            # NOT leak the family name.
            self.assertTrue(rec["object_path"])
            self.assertNotIn("embedded_archive", rec["object_path"])
            # Full-object positive range.
            self.assertEqual(rec["offset_start"], 0)
            self.assertEqual(rec["offset_end"], rec["size"])
            self.assertIsNotNone(rec["xor_key"])

    def test_inner_jar_recoverable_via_outer_xor_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)
            original_payload = payload_path.read_bytes()

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="embedded_archive",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            rec = result.manifest["injected_objects"][0]
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                encrypted_archive = archive.read(rec["object_path"])
            inner_bytes = xor_bytes(encrypted_archive, rec["xor_key"])
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_jar:
                names = set(inner_jar.namelist())
                self.assertEqual(names, {"META-INF/MANIFEST.MF", "classes.dex"})
                self.assertEqual(inner_jar.read("classes.dex"), original_payload)

    def test_encrypted_bytes_do_not_start_with_pk_magic(self):
        """Outer asset is XOR-encrypted so the first 4 bytes must not
        match ``PK\\x03\\x04`` (ZIP local file header magic).

        Otherwise naive detectors would find it as a plain nested ZIP
        and the Gen3 threat claim would be hollow.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="embedded_archive",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            rec = result.manifest["injected_objects"][0]
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                encrypted = archive.read(rec["object_path"])
            self.assertNotEqual(encrypted[:4], b"PK\x03\x04")


# ---------------------------------------------------------------------------
# dex_string_encrypted
# ---------------------------------------------------------------------------


class DexStringEncryptedTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("dex_string_encrypted", TRANSFORMS)

    def test_encrypts_string_data_span_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            host_dex_bytes = _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="dex_string_encrypted",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            injected = result.manifest["injected_objects"]
            self.assertGreaterEqual(len(injected), 1, "must emit at least one span")

            for rec in injected:
                self.assertEqual(rec["object_path"], "classes.dex")
                self.assertEqual(rec["host_object_path"], "classes.dex")
                self.assertLess(rec["offset_start"], len(host_dex_bytes))
                self.assertLessEqual(rec["offset_end"], len(host_dex_bytes))
                self.assertLess(rec["offset_start"], rec["offset_end"])
                self.assertIsNotNone(rec["xor_key"])

            # DEX written back to the APK has the same total size as the
            # original (in-place encryption, no growth).
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                new_dex = archive.read("classes.dex")
            self.assertEqual(len(new_dex), len(host_dex_bytes))
            # DEX magic preserved (string_data spans do not overlap header).
            self.assertTrue(new_dex.startswith(b"dex\n"))

    def test_positive_ranges_are_monotone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seed = tmp / "seed.apk"
            _write_seed_apk_with_host(seed)
            payload_path = _make_payload_file(tmp)

            result = build_synthetic_apk(
                seed_apk=seed,
                generated_apk_out=tmp / "generated.apk",
                transform_family="dex_string_encrypted",
                rng_seed=0,
                payload_path=payload_path,
                enforce_payload_size_range=False,
            )
            injected = result.manifest["injected_objects"]
            offsets = [r["offset_start"] for r in injected]
            self.assertEqual(offsets, sorted(offsets))


if __name__ == "__main__":
    unittest.main()
