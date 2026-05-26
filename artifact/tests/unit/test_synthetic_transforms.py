import random
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    # ``_dex_fixtures`` lives next to this test file; it provides
    # :func:`build_minimal_dex` which we use as a parseable DEX host for
    # the Gen3 ``dex_method_inlined`` transform.
    sys.path.insert(0, str(_TESTS_DIR))

from _dex_fixtures import build_minimal_dex  # noqa: E402

from android_packer.synthetic import (  # noqa: E402
    SUPPORTED_TRANSFORMS,
    TRANSFORMS,
    TransformContext,
    build_synthetic_apk,
    register_transform,
)
from android_packer.synthetic.records import InjectedPayload  # noqa: E402


def _write_seed_apk(path: Path) -> bytes:
    payload = b"dex\n035\x00" + (b"payload bytes " * 16)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.dex", payload)
        archive.writestr("assets/readme.txt", b"benign")
    return payload


def _write_gen3_seed_apk(path: Path, payload_path: Path) -> dict:
    """Seed APK that exercises all three sub-range (Gen3) transforms.

    Contains a parseable host DEX (big enough for ``dex_method_inlined``),
    a large asset (for ``embedded_asset``), and a valid ELF .so (for
    ``so_embedded``). The payload bytes are written to ``payload_path``
    so the caller passes them via ``payload_path=`` to ``build_synthetic_apk``;
    this keeps the payload source out of ``seed_objects`` and guarantees
    the classes.dex host isn't taken as the payload source itself.

    Returns a dict describing each fixture for assertions:
    ``payload_bytes``, ``host_dex_bytes``, ``asset_bytes``, ``so_bytes``.
    """

    # Payload DEX: needs to clear B1's per-family minimum (64 KiB for both
    # whole-object and sub-range families, see transforms.py
    # _PAYLOAD_SIZE_SUB_RANGE / _PAYLOAD_SIZE_WHOLE_OBJECT). We pad the DEX
    # magic with enough filler to reach 80 KiB so the transforms accept it
    # without needing the seed-size heuristic to be bypassed.
    payload_bytes = b"dex\n035\x00" + (b"payload bytes " * 6000)
    assert len(payload_bytes) >= 64 * 1024
    payload_path.write_bytes(payload_bytes)

    # Host DEX for dex_method_inlined: needs file size >= 32 KiB (the
    # min_host_bytes guard in transforms.py, protecting against tiny DEX
    # hosts where the inlined region would dominate). B2 additionally
    # requires at least 3 code_item spans >= _DEX_INLINE_MIN_SPAN_SIZE
    # (512 B) to scatter across; num_code_items=4000 gives 20*4000 = 80KB
    # of code_item, which build_minimal_dex emits as a single span. We
    # side-step that by calling build_minimal_dex once with a large
    # num_code_items and letting the parser roll it up: the B2 code path
    # only checks span_count >= pool_size, and ``pool_size = min(ranked,
    # k_max)`` so a single giant span is OK — it just means k clamps to
    # the number of distinct spans (often 1 in our fixtures). See the
    # dedicated B2 test below for the multi-span assertion.
    host_dex_bytes, _layout = build_minimal_dex(num_code_items=4000)

    # Host asset for embedded_asset: >= 128 KiB of mixed benign bytes, so
    # the B1 ratio constraint (p/total <= 0.75) holds for an 80 KB
    # payload (need host >= 80 KB * (1/0.75 - 1) ≈ 27 KB; 128 KB is a
    # comfortable margin and also ensures the generated entry crosses
    # several region windows for downstream tests).
    rng = random.Random(0xB00B1E5)
    asset_bytes = (
        b"license header\n"
        + b"".join(
            bytes(rng.choices(b"ABCDEF0123456789 \n", k=512))
            for _ in range(256)
        )
    )
    assert len(asset_bytes) >= 128 * 1024

    # Host ELF .so: minimal 64-bit LE ELF header + padding to 128 KiB so
    # the same p/total <= 0.75 constraint holds. readelf won't love it,
    # but our so_embedded host filter only checks ``startswith(b"\x7fELF")``
    # and size.
    elf_header = (
        b"\x7fELF"           # magic
        b"\x02"              # 64-bit
        b"\x01"              # little-endian
        b"\x01"              # ELF version 1
        b"\x00"              # ABI: System V
        + b"\x00" * 8        # pad
        + b"\x03\x00"        # e_type = ET_DYN (.so)
        + b"\xb7\x00"        # e_machine = AArch64 (arbitrary)
        + b"\x01\x00\x00\x00"  # e_version
    )
    so_bytes = elf_header + b"\x00" * (128 * 1024 - len(elf_header))

    with zipfile.ZipFile(path, "w") as archive:
        # Only one DEX in the seed APK (the host). The payload is external.
        archive.writestr("classes.dex", host_dex_bytes)
        archive.writestr("assets/license.txt", asset_bytes)
        archive.writestr("lib/arm64-v8a/libdemo.so", so_bytes)

    return {
        "payload_bytes": payload_bytes,
        "host_dex_bytes": host_dex_bytes,
        "asset_bytes": asset_bytes,
        "so_bytes": so_bytes,
    }


class TransformRegistryTests(unittest.TestCase):
    def test_builtin_families_are_registered(self):
        self.assertIn("xor", TRANSFORMS)
        self.assertIn("base64", TRANSFORMS)
        self.assertIn("split_xor", TRANSFORMS)
        self.assertIn("path_randomized", TRANSFORMS)
        # Gen3 sub-range families added in April 2026. See
        # docs/method/threat_model.md §"Synthetic 威胁覆盖矩阵".
        self.assertIn("embedded_asset", TRANSFORMS)
        self.assertIn("so_embedded", TRANSFORMS)
        self.assertIn("dex_method_inlined", TRANSFORMS)
        self.assertEqual(set(SUPPORTED_TRANSFORMS), set(TRANSFORMS))

    def test_register_transform_exposes_new_family(self):
        family_name = "prefixed_xor_test"
        self.addCleanup(TRANSFORMS.pop, family_name, None)

        def _build(ctx: TransformContext):
            # Prefix the payload with a literal header, then XOR with a
            # deterministic key. Used purely to exercise the registration
            # path; no attempt at realism.
            transformed = b"PREFIX" + bytes(b ^ 0x55 for b in ctx.payload)
            return [
                InjectedPayload(
                    object_path=f"assets/synthetic/{family_name}.bin",
                    data=transformed,
                    transform_family=family_name,
                    payload_offset_start=0,
                    payload_offset_end=len(ctx.payload),
                    part_index=None,
                    part_count=None,
                    xor_key=0x55,
                )
            ]

        register_transform(family_name, _build)

        # The mutable registry is authoritative. ``SUPPORTED_TRANSFORMS``
        # names imported at module load time are snapshots and may lag behind
        # until callers re-import the symbol.
        self.assertIn(family_name, TRANSFORMS)
        # However, the freshly imported module-level ``SUPPORTED_TRANSFORMS``
        # must contain the new family.
        from android_packer.synthetic import transforms as transforms_module

        self.assertIn(family_name, transforms_module.SUPPORTED_TRANSFORMS)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload = _write_seed_apk(seed_apk)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "generated.apk",
                transform_family=family_name,
                rng_seed=1,
            )

            injected = result.manifest["injected_objects"][0]
            self.assertEqual(injected["transform_family"], family_name)
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                transformed = archive.read(injected["object_path"])
            self.assertTrue(transformed.startswith(b"PREFIX"))
            # Recover payload: strip prefix, reverse XOR.
            recovered = bytes(b ^ 0x55 for b in transformed[len(b"PREFIX"):])
            self.assertEqual(recovered, payload)

    def test_register_transform_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            register_transform("xor", lambda ctx: [])


class SubRangeTransformTests(unittest.TestCase):
    """Gen3 sub-range transforms: embed payload into an existing host.

    These tests pin down the invariants that the downstream labelling and
    region-entropy plumbing relies on:

    1. The host ZIP entry is *overwritten*, not duplicated (otherwise the
       generated APK ends up with both the original benign asset and the
       payload, which inflates the negative region count for the baseline
       entropy comparison and defeats the Gen3 threat model).
    2. ``offset_start`` / ``offset_end`` in the emitted ``SyntheticLabel``
       match the object-local byte range of the payload inside the
       overwritten host, so ``build_training_labels`` can align it
       correctly to the extracted regions.
    3. The host's pre-payload bytes are preserved verbatim, which is what
       lets ``entropy_delta_entry`` see benign/payload contrast.
    """

    def test_embedded_asset_overwrites_host_and_preserves_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload_file = tmp / "payload.dex"
            fixtures = _write_gen3_seed_apk(seed_apk, payload_file)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "gen.apk",
                payload_path=payload_file,
                transform_family="embedded_asset",
                rng_seed=7,
            )

            # Exactly one label, pointing at the host asset.
            self.assertEqual(len(result.labels), 1)
            label = result.labels[0]
            self.assertEqual(label.object_path, "assets/license.txt")
            # Sub-range label: offset range equals payload position in host.
            self.assertEqual(label.offset_start, len(fixtures["asset_bytes"]))
            self.assertEqual(
                label.offset_end,
                len(fixtures["asset_bytes"]) + len(fixtures["payload_bytes"]),
            )
            # Host object is present once in the generated APK (no duplicate).
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                names = archive.namelist()
                self.assertEqual(names.count("assets/license.txt"), 1)
                overwritten = archive.read("assets/license.txt")
            # Host prefix preserved verbatim.
            self.assertTrue(overwritten.startswith(fixtures["asset_bytes"]))
            # Payload tail is XOR-encrypted; verify we can recover it.
            xor_key = result.manifest["parameters"]["xor_keys"][0]
            encrypted = overwritten[len(fixtures["asset_bytes"]):]
            recovered = bytes(b ^ xor_key for b in encrypted)
            self.assertEqual(recovered, fixtures["payload_bytes"])

    def test_so_embedded_keeps_elf_magic_at_offset_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload_file = tmp / "payload.dex"
            fixtures = _write_gen3_seed_apk(seed_apk, payload_file)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "gen.apk",
                payload_path=payload_file,
                transform_family="so_embedded",
                rng_seed=11,
            )

            self.assertEqual(len(result.labels), 1)
            label = result.labels[0]
            self.assertEqual(label.object_path, "lib/arm64-v8a/libdemo.so")
            self.assertEqual(label.offset_start, len(fixtures["so_bytes"]))

            with zipfile.ZipFile(result.generated_apk_path) as archive:
                overwritten = archive.read("lib/arm64-v8a/libdemo.so")
            # ELF magic must still be at offset 0 so "readelf -h" style
            # probes keep classifying this entry as a legitimate .so.
            self.assertEqual(overwritten[:4], b"\x7fELF")
            self.assertTrue(overwritten.startswith(fixtures["so_bytes"]))

    def test_dex_method_inlined_overwrites_code_item_tail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload_file = tmp / "payload.dex"
            fixtures = _write_gen3_seed_apk(seed_apk, payload_file)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "gen.apk",
                payload_path=payload_file,
                transform_family="dex_method_inlined",
                rng_seed=13,
            )

            # B2: transform emits >= 1 InjectedPayload (multiple segments
            # possible). Each label must land inside the host DEX and
            # target the same object.
            self.assertGreaterEqual(len(result.labels), 1)
            for label in result.labels:
                self.assertEqual(label.object_path, "classes.dex")
                self.assertLess(label.offset_start, label.offset_end)
                self.assertLessEqual(
                    label.offset_end, len(fixtures["host_dex_bytes"])
                )

            with zipfile.ZipFile(result.generated_apk_path) as archive:
                overwritten = archive.read("classes.dex")
            # DEX file length unchanged (Gen3 inlining does not resize the
            # host; it only replaces slices of the code_item span).
            self.assertEqual(len(overwritten), len(fixtures["host_dex_bytes"]))
            # DEX header (0x70 bytes) must be byte-identical to the host so
            # that parse_dex_item_spans still succeeds on the overwritten
            # DEX (a downstream invariant the pretrain MLM corpus depends
            # on; see docs/method/ours_method_spec.md §5.1).
            self.assertEqual(overwritten[:0x70], fixtures["host_dex_bytes"][:0x70])
            # Bytes inside each segment differ from host; bytes in the
            # gaps between segments and before the first segment are
            # byte-identical to host.
            segments = sorted(
                [(lbl.offset_start, lbl.offset_end) for lbl in result.labels]
            )
            for s, e in segments:
                self.assertNotEqual(
                    overwritten[s:e], fixtures["host_dex_bytes"][s:e]
                )
            # Prefix bytes (before first segment) untouched.
            first_start = segments[0][0]
            self.assertEqual(
                overwritten[:first_start],
                fixtures["host_dex_bytes"][:first_start],
            )
            # Gap between consecutive segments untouched.
            for (_s1, e1), (s2, _e2) in zip(segments, segments[1:]):
                self.assertEqual(
                    overwritten[e1:s2], fixtures["host_dex_bytes"][e1:s2]
                )

    def test_dex_method_inlined_b2_multi_segment_invariants(self):
        """B2 (2026-04-29): scatter invariants for dex_method_inlined.

        Pins down the three invariants the Gen3 "single-method hiding"
        threat model relies on and the downstream labelling/alignment
        code assumes:

        * Each segment size falls in [_DEX_INLINE_SEG_SIZE_RANGE].
        * Segments are pairwise non-overlapping (the scatter is honest,
          not one big overwrite in disguise).
        * Segments carry **different** XOR keys (independent-key story).

        We do *not* assert ``k >= 3`` here because the fixture's DEX is
        built by build_minimal_dex which rolls all code_items into a
        single span; k can still be as small as 1 on tiny fixtures.
        Production runs on F-Droid / AndroZoo APKs reliably produce
        k ∈ [3, 10] thanks to the much larger code_item span — see
        scripts/_inspect_gen3_payload_ratio.py for post-run audit.
        """

        from android_packer.synthetic.transforms import (
            _DEX_INLINE_SEG_SIZE_RANGE,
        )
        seg_lo, seg_hi = _DEX_INLINE_SEG_SIZE_RANGE

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload_file = tmp / "payload.dex"
            _write_gen3_seed_apk(seed_apk, payload_file)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "gen.apk",
                payload_path=payload_file,
                transform_family="dex_method_inlined",
                rng_seed=13,
            )

            # Segment size bounds.
            for label in result.labels:
                size = label.offset_end - label.offset_start
                self.assertGreaterEqual(
                    size, seg_lo,
                    f"segment size {size} < lower bound {seg_lo}",
                )
                self.assertLessEqual(
                    size, seg_hi,
                    f"segment size {size} > upper bound {seg_hi}",
                )

            # Non-overlapping.
            sorted_ranges = sorted(
                [(lbl.offset_start, lbl.offset_end) for lbl in result.labels]
            )
            for (s1, e1), (s2, e2) in zip(sorted_ranges, sorted_ranges[1:]):
                self.assertLessEqual(
                    e1, s2,
                    f"segments overlap: [{s1},{e1}) and [{s2},{e2})",
                )

            # Independent XOR keys whenever k > 1. With k == 1 there is
            # nothing to assert on key diversity. The manifest stores the
            # *set* of keys (deduplicated) in ``parameters.xor_keys`` —
            # so if there are k > 1 segments but fewer unique keys than
            # segments, the B2 independence invariant is violated.
            unique_keys = set(result.manifest["parameters"]["xor_keys"])
            n_segments = len(result.labels)
            if n_segments > 1:
                # Can't assert ``len(unique_keys) == n_segments`` because
                # random 1..255 draws with small k have a nonzero (but
                # small) collision probability; assert the weaker but
                # still-meaningful "not all identical".
                self.assertGreaterEqual(
                    len(unique_keys), 2,
                    "dex_method_inlined with k > 1 must use at least "
                    "two distinct XOR keys; found only one",
                )

    def test_embedded_asset_raises_when_no_host_available(self):
        """If the seed APK has no suitable host, the transform must fail loudly.

        Silent fallbacks (e.g., downgrading to whole-object injection)
        would make ``region_labels.jsonl`` inconsistent with what the
        CLI advertises as the active transform family; failing loudly
        is the contract enforced here.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            # Seed APK carries only a DEX (tiny) and a tiny readme, so no
            # asset candidate meets the 16 KiB minimum.
            _write_seed_apk(seed_apk)

            from android_packer.synthetic.records import SyntheticPackerError
            with self.assertRaises(SyntheticPackerError):
                build_synthetic_apk(
                    seed_apk=seed_apk,
                    generated_apk_out=tmp / "gen.apk",
                    transform_family="embedded_asset",
                    rng_seed=0,
                )


if __name__ == "__main__":
    unittest.main()
