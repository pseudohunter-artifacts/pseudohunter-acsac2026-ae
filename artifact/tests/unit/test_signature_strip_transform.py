"""Unit tests for the ``signature_strip`` adversarial transform family.

Motivation: F0c introduced ``signature_strip`` to give synthetic APKs a
hard adversarial variant. A-v2 Fix-2 (2026-04-30) extended the transform
to scramble the **full 80-byte DEX header** (not just the 8-byte magic)
using 20 independent DWORD-aligned random mask bytes, aligning with the
360 Jiagu / 爱加密 v2 Gen2 threat model. The body (offset >= 80) stays
byte-identical so detectors keyed on entropy / bigram statistics still
behave predictably.

These tests intentionally cover the transform itself in isolation, not
the end-to-end baseline behaviour (which is evaluated in F0c's
multi-baseline runner against ``configs/eval/synthetic_multi_baseline.json``).
"""

from __future__ import annotations

import random
import unittest

from android_packer.synthetic.transforms import (
    SUPPORTED_TRANSFORMS,
    TRANSFORMS,
    TransformContext,
    build_injected_payloads,
)

# The real DEX magic our transform is designed to scrub. Keep as a
# hard-coded literal so the test fails loudly if someone re-uses the
# constant from the transform module and silently changes it.
DEX_MAGIC = b"dex\n035\x00"


def _fake_dex_payload(size: int = 4096) -> bytes:
    """Build a fake DEX-ish payload: real magic + structured body.

    The body pattern is deterministic but non-uniform (simulates DEX
    opcodes with a skewed distribution), so ``entropy`` / ``printable``
    statistics differ noticeably from random bytes.
    """

    head = DEX_MAGIC
    # Body: heavily biased toward low bytes to mimic DEX opcode frequency.
    body = bytes((i % 32) ^ ((i // 16) % 4) for i in range(size - len(head)))
    return head + body


class SignatureStripTransformTests(unittest.TestCase):
    def test_registered_and_exported(self):
        self.assertIn("signature_strip", SUPPORTED_TRANSFORMS)
        self.assertIn("signature_strip", TRANSFORMS)

    def test_leading_80_bytes_scrambled_body_preserved(self):
        payload = _fake_dex_payload(4096)
        self.assertEqual(payload[:8], DEX_MAGIC, "fixture sanity")

        ctx = TransformContext(
            payload=payload,
            rng=random.Random(0),
            existing_paths=set(),
            asset_prefix="assets/payload",
            xor_key=0x5A,
            enforce_payload_size_range=False,
        )
        injected = build_injected_payloads("signature_strip", ctx)

        self.assertEqual(len(injected), 1)
        produced = injected[0].data
        # A-v2 Fix-2: the transform now scrambles the full 80-byte DEX
        # header. Same total length.
        self.assertEqual(len(produced), len(payload))
        # DEX magic (first 8 bytes) must differ.
        self.assertNotEqual(
            produced[:8],
            DEX_MAGIC,
            "signature_strip must scramble the DEX magic",
        )
        # The full 80-byte header must differ from the original header.
        self.assertNotEqual(
            produced[:80],
            payload[:80],
            "signature_strip must scramble the full 80-byte DEX header",
        )
        # But the body past the header is byte-identical so
        # structural / body-level detectors still see the original DEX.
        self.assertEqual(produced[80:], payload[80:])

    def test_xor_key_field_is_populated_and_derives_from_mask(self):
        # A-v2 Fix-2 (2026-04-30): the transform no longer uses a single
        # caller-provided XOR key; it samples an 80-byte mask from
        # ``ctx.rng``. The InjectedPayload.xor_key field is kept for
        # backwards compatibility and records the first byte of the
        # derived mask. We cannot recover the full 80-byte mask from
        # this single field, so the test only asserts the field is
        # populated and deterministic given the seed.
        def run(seed: int):
            payload = _fake_dex_payload(4096)
            ctx = TransformContext(
                payload=payload,
                rng=random.Random(seed),
                existing_paths=set(),
                asset_prefix="assets/payload",
                # ``xor_key`` is accepted but no longer controls the
                # mask under A-v2 Fix-2; kept for API back-compat.
                xor_key=0x5A,
                enforce_payload_size_range=False,
            )
            return build_injected_payloads("signature_strip", ctx)[0]

        a = run(seed=0)
        b = run(seed=0)
        c = run(seed=1)
        self.assertIsInstance(a.xor_key, int)
        self.assertTrue(0 <= a.xor_key <= 255)
        # Same seed -> same nominal key.
        self.assertEqual(a.xor_key, b.xor_key)
        # Different seed -> almost certainly different nominal key.
        # (1/256 chance of coincidental collision; accept it and use
        # the byte stream as a stronger assertion.)
        self.assertNotEqual(a.data[:80], c.data[:80])

    def test_payload_offset_spans_full_payload(self):
        # Even though only the first 8 bytes are modified, the *hidden
        # payload* region is still the full object (labels must cover it).
        payload = _fake_dex_payload(4096)
        ctx = TransformContext(
            payload=payload,
            rng=random.Random(0),
            existing_paths=set(),
            asset_prefix="assets/payload",
            xor_key=0x5A,
            enforce_payload_size_range=False,
        )
        injected = build_injected_payloads("signature_strip", ctx)[0]
        self.assertEqual(injected.payload_offset_start, 0)
        self.assertEqual(injected.payload_offset_end, len(payload))
        self.assertIsNone(injected.part_index)
        self.assertIsNone(injected.part_count)

    def test_path_falls_under_asset_prefix_with_signature_strip_token(self):
        # A-v3 leakage fix (2026-05-07): when no ``naming_profile`` is
        # provided (this test path), the legacy fallback allocator emits
        # ``<asset_prefix>/<token>.bin`` — still under the prefix, but
        # the family name is no longer baked in. Test renamed semantics:
        # path is under asset_prefix and ends in .bin, but does NOT
        # contain ``signature_strip``.
        payload = _fake_dex_payload(4096)
        ctx = TransformContext(
            payload=payload,
            rng=random.Random(0),
            existing_paths=set(),
            asset_prefix="assets/payload",
            xor_key=0x5A,
            enforce_payload_size_range=False,
        )
        injected = build_injected_payloads("signature_strip", ctx)[0]
        self.assertTrue(injected.object_path.startswith("assets/payload/"))
        # Family name MUST NOT appear (regression guard for L1 leak).
        self.assertNotIn("signature_strip", injected.object_path)
        self.assertTrue(injected.object_path.endswith(".bin"))

    def test_payload_too_short_raises(self):
        # < 80 bytes: the full-header mask is undefined; must refuse.
        ctx = TransformContext(
            payload=b"abc",
            rng=random.Random(0),
            existing_paths=set(),
            asset_prefix="assets/payload",
            xor_key=0x5A,
            enforce_payload_size_range=False,
        )
        with self.assertRaises(ValueError):
            build_injected_payloads("signature_strip", ctx)

    def test_deterministic_given_same_seed_and_key(self):
        payload = _fake_dex_payload(4096)

        def run():
            ctx = TransformContext(
                payload=payload,
                rng=random.Random(42),
                existing_paths=set(),
                asset_prefix="assets/payload",
                xor_key=0x77,
                enforce_payload_size_range=False,
            )
            return build_injected_payloads("signature_strip", ctx)[0]

        a, b = run(), run()
        self.assertEqual(a.data, b.data)
        self.assertEqual(a.object_path, b.object_path)


if __name__ == "__main__":
    unittest.main()
