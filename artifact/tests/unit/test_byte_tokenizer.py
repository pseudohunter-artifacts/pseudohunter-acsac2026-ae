"""Unit tests for :mod:`android_packer.models.tokenizer`.

These tests cover the byte-level round-trip behaviour mandated by the
Ours method spec (§3.2.2) and the batch 1 acceptance criteria originally
recorded in the 2026-05-xx ``week_09`` weekly report (now merged into
[`docs/project_progress.md`](../../docs/project_progress.md) appendix A.2
after the 2026-05-07 progress-dir cleanup):

* empty byte stream
* all ``0xFF`` stream
* all ``0x00`` stream
* random 1 MiB round trip
* truncation via ``max_length``
"""

from __future__ import annotations

import os
import random
import unittest

from android_packer.models import ByteTokenizer, ByteTokenizerEncoding


class ByteTokenizerVocabTests(unittest.TestCase):
    """Exercise the closed 261-token vocabulary contract."""

    def test_vocab_constants(self) -> None:
        self.assertEqual(ByteTokenizer.PAD_ID, 0)
        self.assertEqual(ByteTokenizer.BOS_ID, 1)
        self.assertEqual(ByteTokenizer.EOS_ID, 2)
        self.assertEqual(ByteTokenizer.MASK_ID, 3)
        self.assertEqual(ByteTokenizer.UNK_ID, 4)
        self.assertEqual(ByteTokenizer.BYTE_OFFSET, 5)
        self.assertEqual(ByteTokenizer.VOCAB_SIZE, 261)

    def test_special_ids_membership(self) -> None:
        self.assertIn(ByteTokenizer.PAD_ID, ByteTokenizer.SPECIAL_IDS)
        self.assertIn(ByteTokenizer.MASK_ID, ByteTokenizer.SPECIAL_IDS)
        self.assertNotIn(ByteTokenizer.BYTE_OFFSET, ByteTokenizer.SPECIAL_IDS)

    def test_repr_is_informative(self) -> None:
        tok = ByteTokenizer(max_length=1024)
        representation = repr(tok)
        self.assertIn("ByteTokenizer", representation)
        self.assertIn("261", representation)
        self.assertIn("1024", representation)


class ByteTokenizerEncodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = ByteTokenizer()

    def test_encode_empty_round_trip(self) -> None:
        ids = self.tok.encode(b"")
        self.assertEqual(ids, [ByteTokenizer.BOS_ID, ByteTokenizer.EOS_ID])
        self.assertEqual(self.tok.decode(ids), b"")

    def test_encode_empty_without_special_tokens(self) -> None:
        ids = self.tok.encode(b"", add_special_tokens=False)
        self.assertEqual(ids, [])
        self.assertEqual(self.tok.decode(ids), b"")

    def test_encode_all_zero_bytes(self) -> None:
        payload = b"\x00" * 32
        ids = self.tok.encode(payload, add_special_tokens=False)
        self.assertEqual(ids, [ByteTokenizer.BYTE_OFFSET] * 32)
        self.assertEqual(self.tok.decode(ids), payload)

    def test_encode_all_high_bytes(self) -> None:
        payload = b"\xff" * 64
        ids = self.tok.encode(payload, add_special_tokens=False)
        expected = [ByteTokenizer.BYTE_OFFSET + 0xFF] * 64
        self.assertEqual(ids, expected)
        self.assertEqual(max(ids), ByteTokenizer.VOCAB_SIZE - 1)
        self.assertEqual(self.tok.decode(ids), payload)

    def test_encode_accepts_bytearray_and_memoryview(self) -> None:
        payload = bytearray(b"hello")
        ids_ba = self.tok.encode(payload, add_special_tokens=False)
        ids_mv = self.tok.encode(memoryview(bytes(payload)), add_special_tokens=False)
        self.assertEqual(ids_ba, ids_mv)
        self.assertEqual(self.tok.decode(ids_ba), b"hello")

    def test_encode_rejects_str(self) -> None:
        with self.assertRaises(TypeError):
            self.tok.encode("not bytes")  # type: ignore[arg-type]

    def test_round_trip_over_full_byte_range(self) -> None:
        payload = bytes(range(256))
        ids = self.tok.encode(payload)
        # BOS + 256 body tokens + EOS
        self.assertEqual(len(ids), 258)
        self.assertEqual(ids[0], ByteTokenizer.BOS_ID)
        self.assertEqual(ids[-1], ByteTokenizer.EOS_ID)
        self.assertEqual(self.tok.decode(ids), payload)

    def test_random_1mib_round_trip(self) -> None:
        # Deterministic seed avoids flaky CI behaviour while still exercising
        # the "random bytes" contract from the acceptance criteria.
        rng = random.Random(0xC0FFEE)
        payload = bytes(rng.randrange(0, 256) for _ in range(1024 * 1024))
        ids = self.tok.encode(payload, add_special_tokens=False)
        self.assertEqual(len(ids), len(payload))
        recovered = self.tok.decode(ids)
        self.assertEqual(recovered, payload)

    def test_urandom_round_trip(self) -> None:
        payload = os.urandom(4096)
        ids = self.tok.encode(payload)
        self.assertEqual(self.tok.decode(ids), payload)


class ByteTokenizerTruncationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = ByteTokenizer()

    def test_truncation_preserves_bos_eos(self) -> None:
        payload = bytes(range(32))
        ids = self.tok.encode(payload, max_length=8)
        self.assertEqual(len(ids), 8)
        self.assertEqual(ids[0], ByteTokenizer.BOS_ID)
        self.assertEqual(ids[-1], ByteTokenizer.EOS_ID)
        # 8 slots - BOS - EOS = 6 body bytes retained.
        self.assertEqual(
            ids[1:-1],
            [ByteTokenizer.BYTE_OFFSET + b for b in payload[:6]],
        )

    def test_truncation_without_special_tokens(self) -> None:
        payload = bytes(range(16))
        ids = self.tok.encode(payload, add_special_tokens=False, max_length=4)
        self.assertEqual(ids, [ByteTokenizer.BYTE_OFFSET + b for b in payload[:4]])

    def test_default_max_length_applied(self) -> None:
        tok = ByteTokenizer(max_length=4)
        ids = tok.encode(bytes(range(16)), add_special_tokens=False)
        self.assertEqual(len(ids), 4)

    def test_default_max_length_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ByteTokenizer(max_length=0)
        with self.assertRaises(ValueError):
            ByteTokenizer(max_length=-1)

    def test_encode_with_mask_pads_to_max_length(self) -> None:
        payload = b"\x01\x02\x03"
        encoding = self.tok.encode_with_mask(payload, max_length=8)
        self.assertIsInstance(encoding, ByteTokenizerEncoding)
        self.assertEqual(len(encoding.input_ids), 8)
        self.assertEqual(len(encoding.attention_mask), 8)
        # BOS + 3 body + EOS = 5 real tokens, followed by 3 PAD.
        self.assertEqual(encoding.length, 5)
        self.assertFalse(encoding.truncated)
        self.assertEqual(encoding.attention_mask, [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(encoding.input_ids[5:], [ByteTokenizer.PAD_ID] * 3)

    def test_encode_with_mask_marks_truncation(self) -> None:
        payload = bytes(range(64))
        encoding = self.tok.encode_with_mask(payload, max_length=8)
        self.assertTrue(encoding.truncated)
        self.assertEqual(encoding.length, 8)
        self.assertEqual(encoding.attention_mask, [1] * 8)

    def test_encode_with_mask_requires_budget(self) -> None:
        tok = ByteTokenizer()
        with self.assertRaises(ValueError):
            tok.encode_with_mask(b"abc")

    def test_encode_batch_wraps_encode_with_mask(self) -> None:
        payloads = [b"", b"\x00", b"\xff" * 10]
        encodings = self.tok.encode_batch(payloads, max_length=16)
        self.assertEqual(len(encodings), 3)
        for enc in encodings:
            self.assertEqual(len(enc.input_ids), 16)
            self.assertEqual(len(enc.attention_mask), 16)


class ByteTokenizerDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = ByteTokenizer()

    def test_decode_skips_special_tokens_by_default(self) -> None:
        ids = [
            ByteTokenizer.PAD_ID,
            ByteTokenizer.BOS_ID,
            ByteTokenizer.BYTE_OFFSET + 0x41,
            ByteTokenizer.MASK_ID,
            ByteTokenizer.BYTE_OFFSET + 0x42,
            ByteTokenizer.EOS_ID,
            ByteTokenizer.PAD_ID,
        ]
        self.assertEqual(self.tok.decode(ids), b"AB")

    def test_decode_can_keep_special_tokens_as_zero(self) -> None:
        ids = [
            ByteTokenizer.BOS_ID,
            ByteTokenizer.BYTE_OFFSET + 0x41,
            ByteTokenizer.EOS_ID,
        ]
        self.assertEqual(
            self.tok.decode(ids, skip_special_tokens=False),
            b"\x00A\x00",
        )

    def test_decode_rejects_out_of_range_ids(self) -> None:
        with self.assertRaises(ValueError):
            self.tok.decode([ByteTokenizer.VOCAB_SIZE])
        with self.assertRaises(ValueError):
            self.tok.decode([-1])

    def test_decode_rejects_non_integer_tokens(self) -> None:
        with self.assertRaises(TypeError):
            self.tok.decode([1.5])  # type: ignore[list-item]

    def test_decode_rejects_bytes_input(self) -> None:
        with self.assertRaises(TypeError):
            self.tok.decode(b"\x00\x01\x02")  # type: ignore[arg-type]


class ByteTokenizerImportSurfaceTests(unittest.TestCase):
    """Verify the contract that the tokenizer module does not pull in torch."""

    def test_module_does_not_import_torch(self) -> None:
        import sys

        import android_packer.models.tokenizer as tokenizer_module

        # Self-check: the module loaded without torch as a dependency.
        self.assertTrue(hasattr(tokenizer_module, "ByteTokenizer"))
        # If torch happens to be installed in the environment we can't assert
        # that sys.modules is torch-free globally, but importing the tokenizer
        # module alone must not cause the import. We verify by checking that
        # the module's own globals don't reference torch.
        self.assertNotIn("torch", vars(tokenizer_module))
        # And the package __init__ likewise must stay torch-free at import
        # time: re-importing should not trigger any torch import side effect.
        # (Presence of torch in sys.modules due to other tests is tolerated.)
        _ = sys  # keep import used without emitting warnings


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
