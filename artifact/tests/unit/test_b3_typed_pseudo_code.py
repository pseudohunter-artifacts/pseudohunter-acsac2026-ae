from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from android_packer.decoders.byte_pattern_decoder import decode_byte_pattern_region
from android_packer.decoders.dalvik_decoder import decode_dalvik_region
from android_packer.decoders.native_decoder import decode_native_region
from android_packer.decoders.pseudo_tokenizer import (
    BYTE_REPRESENTATION_LEGACY_RAW,
    BYTE_REPRESENTATION_TYPED_V1,
    PseudoCodeTokenizer,
    TOKEN_TYPE_BYTE,
    UNIFIED_VOCAB_SIZE_LEGACY_RAW,
    UNIFIED_VOCAB_SIZE_TYPED_V1,
)
from android_packer.models.fusion_encoder import FusionEncoderConfig


ROOT = Path(__file__).resolve().parents[2]
LOPO_SCRIPT = ROOT / "scripts" / "experiments" / "run_lopo_eval.py"


def _load_lopo_module():
    spec = importlib.util.spec_from_file_location("run_lopo_eval_b3_test", LOPO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dalvik_decoder_emits_typed_components() -> None:
    tokens = decode_dalvik_region(
        bytes([0x6e, 0x10, 0x05, 0x00, 0x00, 0x00]),
        max_tokens=12,
        method_ids_count=2,
    )
    names = [token.token for token in tokens]

    assert "invoke" in names
    assert "INVOKE_VIRTUAL" in names
    assert "METHOD_IDX_BAD" in names
    assert "REG_LIST" in names
    assert any(token.is_abnormal for token in tokens if token.token == "METHOD_IDX_BAD")


def test_native_decoder_emits_typed_components() -> None:
    # ARM64 B with an out-of-range target when code_size is tiny.
    tokens = decode_native_region(
        bytes.fromhex("05000014"),
        arch="arm64",
        max_tokens=8,
        code_size=4,
    )
    names = [token.token for token in tokens]

    assert "branch_uncond" in names
    assert "TARGET_BAD" in names
    assert any(token.is_abnormal for token in tokens if token.token == "TARGET_BAD")

    svc_tokens = decode_native_region(
        bytes.fromhex("010000d4"),
        arch="arm64",
        max_tokens=8,
    )
    assert "SYSCALL_LIKE" in [token.token for token in svc_tokens]


def test_byte_pattern_decoder_emits_context_and_shape_tokens() -> None:
    tokens = decode_byte_pattern_region(
        b"dex\n035\x00" + b"\x00" * 40,
        entry_type="dex",
        max_tokens=16,
    )
    names = [token.token for token in tokens]

    assert names[0] == "[BOS]"
    assert "ENTRY_DEX" in names
    assert "MAGIC_DEX" in names
    assert "ENTROPY_LOW" in names
    assert "RUN_ZERO_LONG" in names
    assert "PATTERN_END" in names
    assert names[-1] == "[EOS]"


def test_typed_byte_tokenizer_uses_pattern_vocab_not_raw_bytes() -> None:
    legacy = PseudoCodeTokenizer(max_length=16)
    typed = PseudoCodeTokenizer(
        max_length=16,
        byte_representation=BYTE_REPRESENTATION_TYPED_V1,
    )

    legacy_enc = legacy.encode_bytes(b"dex\n035\x00", entry_type="dex")
    typed_enc = typed.encode_bytes(b"dex\n035\x00", entry_type="dex")

    assert legacy.byte_representation == BYTE_REPRESENTATION_LEGACY_RAW
    assert typed.byte_representation == BYTE_REPRESENTATION_TYPED_V1
    assert legacy.vocab_size == UNIFIED_VOCAB_SIZE_LEGACY_RAW
    assert typed.vocab_size == UNIFIED_VOCAB_SIZE_TYPED_V1
    assert legacy.vocab_size == 358
    assert typed.vocab_size > 100
    assert legacy_enc.token_type_ids == [TOKEN_TYPE_BYTE] * 16
    assert typed_enc.token_type_ids == [TOKEN_TYPE_BYTE] * 16
    assert typed_enc.length < legacy_enc.length
    assert typed_enc.token_ids != legacy_enc.token_ids

    dalvik_enc, native_enc, _ = typed.encode_region(
        bytes([0x6e, 0x10, 0x05, 0x00, 0x00, 0x00]),
        entry_type="dex",
        dex_header_counts=(0, 0, 2, 0),
    )
    assert max(dalvik_enc.token_ids) < typed.vocab_size
    assert max(native_enc.token_ids) < typed.vocab_size


def test_fusion_config_records_byte_representation() -> None:
    cfg = FusionEncoderConfig(byte_representation=BYTE_REPRESENTATION_TYPED_V1)

    assert cfg.byte_representation == BYTE_REPRESENTATION_TYPED_V1


def test_lopo_cache_key_includes_byte_representation(tmp_path: Path) -> None:
    module = _load_lopo_module()
    apk = tmp_path / "sample.apk"
    apk.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    legacy_path = module._bag_cache_path(
        apk,
        0,
        "sample",
        None,
        tmp_path,
        BYTE_REPRESENTATION_LEGACY_RAW,
    )
    typed_path = module._bag_cache_path(
        apk,
        0,
        "sample",
        None,
        tmp_path,
        BYTE_REPRESENTATION_TYPED_V1,
    )

    assert legacy_path != typed_path
