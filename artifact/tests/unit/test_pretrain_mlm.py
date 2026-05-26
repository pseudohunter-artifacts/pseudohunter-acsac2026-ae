"""Unit tests for :mod:`android_packer.training.pretrain_mlm` (F-MIL-c).

These tests exercise the **pure stdlib** portion of the pre-training
stack:

* corpus construction with the benign-only hard constraint;
* MLM collator 80/10/10 masking arithmetic;
* per-token item-type label plumbing (ignore_index === DEX_ITEM_TYPE_PAD_ID);
* ``item_type_aux_weight == 0`` short-circuit on loss assembly (torch-only
  test, skipped when [dl] not available).
"""

from __future__ import annotations

import random

import pytest

from android_packer.features.dex_item_parser import DEX_ITEM_TYPES
from android_packer.models.item_type_head import DEX_ITEM_TYPE_PAD_ID
from android_packer.models.tokenizer import ByteTokenizer
from android_packer.training.pretrain_mlm import (
    BenignCorpusError,
    MLMCollator,
    MLMCorpusBuilder,
    MLMCorpusStats,
    PretrainMLMConfig,
    build_mlm_corpus,
    build_mlm_example,
    compute_pretrain_loss,
)

from tests.unit._dex_fixtures import build_minimal_dex


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPretrainMLMConfig:
    def test_defaults_are_valid(self):
        cfg = PretrainMLMConfig()
        assert cfg.item_type_aux_weight == pytest.approx(0.2)
        assert cfg.benign_exclusion_max_ratio == pytest.approx(0.05)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_seq_length": 1},
            {"mlm_mask_prob": 0.0},
            {"mlm_mask_prob": 1.0},
            {"mlm_replace_mask_prob": -0.1},
            {"mlm_replace_random_prob": 1.1},
            {"item_type_aux_weight": -0.01},
            {"benign_exclusion_max_ratio": -0.01},
            {"benign_exclusion_max_ratio": 1.01},
        ],
    )
    def test_rejects_out_of_range(self, kwargs):
        with pytest.raises(ValueError):
            PretrainMLMConfig(**kwargs)

    def test_rejects_mask_plus_random_over_one(self):
        with pytest.raises(ValueError):
            PretrainMLMConfig(
                mlm_replace_mask_prob=0.7,
                mlm_replace_random_prob=0.5,
            )


# ---------------------------------------------------------------------------
# build_mlm_example
# ---------------------------------------------------------------------------


class TestBuildMLMExample:
    def test_produces_correct_shape_and_labels(self):
        dex, layout = build_minimal_dex()
        tok = ByteTokenizer(max_length=4098)
        ex = build_mlm_example(dex, tokenizer=tok, max_seq_length=4098)

        assert len(ex.input_ids) == 4098
        assert len(ex.attention_mask) == 4098
        assert len(ex.item_type_labels) == 4098
        # BOS/EOS positions carry PAD id.
        assert ex.item_type_labels[0] == DEX_ITEM_TYPE_PAD_ID
        # Position 1 is the first DEX body byte == header[0] == 'd' == 0x64,
        # and it must fall in the "header" item-type (id 0).
        assert ex.item_type_labels[1] == DEX_ITEM_TYPES.index("header")

    def test_rejects_non_benign_dex(self):
        # Random bytes will fail parse_dex_item_spans.
        bogus = b"\x00" * 512
        tok = ByteTokenizer(max_length=4098)
        with pytest.raises(Exception):  # noqa: BLE001 — either DexParseError or ValueError
            build_mlm_example(bogus, tokenizer=tok, max_seq_length=4098)

    def test_source_length_tracks_body_budget(self):
        dex, _layout = build_minimal_dex()
        tok = ByteTokenizer(max_length=4098)
        ex = build_mlm_example(dex, tokenizer=tok, max_seq_length=4098)
        # max_seq_length=4098 minus BOS/EOS gives 4096 body budget; the
        # minimal DEX is well under that, so body_len should equal
        # len(dex).
        assert ex.source_length == len(dex)


# ---------------------------------------------------------------------------
# MLMCorpusBuilder — benign-only invariant
# ---------------------------------------------------------------------------


class TestMLMCorpusBuilder:
    def test_keeps_benign_and_drops_garbage(self):
        dex, _ = build_minimal_dex()
        cfg = PretrainMLMConfig(
            benign_exclusion_max_ratio=0.5,  # permissive for this test
        )
        builder = MLMCorpusBuilder(cfg)
        builder.add(dex)
        builder.add(dex)
        builder.add(b"\x00" * 512)  # non-benign
        examples, stats = builder.finalise()

        assert stats.total_seen == 3
        assert stats.kept == 2
        assert stats.dropped_parse_error == 1
        assert len(examples) == 2

    def test_raises_when_exclusion_ratio_exceeds_cap(self):
        dex, _ = build_minimal_dex()
        cfg = PretrainMLMConfig(benign_exclusion_max_ratio=0.05)
        builder = MLMCorpusBuilder(cfg)
        builder.add(dex)
        # 2 non-benign out of 3 → 67 % exclusion, well over 5 %.
        builder.add(b"\x00" * 256)
        builder.add(b"garbage")
        with pytest.raises(BenignCorpusError) as exc_info:
            builder.finalise()
        assert "exclusion ratio" in str(exc_info.value)
        assert "ours_method_spec" in str(exc_info.value)

    def test_raises_on_empty_corpus(self):
        cfg = PretrainMLMConfig()
        builder = MLMCorpusBuilder(cfg)
        with pytest.raises(BenignCorpusError):
            builder.finalise()

    def test_raises_when_every_buffer_non_benign(self):
        cfg = PretrainMLMConfig(benign_exclusion_max_ratio=1.0)
        builder = MLMCorpusBuilder(cfg)
        builder.add(b"\x00" * 128)
        builder.add(b"garbage!")
        with pytest.raises(BenignCorpusError) as exc_info:
            builder.finalise()
        assert "every buffer" in str(exc_info.value)

    def test_build_mlm_corpus_helper(self):
        dex, _ = build_minimal_dex()
        examples, stats = build_mlm_corpus(
            [dex, dex, dex],
            PretrainMLMConfig(benign_exclusion_max_ratio=0.5),
        )
        assert len(examples) == 3
        assert stats.kept == 3
        assert stats.exclusion_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MLMCollator — 80/10/10 masking + label plumbing
# ---------------------------------------------------------------------------


class TestMLMCollator:
    def _corpus(self, n: int = 4, max_len: int = 512):
        dex, _ = build_minimal_dex()
        cfg = PretrainMLMConfig(
            max_seq_length=max_len,
            benign_exclusion_max_ratio=0.0,
            seed=42,
        )
        tok = ByteTokenizer(max_length=cfg.max_seq_length)
        builder = MLMCorpusBuilder(cfg, tokenizer=tok)
        for _ in range(n):
            builder.add(dex)
        examples, _ = builder.finalise()
        return cfg, tok, examples

    def test_collate_shapes_match_input(self):
        cfg, tok, examples = self._corpus()
        collator = MLMCollator(cfg, tokenizer=tok)
        batch = collator.collate(examples)
        assert len(batch.input_ids) == len(examples)
        assert all(len(row) == cfg.max_seq_length for row in batch.input_ids)
        assert all(len(row) == cfg.max_seq_length for row in batch.labels)
        assert all(len(row) == cfg.max_seq_length for row in batch.item_type_labels)

    def test_supervised_positions_only_on_body_tokens(self):
        cfg, tok, examples = self._corpus()
        collator = MLMCollator(cfg, tokenizer=tok)
        batch = collator.collate(examples)
        for row_input, row_labels, row_mask in zip(
            batch.input_ids, batch.labels, batch.attention_mask
        ):
            for tok_id, lbl, msk in zip(row_input, row_labels, row_mask):
                if msk == 0:
                    assert lbl == -100, "PAD must not be supervised"
                if lbl != -100:
                    # Supervised positions must be body bytes or a MASK
                    # produced from one (never BOS/EOS).
                    assert tok_id != tok.BOS_ID
                    assert tok_id != tok.EOS_ID
                    assert msk == 1

    def test_mask_ratio_roughly_matches_config(self):
        cfg, tok, examples = self._corpus(n=8, max_len=1024)
        collator = MLMCollator(cfg, tokenizer=tok)
        batch = collator.collate(examples)

        n_supervised = sum(
            1 for row in batch.labels for v in row if v != -100
        )
        n_body = 0
        for row_input, row_mask in zip(batch.input_ids, batch.attention_mask):
            for tok_id, msk in zip(row_input, row_mask):
                if msk == 1 and tok_id not in tok.SPECIAL_IDS:
                    n_body += 1
        # Note: after masking some body positions carry MASK_ID — they
        # were body originally. Count them too.
        n_mask_in_batch = sum(
            1 for row in batch.input_ids for v in row if v == tok.MASK_ID
        )
        assert n_body + n_mask_in_batch > 0
        observed = n_supervised / max(1, n_body + n_mask_in_batch)
        # Generous tolerance around 0.15; seed is fixed so this test is
        # deterministic but we stay inclusive to ~0.08..0.22.
        assert 0.08 <= observed <= 0.22

    def test_collate_is_deterministic_with_seed(self):
        cfg, tok, examples = self._corpus()
        batch_a = MLMCollator(cfg, tokenizer=tok).collate(examples)
        batch_b = MLMCollator(cfg, tokenizer=tok).collate(examples)
        assert batch_a.input_ids == batch_b.input_ids
        assert batch_a.labels == batch_b.labels

    def test_collate_rejects_empty_batch(self):
        cfg = PretrainMLMConfig(benign_exclusion_max_ratio=0.5)
        with pytest.raises(ValueError):
            MLMCollator(cfg).collate([])


# ---------------------------------------------------------------------------
# compute_pretrain_loss — torch contract
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch", reason="requires [dl] extra")


class TestComputePretrainLoss:
    def _make_tensors(self):
        B, L, V, T = 2, 16, 261, len(DEX_ITEM_TYPES)
        mlm_logits = torch.randn(B, L, V, requires_grad=True)
        item_logits = torch.randn(B, L, T, requires_grad=True)
        mlm_labels = torch.full((B, L), -100, dtype=torch.long)
        item_labels = torch.full((B, L), -100, dtype=torch.long)
        # Supervise a handful of positions on both sides.
        mlm_labels[0, 3] = 42
        mlm_labels[1, 7] = 5
        item_labels[0, 3] = 0  # header
        item_labels[1, 7] = 3  # proto_ids
        return mlm_logits, item_logits, mlm_labels, item_labels

    def test_combined_loss_backprops_through_both_heads(self):
        mlm_logits, item_logits, mlm_labels, item_labels = self._make_tensors()
        loss = compute_pretrain_loss(
            mlm_logits,
            item_logits,
            mlm_labels,
            item_labels,
            item_type_aux_weight=0.2,
        )
        loss.backward()
        assert mlm_logits.grad is not None
        assert item_logits.grad is not None
        assert torch.any(mlm_logits.grad != 0)
        assert torch.any(item_logits.grad != 0)

    def test_aux_weight_zero_skips_aux_branch(self):
        mlm_logits, item_logits, mlm_labels, item_labels = self._make_tensors()
        loss = compute_pretrain_loss(
            mlm_logits,
            item_logits,
            mlm_labels,
            item_labels,
            item_type_aux_weight=0.0,
        )
        loss.backward()
        assert mlm_logits.grad is not None
        # Ablation contract: aux head must NOT receive grad when weight=0.
        assert item_logits.grad is None

    def test_rejects_negative_aux_weight(self):
        mlm_logits, item_logits, mlm_labels, item_labels = self._make_tensors()
        with pytest.raises(ValueError):
            compute_pretrain_loss(
                mlm_logits,
                item_logits,
                mlm_labels,
                item_labels,
                item_type_aux_weight=-0.1,
            )
