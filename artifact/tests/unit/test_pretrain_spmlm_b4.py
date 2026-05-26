from __future__ import annotations

import random

import pytest

from android_packer.training.pretrain_spmlm import (
    SpMLMConfig,
    compute_spmlm_loss,
    corrupt_token_sequence,
    create_spmlm_batch,
)


def _sequence(token_type: int = 0) -> dict:
    ids = [1, 5, 6, 7, 8, 9, 2, 0]
    return {
        "token_ids": ids,
        "token_type_ids": [token_type] * len(ids),
        "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0],
        "abnormal_mask": [0] * len(ids),
    }


def test_corrupt_token_sequence_preserves_special_tokens_and_padding() -> None:
    cfg = SpMLMConfig(vocab_size=32)
    rng = random.Random(7)

    corrupted = corrupt_token_sequence(
        [1, 5, 6, 7, 8, 9, 2, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        cfg,
        rng,
    )

    assert corrupted[0] == 1
    assert corrupted[6] == 2
    assert corrupted[7] == 0
    assert corrupted != [1, 5, 6, 7, 8, 9, 2, 0]


def test_spmlm_batch_keeps_clean_inputs_for_normality_aux() -> None:
    cfg = SpMLMConfig(vocab_size=32, mask_prob=0.5)
    batch = create_spmlm_batch([_sequence()], cfg, random.Random(1))

    assert batch.clean_input_ids.tolist() == [[1, 5, 6, 7, 8, 9, 2, 0]]
    assert batch.input_ids.shape == batch.clean_input_ids.shape


torch = pytest.importorskip("torch", reason="requires [dl] extra")


class _TinyPretrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = torch.nn.Module()
        self.bert.token_embed = torch.nn.Embedding(32, 12)
        self.normality = torch.nn.Linear(12, 1)

    def forward_with_mlm(self, token_ids, token_type_ids, attention_mask):
        hidden = self.bert.token_embed(token_ids)
        cls = hidden[:, 0, :]
        return cls, hidden

    def forward_pretrain_normality(self, cls_embedding):
        return self.normality(cls_embedding).squeeze(-1)


def test_spmlm_loss_with_normality_aux_backprops() -> None:
    cfg = SpMLMConfig(
        vocab_size=32,
        mask_prob=0.5,
        normality_loss_weight=0.2,
        corruption_prob=0.5,
        use_fp16=False,
    )
    batch = create_spmlm_batch([_sequence(0), _sequence(2)], cfg, random.Random(2))
    model = _TinyPretrainModel()

    loss = compute_spmlm_loss(
        model,
        batch,
        torch.device("cpu"),
        cfg,
        random.Random(3),
    )
    loss.backward()

    assert loss.item() > 0.0
    assert model.bert.token_embed.weight.grad is not None
    assert model.normality.weight.grad is not None
