import pytest
import torch
from transformers import DebertaV2Config

from findalm.models.moe.deberta_v2 import (
    DebertaV2FFN,
    DebertaV2MoEForMaskedLM,
    DebertaV2MoEForSequenceClassification,
    DebertaV2MoEForTokenClassification,
    DebertaV2MoELayer,
    load_pretrained_deberta_v2_into_moe,
)

from ... import MAP_TEST_MODELS


def test_DebertaV2FFN() -> None:
    config = DebertaV2Config.from_pretrained(MAP_TEST_MODELS["deberta-v2"])
    model = DebertaV2FFN(config)
    _ = model(torch.rand((5, 4, config.hidden_size)))


@pytest.mark.parametrize("moe_type", ("top2-skip", "top2", "top1", "dense"))
def test_DebertaV2MoELayer(moe_type: str) -> None:
    config = DebertaV2Config.from_pretrained(MAP_TEST_MODELS["deberta-v2"])
    config.moe_type = moe_type
    config.num_experts = 13
    seq_len: int = 3
    batch_size: int = 5
    model = DebertaV2MoELayer(config)
    _ = model(
        torch.rand((batch_size, seq_len, config.hidden_size)),
        torch.randint(0, 2, (batch_size, config.num_attention_heads, seq_len, seq_len)),
    )


@pytest.mark.parametrize(
    "cls",
    (
        DebertaV2MoEForMaskedLM,
        DebertaV2MoEForSequenceClassification,
        DebertaV2MoEForTokenClassification,
    ),
)
@pytest.mark.parametrize("moe_type", ("top2-skip", "top2", "top1", "dense"))
def test_load_pretrained_into_moe(cls, moe_type: str) -> None:
    _ = load_pretrained_deberta_v2_into_moe(
        cls,
        moe_type,
        model_names=[MAP_TEST_MODELS["deberta-v2"]] * 3,
        torch_dtype=torch.bfloat16,
    )
