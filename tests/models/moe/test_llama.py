import pytest
import torch
from transformers import LlamaConfig

from findalm.models.moe.llama import (
    LlamaMoEDecoderLayer,
    LlamaMoEForCausalLM,
    load_pretrained_llama_into_moe,
)

from ... import MAP_TEST_MODELS


@pytest.mark.parametrize("moe_type", ("top2-skip", "top2", "top1", "dense"))
def test_LlamaMoEDecoderLayer(moe_type: str) -> None:
    config = LlamaConfig.from_pretrained(MAP_TEST_MODELS["llama"])
    config.moe_type = moe_type
    config.num_experts = 13
    seq_len: int = 3
    batch_size: int = 5
    model = LlamaMoEDecoderLayer(config, layer_idx=0)
    _ = model(
        torch.rand((batch_size, seq_len, config.hidden_size)),
        torch.randint(0, 2, (batch_size, config.num_attention_heads, seq_len, seq_len)),
        torch.stack([torch.arange(seq_len) * batch_size]),
    )


@pytest.mark.parametrize("cls", (LlamaMoEForCausalLM,))
@pytest.mark.parametrize("moe_type", ("top2-skip", "top2", "top1", "dense"))
@pytest.mark.parametrize("front_frozen_layers", (0, 2))
def test_load_pretrained_into_moe(cls, moe_type: str, front_frozen_layers: int) -> None:
    _ = load_pretrained_llama_into_moe(
        cls,
        moe_type,
        model_names=[MAP_TEST_MODELS["llama"]] * 3,
        front_frozen_layers=front_frozen_layers,
        torch_dtype=torch.bfloat16,
    )
