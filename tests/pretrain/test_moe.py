import math
from typing import Optional

import pytest
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from findalm.models.moe.deberta_v2 import (
    DebertaV2MoEForMaskedLM,
    load_pretrained_deberta_v2_into_moe,
)
from findalm.models.moe.llama import LlamaMoEForCausalLM, load_pretrained_llama_into_moe
from findalm.pretrain.moe import (
    freeze_except_mlp,
    freeze_except_router,
    get_trainable_million_params,
)

from .. import MAP_TEST_MODELS


def test_get_trainable_million_params() -> None:
    model = AutoModelForMaskedLM.from_pretrained(MAP_TEST_MODELS["deberta-v2"])
    assert math.isclose(get_trainable_million_params(model), 4.28, abs_tol=1e-2)


@pytest.mark.parametrize(
    "model_type,exclude_mlm_head, num_params,front_frozen_layers",
    [
        ("deberta-v2", True, 0.14, 0),
        ("deberta-v2", False, 0.01, 0),
        ("llama", False, None, 0),
        ("llama", False, None, 2),
    ],
)
def test_freeze_except_mlp(
    model_type: str,
    exclude_mlm_head: bool,
    num_params: Optional[float],
    front_frozen_layers: int,
) -> None:
    if model_type == "deberta-v2":
        model = AutoModelForMaskedLM.from_pretrained(MAP_TEST_MODELS[model_type])
    elif model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(MAP_TEST_MODELS[model_type])
    freeze_except_mlp(model, exclude_mlm_head, front_frozen_layers)
    if num_params is not None:
        assert math.isclose(
            get_trainable_million_params(model), num_params, abs_tol=1e-2
        )


@pytest.mark.parametrize(
    "model_type,exclude_mlm_head, num_params,front_frozen_layers",
    [
        ("deberta-v2", True, 0.129601, 0),
        ("deberta-v2", False, 0.00048, 0),
        ("llama", False, None, 0),
        ("llama", False, None, 1),
    ],
)
def test_freeze_except_router(
    model_type: str,
    exclude_mlm_head: bool,
    num_params: Optional[float],
    front_frozen_layers: int,
) -> None:
    if model_type == "deberta-v2":
        model = load_pretrained_deberta_v2_into_moe(
            DebertaV2MoEForMaskedLM,
            "dense",
            model_names=[MAP_TEST_MODELS[model_type]] * 3,
        )
    elif model_type == "llama":
        model = load_pretrained_llama_into_moe(
            LlamaMoEForCausalLM, "dense", model_names=[MAP_TEST_MODELS[model_type]] * 3
        )
    freeze_except_router(model, exclude_mlm_head, front_frozen_layers)
    if num_params is not None:
        assert math.isclose(
            get_trainable_million_params(model), num_params, abs_tol=1e-6
        )
