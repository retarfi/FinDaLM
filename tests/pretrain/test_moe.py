import math

from transformers import AutoModelForMaskedLM

from findalm.models.moe.deberta_v2 import (
    DebertaV2MoEForMaskedLM,
    load_pretrained_deberta_v2_into_moe,
)
from findalm.pretrain.moe import (
    freeze_except_mlp,
    freeze_except_router,
    get_trainable_million_params,
)

from .. import MAP_TEST_MODELS


def test_get_trainable_million_params() -> None:
    model = AutoModelForMaskedLM.from_pretrained(MAP_TEST_MODELS["deberta-v2"])
    assert math.isclose(get_trainable_million_params(model), 4.28, abs_tol=1e-2)


def test_freeze_except_mlp() -> None:
    model = AutoModelForMaskedLM.from_pretrained(MAP_TEST_MODELS["deberta-v2"])
    freeze_except_mlp(model)
    assert math.isclose(get_trainable_million_params(model), 0.01, abs_tol=1e-2)


def test_freeze_except_router() -> None:
    model = load_pretrained_deberta_v2_into_moe(
        DebertaV2MoEForMaskedLM,
        "dense",
        model_names=[MAP_TEST_MODELS["deberta-v2"]] * 3,
    )
    freeze_except_router(model)
    assert math.isclose(get_trainable_million_params(model), 0.00048, abs_tol=1e-6)
