from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest
from findalm.pretrain.models import from_pretrained_with_modelforpretraining


@pytest.mark.parametrize(
    "model_type,model_name_or_path,expectation",
    [
        (
            "deberta-v2",
            "hf-internal-testing/tiny-random-DebertaForMaskedLM",
            does_not_raise(),
        ),
        ("llama-2", "HuggingFaceM4/tiny-random-LlamaForCausalLM", does_not_raise()),
        ("roberta", "FacebookAI/roberta-base", pytest.raises(ValueError)),
    ],
)
def test_from_pretrained_with_modelforpretraining(
    model_type: str, model_name_or_path: str, expectation: AbstractContextManager
) -> None:
    with expectation:
        _ = from_pretrained_with_modelforpretraining(model_type, model_name_or_path)
