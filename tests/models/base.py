from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest
from findalm.models.base import from_pretrained_with_modelforpretraining

from .. import MAP_TEST_MODELS


@pytest.mark.parametrize(
    "model_type,model_name_or_path,expectation",
    [
        (model, model_path, does_not_raise())
        for model, model_path in MAP_TEST_MODELS.items()
    ]
    + [
        (
            "electra",
            "hf-internal-testing/tiny-random-ElectraForMaskedLM",
            pytest.raises(ValueError),
        )
    ],
)
def test_from_pretrained_with_modelforpretraining(
    model_type: str, model_name_or_path: str, expectation: AbstractContextManager
) -> None:
    with expectation:
        _ = from_pretrained_with_modelforpretraining(model_type, model_name_or_path)
