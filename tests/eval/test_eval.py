import os
import sys
from unittest.mock import patch

import pytest

from findalm.eval.eval import main

from .. import MAP_TEST_MODELS

os.environ["WANDB_MODE"] = "offline"


@pytest.mark.parametrize(
    "task, model_type",
    [
        ("finerord", "deberta-v2"),
        ("fiqasa", "deberta-v2"),
        ("fomc", "deberta-v2"),
        ("fpb", "deberta-v2"),
        ("headline-price", "deberta-v2"),
        ("ner", "deberta-v2"),
    ],
)
def test_main(task: str, model_type: str):
    args: list[str] = [
        __file__,
        task,
        "-m",
        MAP_TEST_MODELS[model_type],
        "--max_length",
        "16",
    ]
    with patch.object(sys, "argv", args):
        main()
