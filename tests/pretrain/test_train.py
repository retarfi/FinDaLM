import os
import sys
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest
from findalm.pretrain.train import main

from .. import MAP_TEST_MODELS

os.environ["WANDB_MODE"] = "offline"
THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
MAX_LENGTH: int = 64


@pytest.mark.parametrize(
    "model_type,args",
    [
        ("deberta-v2", ""),  # model,mask
        ("deberta-v2", "--is_dataset_masked"),  # model
        ("deberta-v2", "--is_dataset_masked --do_eval"),  # do_eval
        ("llama", ""),  # model
        ("llama", "--do_eval"),  # do_eval
        ("llama", "--do_eval --prediction_loss_only"),  # do_eval,prediction_loss_only
        ("roberta", ""),
        ("t5", ""),
    ],
)
def test_main(model_type: str, args: str) -> None:
    postfix_mask: str = "_mask" if "--is_dataset_masked" in args else ""
    dataset_dir: str = os.path.join(
        THIS_DIR, "../data/materials/datasets/", model_type + postfix_mask
    )

    args_fixed: list[str] = [
        "--model_type",
        model_type,
        "--pretrained_model_name_or_dir",
        MAP_TEST_MODELS[model_type],
        "--dataset_names",
        dataset_dir,
        "--output_dir",
        os.path.join(THIS_DIR, "../data/materials/model", model_type),
        "--overwrite_output_dir",
        "--do_train",
        "--max_steps",
        "5",
        "--save_steps",
        "5",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--bf16",
    ]
    with patch.object(sys, "argv", ["test_pretrain.py"] + args_fixed):
        main()


@pytest.mark.parametrize(
    "pattern, expectation",
    [
        ("A", does_not_raise()),
        ("B", does_not_raise()),
        ("C", pytest.raises(ValueError)),
    ],
)
def test_resume_from_checkpoint(
    pattern: str, expectation: AbstractContextManager
) -> None:
    model_type: str = "deberta-v2"
    is_dataset_masked: bool = True

    # first, we must create dataset if not exists
    postfix_mask: str = "_mask" if is_dataset_masked else ""
    dataset_dir: str = os.path.join(
        THIS_DIR, "../data/materials/datasets/", model_type + postfix_mask
    )
    # dataset must be created in test_main()

    args_fixed: list[str] = [
        "--model_type",
        model_type,
        "--pretrained_model_name_or_dir",
        MAP_TEST_MODELS[model_type],
        "--dataset_names",
        dataset_dir,
        "--do_train",
        "--max_steps",
        "5",
        "--save_steps",
        "1000",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--bf16",
        "--output_dir",
    ]
    if pattern == "A":
        args_fixed.append(os.path.join(THIS_DIR, "../data/materials/model", model_type))
    elif pattern == "B":
        args_fixed.extend(
            [
                os.path.join(THIS_DIR, "../data/materials/model/", model_type),
                "--resume_from_checkpoint",
                os.path.join(THIS_DIR, "../data/materials/model", model_type),
            ]
        )
    elif pattern == "C":
        args_fixed.append("../data/model/")
    with expectation:
        with patch.object(sys, "argv", ["test_pretrain.py"] + args_fixed):
            main()
