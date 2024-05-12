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
    "pretrain_mode,model_type,args",
    [
        ("moe-stage1", "deberta-v2", ""),
        ("moe-stage1", "llama", ""),
        ("moe-stage1", "t5", ""),
        ("default", "deberta-v2", ""),  # model,mask
        ("default", "deberta-v2", "--is_dataset_masked"),  # model
        ("default", "deberta-v2", "--is_dataset_masked --do_eval"),  # do_eval
        ("default", "llama", ""),  # model
        ("default", "llama", "--do_eval"),  # do_eval
        (
            "default",
            "llama",
            "--do_eval --prediction_loss_only",
        ),  # do_eval,prediction_loss_only
    ],
)
def test_main(pretrain_mode: str, model_type: str, args: str) -> None:
    postfix_mask: str = "_mask" if "--is_dataset_masked" in args else ""
    dataset_dir: str = os.path.join(
        THIS_DIR, "../data/materials/datasets/", model_type + postfix_mask
    )

    args_fixed: list[str] = [
        "--pretrain_mode",
        pretrain_mode,
        "--model_type",
        model_type,
        "--pretrained_model_name_or_dir",
        MAP_TEST_MODELS[model_type],
        "--dataset_names",
        dataset_dir,
        "--validation_split_size",
        "100",
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
        "--report_to",
        "none",
    ]
    if args:
        args_fixed.extend(args.split(" "))
    with patch.object(sys, "argv", ["test_pretrain.py"] + args_fixed):
        main()


@pytest.mark.parametrize(
    "pattern, expectation",
    [("A", does_not_raise()), ("B", does_not_raise())],
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
        "--pretrain_mode",
        "default",
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
        "--report_to",
        "none",
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
    with expectation:
        with patch.object(sys, "argv", ["test_pretrain.py"] + args_fixed):
            main()


@pytest.mark.parametrize(
    "model_type,dataset_names,moe_type",
    [
        ("llama", "finerord", "top2-skip"),
        ("deberta-v2", "fiqasa", "top2"),
        ("t5", "fomc", "top1"),
        ("llama", "ner", "top1"),
        ("deberta-v2", "headline-price", "dense"),
    ],
)
def test_main_moe_stage2(model_type: str, dataset_names: str, moe_type: str) -> None:
    args_fixed: list[str] = [
        "--pretrain_mode",
        "moe-stage2",
        "--model_type",
        model_type,
        "--pretrained_model_name_or_dir",
        *[MAP_TEST_MODELS[model_type]] * 2,
        "--moe_type",
        moe_type,
        "--dataset_names",
        dataset_names,
        "--output_dir",
        os.path.join(THIS_DIR, "../data/materials/model", f"moe-{model_type}"),
        "--overwrite_output_dir",
        "--do_train",
        "--max_steps",
        "1",
        "--save_steps",
        "5",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--bf16",
        "--report_to",
        "none",
    ]
    with patch.object(sys, "argv", ["test_pretrain.py"] + args_fixed):
        main()
