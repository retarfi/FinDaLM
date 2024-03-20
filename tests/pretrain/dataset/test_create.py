import collections
import os
import sys
from pathlib import Path
from typing import Optional, OrderedDict
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from findalm.llama2 import add_pad_token
from findalm.pretrain.dataset.create import (
    apply_masking,
    convert_batchencoding_to_dict_and_pad,
    convert_sentence_to_ids,
    create_examples_from_batch,
    create_examples_from_document,
    main,
    sentence_to_ids,
)
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

P_THIS_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))
P_TEST_ROOT_DIR: Path = P_THIS_DIR.parent.parent
MAP_TEST_MODELS: dict[str, str] = {
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaForMaskedLM",
    "llama-2": "HuggingFaceM4/tiny-random-LlamaForCausalLM",
}
TOKENIZERS: dict[str, PreTrainedTokenizer] = dict(
    map(lambda x: (x[0], AutoTokenizer.from_pretrained(x[1])), MAP_TEST_MODELS.items())
)
add_pad_token(TOKENIZERS["llama-2"])


@pytest.fixture(name="tokenizer_and_sent_to_ids")
def fixture_tokenizer_and_sent_to_ids() -> tuple[PreTrainedTokenizer, OrderedDict]:
    tokenizer: PreTrainedTokenizer = TOKENIZERS["llama-2"]
    ordct: OrderedDict[str, list[int]] = collections.OrderedDict(
        [
            ("I am a cat.", [306, 626, 263, 6635, 29889]),
            (
                "I have, as yet, no name.",
                [306, 505, 29892, 408, 3447, 29892, 694, 1024, 29889],
            ),
        ]
    )
    return tokenizer, ordct


def test_sentence_to_ids(
    tokenizer_and_sent_to_ids: tuple[PreTrainedTokenizer, OrderedDict[str, list[int]]]
) -> None:
    tokenizer, sent_to_ids = tokenizer_and_sent_to_ids
    for text, tokens in sent_to_ids.items():
        assert sentence_to_ids({"text": [text]}, tokenizer) == {"tokens": [tokens]}


@pytest.mark.parametrize("num_proc", [None, 2])
def test_convert_sentence_to_ids(
    tokenizer_and_sent_to_ids: tuple[PreTrainedTokenizer, OrderedDict[str, list[int]]],
    num_proc: Optional[int],
) -> None:
    tokenizer, sent_to_ids = tokenizer_and_sent_to_ids
    ds: Dataset = Dataset.from_dict({"text": list(sent_to_ids.keys())})
    ds = convert_sentence_to_ids(ds, tokenizer, num_proc=num_proc)
    assert ds["tokens"] == list(sent_to_ids.values())


@pytest.mark.parametrize(
    "tokenizer,input_ids",
    [
        (
            TOKENIZERS["deberta-v2"],
            [[1, 7, 8, 9, 2], [1, 10, 11, 12, 2], [1, 13, 14, 15, 2]],
        ),
        (TOKENIZERS["llama-2"], [[1, 7, 8, 9, 10], [1, 11, 12, 13, 14]]),
    ],
)
def test_create_examples_from_batch(
    tokenizer: PreTrainedTokenizer, input_ids: list[list[int]]
) -> None:
    batch: dict[str, list[list[int]]] = {
        "tokens": [[7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17]]
    }
    expected: dict[str, list[list[int]]] = {"input_ids": input_ids}
    examples: dict[str, list[list[int]]] = create_examples_from_batch(
        batch=batch, tokenizer=tokenizer, max_length=5
    )
    assert examples == expected


@pytest.mark.parametrize(
    "tokenizer,expected",
    [
        (
            TOKENIZERS["deberta-v2"],
            [[1, 7, 8, 9, 2], [1, 10, 11, 12, 2], [1, 13, 14, 15, 2]],
        ),
        (TOKENIZERS["llama-2"], [[1, 7, 8, 9, 10], [1, 11, 12, 13, 14]]),
    ],
)
def test_create_examples_from_document(
    tokenizer: PreTrainedTokenizer, expected: list[list[int]]
) -> None:
    # num_proc: None
    ds: Dataset = Dataset.from_dict(
        {"tokens": [[7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17]]}
    )
    ds = create_examples_from_document(
        ds=ds, tokenizer=tokenizer, max_length=5, num_proc=None
    )
    assert ds["input_ids"] == expected

    # check for-else
    ds: Dataset = Dataset.from_dict(
        {"tokens": [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16]]}
    )
    ds = create_examples_from_document(
        ds=ds, tokenizer=tokenizer, max_length=5, num_proc=None
    )
    assert ds["input_ids"] == expected

    # num_proc: 2
    ds: Dataset = Dataset.from_dict(
        {"tokens": [[7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17]] * 5000}
    )
    create_examples_from_document(ds=ds, tokenizer=tokenizer, max_length=5, num_proc=2)


@pytest.mark.parametrize(
    "tokenizer,expected",
    [
        (
            TOKENIZERS["deberta-v2"],
            {"input_ids": [10, 11, 12, 0, 0], "attention_mask": [1, 1, 1, 0, 0]},
        ),
        (
            TOKENIZERS["llama-2"],
            {
                "input_ids": [32000, 32000, 10, 11, 12],
                "attention_mask": [0, 0, 1, 1, 1],
            },
        ),
    ],
)
def test_convert_batchencoding_to_dict(
    tokenizer: PreTrainedTokenizer, expected: dict[str, int]
) -> None:
    batch: BatchEncoding = BatchEncoding(
        data={
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, -100]]),
        }
    )
    dct: dict[str, list[int]] = convert_batchencoding_to_dict_and_pad(
        batch=batch, tokenizer=tokenizer, max_length=5
    )
    assert dct["input_ids"][0] == expected["input_ids"]
    assert dct["attention_mask"][0] == expected["attention_mask"]
    assert dct["labels"][0] == [-100, -100, -100, -100, -100]


@pytest.mark.parametrize(
    "cache_file_name,num_proc",
    [
        (None, None),
        (str(P_TEST_ROOT_DIR / "data" / "materials" / "cache.arrow"), None),
        (str(P_TEST_ROOT_DIR / "data" / "materials" / "cache.arrow"), 2),
    ],
)
def test_apply_masking(cache_file_name: Optional[str], num_proc: Optional[int]) -> None:
    tokenizer: PreTrainedTokenizer = TOKENIZERS["deberta-v2"]
    max_length: int = 12
    ds: Dataset = Dataset.from_dict({"input_ids": [[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]]})
    ds = apply_masking(
        ds=ds,
        max_length=max_length,
        tokenizer=tokenizer,
        cache_file_name=cache_file_name,
        num_proc=num_proc,
    )
    pad_id: int = tokenizer.pad_token_id
    gold_ids: list[int] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, pad_id, pad_id]
    for i in range(max_length):
        if ds[0]["input_ids"][i] == tokenizer.mask_token_id:
            assert ds[0]["labels"][i] == gold_ids[i], ds[0]["labels"]
    assert ds[0]["attention_mask"] == [1] * 10 + [0, 0]


@pytest.mark.parametrize(
    "model_type,do_mask,cache_file_name,num_proc",
    [
        ("llama-2", False, None, None),
        ("deberta-v2", True, None, None),
        (
            "deberta-v2",
            False,
            str(P_TEST_ROOT_DIR / "data" / "materials" / "cache.arrow"),
            None,
        ),
        (
            "deberta-v2",
            True,
            str(P_TEST_ROOT_DIR / "data" / "materials" / "cache.arrow"),
            4,
        ),
    ],
)
def test_main(
    model_type: str,
    do_mask: bool,
    cache_file_name: Optional[str],
    num_proc: Optional[int],
):
    output_dir: str = str(
        P_TEST_ROOT_DIR / "data" / "materials" / "datasets" / model_type
    )
    if do_mask:
        output_dir += "_mask"
    args_fixed: list[str] = [
        "test_create.py",
        "--input_datasets",
        str(P_TEST_ROOT_DIR / "data" / "datasets" / "botchan"),
        "--output_dir",
        output_dir,
        "--max_length",
        "64",
    ]
    args_fixed.extend(["--pretrained_model_name_or_dir", MAP_TEST_MODELS[model_type]])
    if do_mask:
        args_fixed.append("--do_mask")
    if cache_file_name:
        args_fixed.extend(["--cache_file_name", cache_file_name])
    if num_proc:
        args_fixed.extend(["--num_proc", str(num_proc)])
    with patch.object(sys, "argv", args_fixed):
        main()
