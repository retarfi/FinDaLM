import argparse
import itertools
import logging
from typing import Any, Optional

from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
)

from ... import get_logger
from ...models.llama import add_pad_token
from .load import load_ds

logger: logging.Logger = get_logger()
BATCH_SIZE: int = 4000


def sentence_to_ids(
    example: dict[str, Any], tokenizer: PreTrainedTokenizer
) -> dict[str, list[list[int]]]:
    batch_tokens: list[list[str]] = [tokenizer.tokenize(x) for x in example["text"]]
    token_ids: list[list[int]] = [
        [tokenizer.convert_tokens_to_ids(tk) for tk in tokens if tk]
        for tokens in batch_tokens
    ]
    return {"tokens": token_ids}


def _sentence_to_ids_for_global_tokenizer(
    example: dict[str, Any]
) -> dict[str, list[list[int]]]:
    return sentence_to_ids(example, _tokenizer)


def filter_empty(
    example: dict[str, list[list[list[int]]]]
) -> dict[str, list[list[list[int]]]]:
    return {"tokens": [x for x in example["tokens"] if len(x) > 0]}


def convert_sentence_to_ids(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    cache_file_name: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    if "_tokenizer" not in globals():
        global _tokenizer
    _tokenizer = tokenizer
    ds = ds.map(
        _sentence_to_ids_for_global_tokenizer,
        batched=True,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        cache_file_name=cache_file_name,
        batch_size=BATCH_SIZE,
        writer_batch_size=BATCH_SIZE,
        num_proc=num_proc,
        desc="convert_sentence_to_ids<map>",
    )
    ds = ds.map(
        filter_empty,
        batched=True,
        load_from_cache_file=False,
        cache_file_name=cache_file_name,
        batch_size=BATCH_SIZE,
        writer_batch_size=BATCH_SIZE,
        num_proc=num_proc,
        desc="convert_sentence_to_ids<filter>",
    )
    logger.info("Tokenize finished")
    return ds


def create_examples_from_batch(
    batch: dict[str, list[list[int]]], tokenizer: PreTrainedTokenizer, max_length: int
) -> dict[str, list[list[int]]]:
    max_num_tokens: int = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    current_chunk: list[int] = []  # a buffer stored current working segments
    current_length: int = 0
    input_ids: list[list[int]] = []
    for segment in batch["tokens"]:
        while True:
            if current_length + len(segment) >= max_num_tokens:
                idx: int = max_num_tokens - current_length
                current_chunk.append(segment[:idx])
                current_chunk = list(itertools.chain.from_iterable(current_chunk))
                # add special tokens
                input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(current_chunk)
                )
                current_chunk = []
                current_length = 0
                segment = segment[idx:]
            else:
                current_chunk.append(segment)
                current_length += len(segment)
                break
    current_chunk = list(itertools.chain.from_iterable(current_chunk))
    if len(current_chunk) >= max_num_tokens * 0.8:
        input_ids.append(tokenizer.build_inputs_with_special_tokens(current_chunk))
    return {"input_ids": input_ids}


def _create_examples_from_batch_for_global_tokenizer(
    batch: dict[str, list[int]], max_length: int
) -> dict[str, list[list[int]]]:
    return create_examples_from_batch(batch, _tokenizer, max_length)


def create_examples_from_document(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    cache_file_name: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    if "_tokenizer" not in globals():
        global _tokenizer
    _tokenizer = tokenizer
    ds = ds.map(
        lambda example: _create_examples_from_batch_for_global_tokenizer(
            example, max_length
        ),
        cache_file_name=cache_file_name,
        num_proc=num_proc,
        batched=True,
        batch_size=BATCH_SIZE,
        writer_batch_size=BATCH_SIZE,
        remove_columns=["tokens"],
        load_from_cache_file=False,
        desc="create_examples_from_document",
    )
    return ds


def convert_batchencoding_to_dict_and_pad(
    batch: BatchEncoding, tokenizer: PreTrainedTokenizer, max_length: int
) -> dict[str, list[int]]:
    # pad input_ids, attention_mask
    dct: dict[str, list[int]] = {
        k: v.tolist()[0] if len(v.shape) == 3 else v.tolist() for k, v in batch.items()
    }

    dct = tokenizer.pad(dct, padding="max_length", max_length=max_length).data
    ignore_index: int = -100
    # pad labels
    for i in range(len(dct["labels"])):
        dct["labels"][i] += [ignore_index] * max(0, max_length - len(dct["labels"][i]))
    return dct


def apply_masking(
    ds: Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    cache_file_name: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    ds = ds.map(
        lambda example: convert_batchencoding_to_dict_and_pad(
            batch=data_collator(
                [
                    {k: v[i] for k, v in example.items()}
                    for i in range(len(example["input_ids"]))
                ]
            ),
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        cache_file_name=cache_file_name,
        num_proc=num_proc,
        batched=True,
        batch_size=10000,
        writer_batch_size=20000,
        load_from_cache_file=False,
        desc="apply_masking",
    )
    return ds


def create_dataset(
    input_datasets: list[str],
    output_dir: str,
    pretrained_model_name_or_dir: str,
    max_length: int,
    cache_file_name: Optional[str],
    do_mask: bool = False,
    num_proc: Optional[int] = None,
) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_dir
    )
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ):
        add_pad_token(tokenizer)
    ds: Dataset = concatenate_datasets([load_ds(path) for path in input_datasets])

    logger.info("Convert sentence to ids...")
    ds = convert_sentence_to_ids(
        ds, tokenizer, cache_file_name=cache_file_name, num_proc=num_proc
    )
    logger.info("Create examples from document...")
    ds = create_examples_from_document(
        ds=ds,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_file_name=cache_file_name,
        num_proc=num_proc,
    )

    if do_mask:
        logger.info("Apply masking...")
        ds = apply_masking(
            ds=ds,
            max_length=max_length,
            tokenizer=tokenizer,
            cache_file_name=cache_file_name,
            num_proc=num_proc,
        )

    ds.flatten_indices().save_to_disk(output_dir)
    logger.info(f"Processed dataset saved in {output_dir}")


def main() -> None:
    # arguments
    parser: argparse.Namespace = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # required
    parser.add_argument(
        "--input_datasets",
        nargs="+",
        required=True,
        help="Directory or HF dataset path for input datasets",
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Directory to save"
    )
    parser.add_argument("--pretrained_model_name_or_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)

    # optional
    parser.add_argument(
        "--cache_file_name",
        help="Provide the name of a path for the cache file. "
        "It is used to store the results of the computation "
        "instead of the automatically generated cache file name. "
        "It would be useful when dealing with a huge dataset.",
    )
    parser.add_argument(
        "--do_mask", action="store_true", help="Mask datasets in advance"
    )
    parser.add_argument(
        "--num_proc", type=int, help="Max number of processes when tokenizing"
    )

    args: argparse.Namespace = parser.parse_args()
    create_dataset(
        input_datasets=args.input_datasets,
        output_dir=args.output_dir,
        pretrained_model_name_or_dir=args.pretrained_model_name_or_dir,
        max_length=args.max_length,
        cache_file_name=args.cache_file_name,
        do_mask=args.do_mask,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
