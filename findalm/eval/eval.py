import argparse
import datetime
import itertools
import os
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import mlflow
import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import PrinterCallback

from ..models.moe.deberta_v2 import (
    DebertaV2MoEForSequenceClassification,
    DebertaV2MoEForTokenClassification,
)
from ..models.tokenizer import load_tokenizer
from ..pretrain.moe import freeze_except_cls
from .base import NER_LABELS, load_compute_metrics
from .tasks import finerord, fomc, fpb, headline

transformers.logging.set_verbosity(transformers.logging.log_levels["error"])
TASKS: tuple[str] = (
    "finerord",
    "fiqasa",
    "fomc",
    "fpb",
    *(f"headline-{x}" for x in headline.MAP_SUBTASK_COLUMNS.keys()),
    "ner",
)
ROOT_DIR: Path = Path(__file__).parents[2]


class HyperParams(NamedTuple):
    max_epoch: int = 10
    real_batch_size: int = 32
    lr: float = 1e-4


class GridHyperParams(NamedTuple):
    max_epoch: tuple[int] = (10,)
    real_batch_size: tuple[int] = (16, 32)
    lr: tuple[float] = (1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4)


def print_with_datetime(obj: Any) -> None:
    print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), obj)


def load_datasetdict(task: str) -> DatasetDict:
    main_task: str = task.split("-")[0]
    sub_task: Optional[str] = None
    if "-" in task:
        sub_task = task.split("-")[1]
    if main_task == "finerord":
        dsd = finerord.load_datasetdict()
    elif main_task == "fiqasa":
        dsd = load_dataset("ChanceFocus/flare-fiqasa")
        dsd = dsd.rename_column("gold", "labels")
    elif main_task == "fomc":
        dsd = fomc.load_datasetdict()
    elif main_task == "fpb":
        dsd = fpb.load_datasetdict()
    elif main_task == "headline":
        dsd = headline.load_datasetdict(sub_task, seed=42)
    elif main_task == "ner":
        dsd = load_dataset("ChanceFocus/flare-ner")
        dsd = dsd.rename_column("label", "bio_tag")
    else:  # pragma: no cover
        raise NotImplementedError(f"Task {main_task} is not implemented")
    return dsd


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    transformers.trainer_utils.set_seed(seed=seed)


def postprocess(
    log_history: list[dict[str, Union[int, float]]],
    eval_metrics: list[str],
    max_epoch: int,
) -> dict[str, float]:
    lst_epochs_result: list[dict[str, Union[int, float]]] = list(
        filter(lambda d: len(set(eval_metrics) & set(d.keys())) > 0, log_history)
    )
    lst_epochs_result_ordered: list[dict[str, float]] = []
    for i in range(1, max_epoch + 1):
        d_tmp = {}
        for d in filter(lambda item: item["epoch"] == i, lst_epochs_result):
            d_tmp.update(d)
        lst_epochs_result_ordered.append(d_tmp)
    best_result: dict[str, float] = max(
        lst_epochs_result_ordered, key=lambda x: x[eval_metrics[0]]
    )
    best_result = dict(
        filter(lambda item: item[0] in eval_metrics, best_result.items())
    )
    return best_result


def train(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    model_name_or_dir: str,
    metric_name: str,
    dsd: DatasetDict,
    params_max_epoch: list[int],
    params_real_batch_size: list[int],
    params_lr: list[float],
    config: Optional[PretrainedConfig] = None,
    accumulation_steps: int = 1,
    seed: int = 0,
    bf16: bool = False,
    tune_only_cls: bool = True,
    disable_print_epoch: bool = False,
) -> dict[str, float]:
    # [key score, metrics(dict)]
    grid_results: list[dict[str, float]] = []
    eval_metrics: list[str] = [x + metric_name for x in ["eval_valid_", "eval_test_"]]
    num_gpu: int = max(1, torch.cuda.device_count())
    product = itertools.product(params_max_epoch, params_real_batch_size, params_lr)
    for max_epoch, real_batch_size, lr in product:
        training_args = transformers.TrainingArguments(
            output_dir="./outputs/",
            do_train=True,
            do_eval=True,
            do_predict=False,  # "test" in raw_datasets,
            evaluation_strategy="epoch",
            per_device_train_batch_size=real_batch_size
            // (accumulation_steps * num_gpu),
            per_device_eval_batch_size=real_batch_size
            // (accumulation_steps * num_gpu),
            gradient_accumulation_steps=accumulation_steps,
            eval_accumulation_steps=accumulation_steps,
            learning_rate=lr,
            num_train_epochs=max_epoch,
            warmup_ratio=0.05,
            logging_strategy="no",
            log_on_each_node=False,
            save_strategy="no",
            seed=seed,
            report_to="none",
            label_names=None,
            disable_tqdm=True,
            bf16=bf16,
            # data_seed=params["seed"],
        )

        main_task: str = task.split("-")[0]
        if main_task in ("fiqasa", "fomc", "fpb", "headline"):
            # Sentence Classification Task
            if config.architectures == ["DebertaV2MoEForMaskedLM"]:
                cls = DebertaV2MoEForSequenceClassification

            else:
                cls = AutoModelForSequenceClassification
        elif main_task in ("finerord", "ner"):
            # Token Classification Task
            if config.architectures == ["DebertaV2MoEForMaskedLM"]:
                cls = DebertaV2MoEForTokenClassification
            else:
                cls = AutoModelForTokenClassification
        else:  # pragma: no cover
            raise NotImplementedError(f"Preprocess for {main_task} is not implemented")
        model: PreTrainedModel = cls.from_pretrained(model_name_or_dir, config=config)
        if tune_only_cls:
            freeze_except_cls(model)

        # train
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=dsd["train"] if training_args.do_train else None,
            eval_dataset={"valid": dsd["valid"], "test": dsd["test"]},
            compute_metrics=load_compute_metrics(metric_name),
            tokenizer=tokenizer,
            data_collator=transformers.default_data_collator,
        )
        if disable_print_epoch:
            trainer.remove_callback(PrinterCallback)
        trainer.train()

        # Postprocess
        result: dict[str, float] = postprocess(
            log_history=trainer.state.log_history,
            eval_metrics=eval_metrics,
            max_epoch=max_epoch,
        )
        grid_results.append(result)
    # get best eval result
    best_result: dict[str, float] = sorted(
        grid_results, key=lambda x: x[eval_metrics[0]], reverse=True
    )[0]
    best_result = dict(
        [
            (k.replace("eval_valid_", "valid_").replace("eval_test_", "test_"), v)
            for k, v in best_result.items()
        ]
    )
    return best_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument(
        "-m",
        "--model_name_or_dir",
        type=str,
        required=True,
        help="model name or model directory",
    )
    parser.add_argument("-a", "--accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--max_length",
        type=int,
        default=-1,
        help="If not specified, use max_position_embeddings of model",
    )
    parser.add_argument("--mlflow_run_name")
    parser.add_argument("--do_grid", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tune_only_cls", action="store_true")
    parser.add_argument("--disable_print_epoch", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    model_name_or_dir: str = args.model_name_or_dir
    seeds: list[int] = args.seed

    # Load Dataset
    dsd: DatasetDict = load_datasetdict(args.task)
    main_task: str = args.task.split("-")[0]
    metric_name: str
    num_labels: int
    if main_task == "finerord":
        metric_name = "f1-macro"
        num_labels = 7
    elif main_task == "fiqasa":
        metric_name = "f1-macro"
        num_labels = 3
    elif main_task == "fomc":
        metric_name = "f1-macro"
        num_labels = 3
    elif main_task == "fpb":
        metric_name = "f1-macro"
        num_labels = 3
    elif main_task == "headline":
        metric_name = "f1-micro"
        num_labels = 2
    elif main_task == "ner":
        metric_name = "f1-macro"
        num_labels = len(NER_LABELS)
    else:  # pragma: no cover
        raise NotImplementedError(f"Task {main_task} is not implemented")

    config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_dir)
    is_llama: bool = "llama" in model_name_or_dir or (
        config.architectures is not None and "Llama" in config.architectures[0]
    )
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(
        model_name_or_dir, is_llama=is_llama
    )
    model_max_length: int = config.max_position_embeddings
    assert config.model_type in ("deberta-v2", "roberta")

    # Preprocess
    config: Optional[PretrainedConfig] = None
    if main_task in ("fomc", "fiqasa", "fpb", "headline"):

        def _tokenization(example: dict[str, Any], max_length: int):
            return tokenizer(
                example["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        # Sentence Classification Task
        dataset_max_length: int = max(
            map(lambda x: len(tokenizer.encode(x)), dsd["train"]["text"])
        )
        max_length: int = min(model_max_length, dataset_max_length)
        dsd = dsd.map(lambda example: _tokenization(example, max_length))
        dsd = dsd.select_columns(["input_ids", "attention_mask", "labels"])
        config = AutoConfig.from_pretrained(model_name_or_dir)
        config.num_labels = num_labels
    elif main_task in ("finerord", "ner"):

        def _tokenization(example: dict[str, Any], max_length: int):
            max_num_tokens: int = max_length - tokenizer.num_special_tokens_to_add()
            example["input_ids"] = []
            converted_labels: list[int] = []
            for i, word in enumerate(example["text"].split(" ")):
                token_ids: list[int] = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1:
                    converted_labels.append(NER_LABELS.index(example["bio_tag"][i]))
                elif len(token_ids) == 0:
                    raise ValueError()
                else:
                    label: str = example["bio_tag"][i]
                    tags_to_add: list[str] = [label.replace("B-", "I-")] * len(
                        token_ids
                    )
                    if label.startswith("B-"):
                        tags_to_add[0] = label
                    converted_labels.extend([NER_LABELS.index(x) for x in tags_to_add])
                example["input_ids"].extend(token_ids)
            # padding or truncation
            if len(example["input_ids"]) < max_num_tokens:
                # pad
                num_add_tokens: int = max_num_tokens - len(example["input_ids"])
                example["attention_mask"] = [1] * (
                    len(example["input_ids"]) + tokenizer.num_special_tokens_to_add()
                ) + [0] * num_add_tokens
                assert tokenizer.num_special_tokens_to_add() == 2
                example["labels"] = (
                    [NER_LABELS.index("O")]
                    + converted_labels
                    + [NER_LABELS.index("O")] * (num_add_tokens + 1)
                )
                example["input_ids"] = (
                    tokenizer.build_inputs_with_special_tokens(example["input_ids"])
                    + [tokenizer.pad_token_id] * num_add_tokens
                )
            elif len(example["input_ids"]) >= max_num_tokens:
                # truncate
                example["input_ids"] = tokenizer.build_inputs_with_special_tokens(
                    example["input_ids"][:max_num_tokens]
                )
                assert tokenizer.num_special_tokens_to_add() == 2
                example["labels"] = (
                    [NER_LABELS.index("O")]
                    + converted_labels[:max_num_tokens]
                    + [NER_LABELS.index("O")]
                )
                example["attention_mask"] = [1] * len(example["input_ids"])
            else:  # pragma: no cover
                raise ValueError()
            return example

        dataset_max_length: int = max(
            map(
                lambda x: sum(
                    map(
                        lambda y: len(tokenizer.encode(y, add_special_tokens=False)),
                        x.split(" "),
                    )
                )
                + tokenizer.num_special_tokens_to_add(pair=False),
                dsd["train"]["text"],
            )
        )
        max_length: int = min(model_max_length, dataset_max_length)
        dsd = dsd.map(lambda example: _tokenization(example, max_length))
        dsd = dsd.select_columns(["input_ids", "attention_mask", "labels"])
        config = AutoConfig.from_pretrained(model_name_or_dir)
        config.num_labels = num_labels
    else:  # pragma: no cover
        raise NotImplementedError(f"Preprocess for {main_task} is not implemented")

    params_max_epoch: tuple[int]
    params_real_batch_size: tuple[int]
    params_lr: tuple[float]
    if args.do_grid:
        ghp = GridHyperParams()
        params_max_epoch = ghp.max_epoch
        params_real_batch_size = ghp.real_batch_size
        params_lr = ghp.lr
    else:
        hp = HyperParams()
        params_max_epoch = (hp.max_epoch,)
        params_real_batch_size = (hp.real_batch_size,)
        params_lr = (hp.lr,)

    if args.mlflow_run_name is not None:
        mlflow.set_tracking_uri(str(ROOT_DIR / "mlruns"))
        mlflow.set_experiment(args.task)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.set_tags({"task": main_task, "machine": os.uname()[1], "seed": seeds})
        mlflow.log_params(
            {
                "max_epoch": params_max_epoch,
                "batch_size": params_real_batch_size,
                "lr": params_lr,
            }
        )
    lst_results: list[dict[str, float]] = []
    for seed in seeds:
        print_with_datetime(f"Seed {seed}")
        set_seed(seed)
        result: dict[str, float] = train(
            task=args.task,
            tokenizer=tokenizer,
            model_name_or_dir=model_name_or_dir,
            metric_name=metric_name,
            dsd=dsd,
            params_max_epoch=params_max_epoch,
            params_real_batch_size=params_real_batch_size,
            params_lr=params_lr,
            config=config,
            accumulation_steps=args.accumulation_steps,
            seed=seed,
            bf16=args.bf16,
            tune_only_cls=args.tune_only_cls,
            disable_print_epoch=args.disable_print_epoch,
        )
        print_with_datetime(f"Result of seed {seed}: {result}")
        lst_results.append(result)
    dct_results: dict[str, list[float]] = {
        metric: [r[metric] for r in lst_results] for metric in lst_results[0].keys()
    }
    ave_results: dict[str, float] = {k: np.mean(v) for k, v in dct_results.items()}
    print_with_datetime(f"Overall results: {ave_results}")
    if args.mlflow_run_name is not None:
        mlflow.log_metrics(ave_results)
        mlflow.set_tags({f"all {k}s": tuple(v) for k, v in dct_results.items()})
        mlflow.end_run()


if __name__ == "__main__":
    main()
