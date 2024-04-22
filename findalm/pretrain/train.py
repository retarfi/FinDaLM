import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional, Union

import datasets
import evaluate
import psutil
import torch
import transformers
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

from .. import get_logger
from ..eval.eval import TASKS, load_datasetdict
from ..models.base import (
    MAP_MODELFORPRETRAINING,
    from_pretrained_with_modelforpretraining,
)
from ..models.llama import set_pad_token_to_model, set_pad_token_to_tokenizer
from ..models.moe.base import MOE_TYPES
from ..models.moe.deberta_v2 import (
    DebertaV2MoEForMaskedLM,
    load_pretrained_deberta_v2_into_moe,
)
from ..models.moe.llama import LlamaMoEForCausalLM, load_pretrained_llama_into_moe
from .configuration_utils import ProfilerCallback
from .dataset import load_ds
from .dataset.create import convert_sentence_to_ids
from .moe import freeze_except_mlp, freeze_except_router

logger = get_logger()

# This files mainly relies on transformers/examples/pytorch/language-modeling/run_clm.py


class PretrainMode(Enum):
    DEFAULT = "default"
    MOE_STAGE1 = "moe-stage1"
    MOE_STAGE2 = "moe-stage2"
    SIMILARITY = "similarity"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    pretrain_mode: Literal["default", "moe-stage1", "moe-stage2", "similarity"] = field(
        metadata={"help": "Choose from: " + ",".join([x.value for x in PretrainMode])}
    )
    model_type: str = field(
        metadata={
            "help": "Model type from the list: "
            + ", ".join(MAP_MODELFORPRETRAINING.keys())
        }
    )
    pretrained_model_name_or_dir: list[str] = field(
        metadata={"help": "Pretrained model names or directories"}
    )
    front_frozen_layers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of layers to freeze and not use with the MoE architecture."
                "Only available for Llama."
            )
        },
    )
    moe_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Adaptation ways of Experts if pretrain_mode is moe-stage2. "
                "Choose from: "
            )
            + ", ".join(MOE_TYPES)
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded "
                "from huggingface.co"
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name "
                "or commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` "
                "(necessary to use this script with private models)."
            )
        },
    )
    do_profile: bool = field(
        default=False, metadata={"help": "Will use profile with pytorch profiler"}
    )
    profile_dir: str = field(
        default="materials/profile",
        metadata={"help": "Profile directory when enables do_profile"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model
    for training and eval.
    """

    dataset_names: list[str] = field(
        metadata={
            "help": (
                "The names or directories of the dataset to use "
                "(via the datasets library or local disk). "
                "If pretrain_mode is moe-stage2, use task name like finerord"
            )
        }
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    validation_split_size: Optional[int] = field(
        default=2000,
        metadata={"help": "The number of the train samples used as validation set"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    is_dataset_masked: bool = field(
        default=False,
        metadata={
            "help": (
                "Skipping mask process if input datasets are already masked "
                "(only for mlm model)"
            )
        },
    )


def main() -> None:
    # arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    pretrain_mode: PretrainMode = PretrainMode[
        dict([(x.value, x.name) for x in PretrainMode])[model_args.pretrain_mode]
    ]

    # global variables
    ram_gb: float = psutil.virtual_memory().available / 1073741824
    datasets.config.IN_MEMORY_MAX_SIZE = (
        ram_gb * 0.6 * 10**9 / max(torch.cuda.device_count(), 1)
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and "
                "is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or "
                "add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_dir[0]
    )
    if model_args.model_type == "llama":
        set_pad_token_to_tokenizer(tokenizer)

    # load model config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_args.pretrained_model_name_or_dir[0], **config_kwargs
    )

    # dataset
    lst_ds: list[Dataset]
    raw_dataset: Dataset
    if pretrain_mode == PretrainMode.MOE_STAGE2 or (
        pretrain_mode == PretrainMode.DEFAULT
        and all([x in TASKS for x in data_args.dataset_names])
    ):
        lst_ds_text: list[Dataset] = [
            load_datasetdict(dsname)["train"] for dsname in data_args.dataset_names
        ]
        lst_ds = [convert_sentence_to_ids(ds, tokenizer) for ds in lst_ds_text]
        raw_dataset: Dataset = concatenate_datasets(lst_ds)
        max_length: int = min(
            config.max_position_embeddings,
            max(map(len, raw_dataset["tokens"]))
            + tokenizer.num_special_tokens_to_add(pair=False),
        )
        raw_dataset = raw_dataset.map(
            lambda example: tokenizer.pad(
                {
                    "input_ids": tokenizer.build_inputs_with_special_tokens(
                        example["tokens"]
                    )
                },
                max_length=max_length,
                padding="max_length",
            ),
            cache_file_name=None,
            remove_columns=["tokens"],
            load_from_cache_file=False,
        )
        if data_args.is_dataset_masked:
            logger.warning(
                "Although is_dataset_masked is enabled, it is disabled in moe-stage2"
            )
            data_args.is_dataset_masked = False
    else:
        lst_ds = [load_ds(dsname) for dsname in data_args.dataset_names]
        raw_dataset = concatenate_datasets(lst_ds)

    # model
    torch_dtype: Optional[torch.dtype] = None
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    model: PreTrainedModel
    if pretrain_mode == PretrainMode.MOE_STAGE2:
        assert model_args.moe_type is not None
        exclude_mlm_head: bool
        if model_args.model_type == "deberta-v2":
            exclude_mlm_head = True
            model = load_pretrained_deberta_v2_into_moe(
                DebertaV2MoEForMaskedLM,
                moe_type=model_args.moe_type,
                model_names=model_args.pretrained_model_name_or_dir,
                torch_dtype=torch_dtype,
            )
        elif model_args.model_type == "llama":
            exclude_mlm_head = False
            model = load_pretrained_llama_into_moe(
                LlamaMoEForCausalLM,
                moe_type=model_args.moe_type,
                model_names=model_args.pretrained_model_name_or_dir,
                torch_dtype=torch_dtype,
                front_frozen_layers=model_args.front_frozen_layers,
            )
        else:
            raise NotImplementedError()
        freeze_except_router(model, exclude_mlm_head, model_args.front_frozen_layers)
    else:
        model = from_pretrained_with_modelforpretraining(
            model_args.model_type,
            model_args.pretrained_model_name_or_dir[0],
            config=config,
            torch_dtype=torch_dtype,
        )
        exclude_mlm_head: bool
        if model_args.model_type == "deberta-v2":
            exclude_mlm_head = True
        elif model_args.model_type == "llama":
            exclude_mlm_head = False
        else:
            raise NotImplementedError()
        if pretrain_mode == PretrainMode.MOE_STAGE1:
            freeze_except_mlp(model, exclude_mlm_head, model_args.front_frozen_layers)
    if model_args.model_type == "llama":
        set_pad_token_to_model(model, tokenizer)

    lm_type: str = MAP_MODELFORPRETRAINING[model_args.model_type].lm_type
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training model - Total size={n_params / 2 ** 20:.2f}M params")

    train_dataset: Dataset
    if not training_args.do_eval:
        train_dataset = raw_dataset
    else:
        dsd: datasets.DatasetDict = raw_dataset.train_test_split(
            test_size=data_args.validation_split_size, seed=training_args.seed
        )
        train_dataset = dsd["train"]
        eval_dataset: Dataset = dsd["test"]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")
        if lm_type == "clm":

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1)
                # has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

        elif lm_type == "mlm":

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1)
                # has been calculated
                # by preprocess_logits_for_metrics
                labels = labels.reshape(-1)
                preds = preds.reshape(-1)
                mask = labels != -100
                labels = labels[mask]
                preds = preds[mask]
                return metric.compute(predictions=preds, references=labels)

        else:  # pragma: no cover
            raise ValueError(f"Invalid lm_type: {lm_type}")

    data_collator: Union[DataCollatorForLanguageModeling, Callable]
    if data_args.is_dataset_masked:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=(lm_type == "mlm"),
            mlm_probability=0.15,
            pad_to_multiple_of=8,
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if model_args.do_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=5, wait=2, warmup=3, active=5, repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                model_args.profile_dir
            ),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ) as prof:
            trainer.add_callback(ProfilerCallback(prof))
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":  # pragma: no cover
    main()
