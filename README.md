# FinDaLM: Financial Domain-adapted Language Model

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue">
  <a href="https://github.com/retarfi/FinDaLM/actions/workflows/format.yml">
    <img alt="Test" src="https://github.com/retarfi/FinDaLM/actions/workflows/format.yml/badge.svg">
  </a>
</p>

## Installation
```sh
poetry lock
poetry install
poetry run pip install flash-attn --no-build-isolation
```

## Usage
### Create Dataset
The directory of datasets to be input with `input-datasets` is assumed to have been created and stored using the `datasets` library as follows:
```python
from datasets import Dataset
ds: Dataset = Dataset.from_dict({
    "text": ["sent1", "sent2", "sent3"]
})
ds.save_to_disk("huga")
```

Each Example `text` is composed of `str`.<br>
Conversion from .txt files to `Dataset` is not implemented here.<br>

Multiple `input-datasets` can be specified as follows.<br>
Also, you can specify more than one datasets at pre-training, so there is no need to combine them here.

```sh
# For DeBERTaV2
poetry run python -m findalm.pretrain.dataset.create \
--input_datasets foo bar \
--output_dir materials/dataset/baz \
--pretrained_model_name_or_dir model-name-or-directory \
--max_length 512 \
(--do_mask \)
--num_proc 4
```


### Training
```sh
# For DeBERTaV2
poetry run python pretrain.py \
--model_type deberta-v2 \
--pretrain_mode <default|moe-stage1|moe-stage2|similarity> \
--pretrained_model_name_or_dir model-name-or-directory \
--dataset_names materials/dataset/baz \
--validation_split_size 10 \
--output_dir materials/model/debug \
--do_train \
--do_eval \
--evaluation_strategy steps \
--max_steps 5 \
--save_steps 5 \
--seed 42 \
--data_seed 42
```

#### wandb
Use `WANDB_ENTITY=retarfi WANDB_PROJECT=deberta WANDB_NAME=runname` after
`wandb login <api key>`


#### torch.profiler
Using `--do_profile`, torch.profile is executed and the json file is saved in materials/profile.
```sh
pip install torch-tb-profiler
tensorboard --logdir materials/profile
```
then you can see the profile in TensorBoard.


### Evaluation
To evaluate with FPB task, you have to be granted access to [FPB dataset](https://huggingface.co/datasets/TheFinAI/en-fpb) and login with `huggingface-cli login`
```sh
poetry run python -m findalm.eval.eval \
<task> \
-m <model_name_or_directory> \
-a <accumulation_steps> \
--seed 0 \
(--mlflow) \
(--grid)
```


### Test
```sh
# Do whole tests including model forwarding with GPU and avoid using wandb
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true poetry run pytest

codecov -t <TOKEN>
```

### Format
```sh
poetry run black --check --diff --quiet --skip-magic-trailing-comma .
```
