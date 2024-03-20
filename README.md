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
`input-datasets`で入力するデータセットのディレクトリは`datasets`ライブラリを用いて次のように作成・保存されたものを想定しています。
```python
from datasets import Dataset
ds: Dataset = Dataset.from_dict({
    "text": ["sent1", "sent2", "sent3"]
})
ds.save_to_disk("huga")
```
各Exampleの`text`は`str`によって成り立ちます。<br>
.txtファイルから`Dataset`への変換はここでは実装していません。

`input-datasets`は以下のように複数指定できます。<br>
また、Pre-train時に複数指定できるようにもしてあるので、ここで無理に結合する必要はありません。

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
poetry run python -m findalm.train.pretrain \
--model_type deberta-v2 \
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

# For Llama
poetry run python -m findalm.train.pretrain \
--model_type llama-2 \
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
--data_seed 42 \
--bf16
```



#### wandb
use `WANDB_ENTITY=retarfi WANDB_PROJECT=deberta WANDB_NAME=runname` after
`wandb login <api key>`


#### torch.profiler
`--do_profile`を使うとtorch.profileが実施されmaterials/profileにjsonファイルが保存される。
```sh
pip install torch-tb-profiler
tensorboard --logdir materials/profile
```
とすればtensorboardでprofileが見れる。



### Test
```sh
# Normal pytest (shorter time consume)
poetry run pytest
# Do whole tests including model forwarding with GPU and avoid using wandb
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true RUN_SLOW=true poetry run pytest

codecov -t <TOKEN>
```

### Format
```sh
poetry run black --check --diff --quiet --skip-magic-trailing-comma .
```
