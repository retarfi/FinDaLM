from typing import Callable

import evaluate
import numpy as np
import transformers
from datasets import Dataset, DatasetDict

NER_LABELS: list[str] = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]


def convert_dataset_to_datasetdict(ds: Dataset, seed: int) -> DatasetDict:
    train_test: Dataset = ds.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    ds_test: Dataset = train_test["test"]
    train_valid: DatasetDict = train_test["train"].train_test_split(
        test_size=0.125, shuffle=True, seed=seed
    )
    ds_train: Dataset = train_valid["train"]
    ds_valid: Dataset = train_valid["test"]
    dsd: DatasetDict = DatasetDict(
        {"train": ds_train, "valid": ds_valid, "test": ds_test}
    )
    return dsd


def load_compute_metrics(metric_name: str) -> Callable:
    if metric_name == "accuracy":

        def compute_metrics(p: transformers.EvalPrediction) -> dict[str, float]:
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=-1)
            metrics: evaluate.Metric = evaluate.load("accuracy")
            result: dict[str, float] = metrics.compute(
                predictions=np.ravel(preds), references=np.ravel(p.label_ids)
            )
            return result

    elif metric_name == "f1-micro":

        def compute_metrics(p: transformers.EvalPrediction) -> dict[str, float]:
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=-1)
            metrics: evaluate.Metric = evaluate.load("f1")
            result: dict[str, float] = metrics.compute(
                predictions=np.ravel(preds),
                references=np.ravel(p.label_ids),
                average="micro",
            )
            result[metric_name] = result.pop("f1")
            return result

    elif metric_name == "f1-macro":

        def compute_metrics(p: transformers.EvalPrediction) -> dict[str, float]:
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=-1)
            metrics: evaluate.Metric = evaluate.load("f1")
            result: dict[str, float] = metrics.compute(
                predictions=np.ravel(preds),
                references=np.ravel(p.label_ids),
                average="macro",
            )
            result[metric_name] = result.pop("f1")
            return result

    else:
        raise NotImplementedError(f"metric {metric_name} is not implemented")
    return compute_metrics
