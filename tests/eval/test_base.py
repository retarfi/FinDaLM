from typing import Callable

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction

from findalm.eval.base import convert_dataset_to_datasetdict, load_compute_metrics


def test_convert_dataset_to_datasetdict() -> None:
    ds: Dataset = Dataset.from_dict({"A": np.arange(1000)})
    dsd: DatasetDict = convert_dataset_to_datasetdict(ds, seed=0)
    assert len(dsd["train"]) == 700
    assert len(dsd["valid"]) == 100
    assert len(dsd["test"]) == 200


@pytest.mark.parametrize(
    "metric_name, score", [("accuracy", 0.75), ("f1-micro", 0.75), ("f1-macro", 0.733)]
)
def test_load_compute_metrics(metric_name: str, score: float) -> None:
    metric: Callable = load_compute_metrics(metric_name)
    p: EvalPrediction = EvalPrediction(
        predictions=np.array([[0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]]),
        label_ids=np.array([0, 1, 1, 1]),
    )
    assert abs(metric(p)[metric_name] - score) < 1e-2
