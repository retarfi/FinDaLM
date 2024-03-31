import pytest

from findalm.eval.tasks.headline import MAP_SUBTASK_COLUMNS, load_datasetdict


@pytest.mark.parametrize("subtask", list(MAP_SUBTASK_COLUMNS.keys()))
def test_load_datasetdict(subtask: str) -> None:
    _ = load_datasetdict(subtask, seed=0)
