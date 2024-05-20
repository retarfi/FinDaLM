from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from ..base import convert_dataset_to_datasetdict

P_ROOT: Path = Path(__file__).parents[3]

MAP_SUBTASK_COLUMNS: dict[str, str] = {
    "price": "Price or Not",
    "dirup": "Direction Up",
    "dirconstant": "Direction Constant",
    "dirdown": "Direction Down",
    "pastprice": "PastPrice",
    "futureprice": "FuturePrice",
    "pastnews": "PastNews",
    "futurenews": "FutureNews",
    "asset": "Asset Comparision",
}


def load_datasetdict(subtask: str, seed: int) -> DatasetDict:
    assert subtask in MAP_SUBTASK_COLUMNS.keys()
    df: pd.DataFrame = pd.read_csv(P_ROOT / "data" / "headline.csv")
    assert all(
        map(lambda x: x in df.columns, MAP_SUBTASK_COLUMNS.values())
    ), f"{set(MAP_SUBTASK_COLUMNS.values()) - set(df.columns)} not in df.columns"
    ds: Dataset = Dataset.from_pandas(df)
    ds = ds.rename_columns({"News": "text", MAP_SUBTASK_COLUMNS[subtask]: "labels"})
    dsd: DatasetDict = convert_dataset_to_datasetdict(ds, seed=seed)
    return dsd
