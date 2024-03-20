import logging
import os
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from ... import get_logger

logger: logging.Logger = get_logger()


def load_ds(
    path: str, use_auth_token: Optional[str] = None, streaming: bool = False
) -> Dataset:
    ds: Dataset
    if os.path.exists(path):
        logger.info(f"Loading dataset in {path} ...")
        ds = load_from_disk(path)
    else:
        logger.info(f"Loading dataset {path} from Hub...")
        dsd: Union[Dataset, DatasetDict] = load_dataset(
            path, use_auth_token=use_auth_token, streaming=streaming
        )
        if isinstance(dsd, DatasetDict):
            ds = dsd["train"]
        else:
            ds = dsd
    return ds
