import os
from pathlib import Path

import pytest
from findalm.pretrain.dataset.load import load_ds

P_THIS_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))
P_TEST_ROOT_DIR: Path = P_THIS_DIR.parent.parent


@pytest.mark.parametrize(
    "path", ["lhoestq/demo1", str(P_TEST_ROOT_DIR / "data" / "datasets" / "botchan")]
)
def test_load_ds(path: str) -> None:
    load_ds(path)
