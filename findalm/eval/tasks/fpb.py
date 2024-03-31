from datasets import DatasetDict, load_dataset


def load_datasetdict() -> DatasetDict:
    dsd: DatasetDict = load_dataset("TheFinAI/en-fpb")
    dsd = dsd.rename_column("gold", "labels")
    return dsd
