from datasets import DatasetDict, load_dataset


def load_datasetdict() -> DatasetDict:
    dsd: DatasetDict = load_dataset("gtfintechlab/fomc_communication")
    dsd = dsd.rename_columns(({"sentence": "text", "label": "labels"}))
    train_valid: DatasetDict = dsd.pop("train").train_test_split(
        test_size=len(dsd["test"])
    )
    dsd["train"] = train_valid["train"]
    dsd["valid"] = train_valid["test"]
    return dsd
