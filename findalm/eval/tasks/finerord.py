from datasets import Dataset, DatasetDict, load_dataset

from ..base import NER_LABELS


def load_datasetdict() -> DatasetDict:
    dsd: DatasetDict = load_dataset("gtfintechlab/finer-ord")
    dsd["valid"] = dsd.pop("validation")
    for split in ("train", "valid", "test"):
        data: dict[str, list[str]] = {"tokens": [[]], "bio_tag": [[]]}
        sent_idx: int = dsd[split][0]["sent_idx"]
        for d in dsd[split]:
            if d["gold_token"] is None:
                d["gold_token"] = "None"
            if sent_idx == d["sent_idx"]:
                data["tokens"][-1].append(d["gold_token"])
                data["bio_tag"][-1].append(NER_LABELS[d["gold_label"]])
            else:
                data["tokens"].append([d["gold_token"]])
                data["bio_tag"].append([NER_LABELS[d["gold_label"]]])
            sent_idx = d["sent_idx"]
        data["text"] = list(map(lambda x: " ".join(x), data["tokens"]))
        del data["tokens"]
        dsd[split] = Dataset.from_dict(data)
    return dsd
