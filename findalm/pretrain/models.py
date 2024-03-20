from typing import NamedTuple, Type

from transformers import DebertaV2ForMaskedLM, LlamaForCausalLM, PreTrainedModel


class ModelInfo(NamedTuple):
    cls_pretrained: Type[PreTrainedModel]
    lm_type: str


MAP_MODELFORPRETRAINING: dict[str, Type[PreTrainedModel]] = {
    "deberta-v2": ModelInfo(DebertaV2ForMaskedLM, "mlm"),
    "llama-2": ModelInfo(LlamaForCausalLM, "clm"),
}


def from_pretrained_with_modelforpretraining(
    model_type: str, model_name_or_path: str, **kwargs
) -> PreTrainedModel:
    model_type = model_type.lower()
    if model_type not in MAP_MODELFORPRETRAINING.keys():
        raise ValueError(f"{model_type} is not in available model list")
    if model_type == "llama-2":
        kwargs.update(attn_implementation="flash_attention_2")
    return MAP_MODELFORPRETRAINING[model_type].cls_pretrained.from_pretrained(
        model_name_or_path, **kwargs
    )
