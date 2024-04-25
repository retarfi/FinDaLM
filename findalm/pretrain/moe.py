from transformers import (
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    LlamaForCausalLM,
    PreTrainedModel,
)

from ..models.moe.deberta_v2 import (
    DebertaV2MoEForMaskedLM,
    DebertaV2MoEForSequenceClassification,
    DebertaV2MoEForTokenClassification,
)
from ..models.moe.llama import LlamaMoEForCausalLM


def get_trainable_million_params(m: PreTrainedModel) -> float:
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 10**6


def freeze_except_mlp(
    model: PreTrainedModel, exclude_mlm_head: bool, front_frozen_layers: int
) -> None:
    num_params: float = get_trainable_million_params(model)
    if isinstance(model, DebertaV2ForMaskedLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(front_frozen_layers, len(model.deberta.encoder.layer)):
            model.deberta.encoder.layer[i].intermediate.dense.weight.requires_grad = (
                True
            )
            model.deberta.encoder.layer[i].intermediate.dense.bias.requires_grad = True
            model.deberta.encoder.layer[i].output.dense.weight.requires_grad = True
            model.deberta.encoder.layer[i].output.dense.bias.requires_grad = True
        if exclude_mlm_head:
            model.cls.predictions.bias.requires_grad = True
            model.cls.predictions.decoder.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.weight.requires_grad = True
            model.cls.predictions.transform.dense.bias.requires_grad = True
            model.cls.predictions.transform.dense.weight.requires_grad = True
    elif isinstance(model, LlamaForCausalLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(front_frozen_layers, len(model.model.layers)):
            model.model.layers[i].mlp.gate_proj.weight.requires_grad = True
            model.model.layers[i].mlp.up_proj.weight.requires_grad = True
            model.model.layers[i].mlp.down_proj.weight.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")


def freeze_except_router(
    model: PreTrainedModel, exclude_mlm_head: bool, front_frozen_layers: int
) -> None:
    num_params: float = get_trainable_million_params(model)
    if isinstance(model, DebertaV2MoEForMaskedLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(front_frozen_layers, len(model.deberta.encoder.layer)):
            model.deberta.encoder.layer[i].router.weight.requires_grad = True
        if exclude_mlm_head:
            model.cls.predictions.bias.requires_grad = True
            model.cls.predictions.decoder.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.weight.requires_grad = True
            model.cls.predictions.transform.dense.bias.requires_grad = True
            model.cls.predictions.transform.dense.weight.requires_grad = True
    elif isinstance(model, LlamaMoEForCausalLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(front_frozen_layers, len(model.model.layers)):
            model.model.layers[i].router.weight.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")


def freeze_except_cls(model: PreTrainedModel) -> None:
    num_params: float = get_trainable_million_params(model)
    if any(
        map(
            lambda x: isinstance(model, x),
            (
                DebertaV2ForSequenceClassification,
                DebertaV2MoEForSequenceClassification,
                DebertaV2ForTokenClassification,
                DebertaV2MoEForTokenClassification,
            ),
        )
    ):
        for _, param in model.named_parameters():
            param.requires_grad = False
        model.classifier.bias.requires_grad = True
        model.classifier.weight.requires_grad = True
        if isinstance(model, DebertaV2ForSequenceClassification) or isinstance(
            model, DebertaV2MoEForSequenceClassification
        ):
            model.pooler.bias.requires_grad = True
            model.pooler.weight.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")
