from transformers import (
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    LlamaForCausalLM,
    PreTrainedModel,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    T5ForTokenClassification,
)

from ..models.moe.deberta_v2 import (
    DebertaV2FFN,
    DebertaV2MoEForMaskedLM,
    DebertaV2MoEForSequenceClassification,
    DebertaV2MoEForTokenClassification,
)
from ..models.moe.llama import LlamaMLP, LlamaMoEForCausalLM
from ..models.moe.t5 import (
    T5LayerFF,
    T5MoEForConditionalGeneration,
    T5MoEForSequenceClassification,
    T5MoEForTokenClassification,
)


def get_trainable_million_params(m: PreTrainedModel) -> float:
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 10**6


def freeze_except_mlp(
    model: PreTrainedModel, exclude_mlm_head: bool, front_frozen_layers: int
) -> None:
    num_params: float = get_trainable_million_params(model)
    for _, param in model.named_parameters():
        param.requires_grad = False
    if isinstance(model, DebertaV2ForMaskedLM):
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
        for i in range(front_frozen_layers, len(model.model.layers)):
            model.model.layers[i].mlp.gate_proj.weight.requires_grad = True
            model.model.layers[i].mlp.up_proj.weight.requires_grad = True
            model.model.layers[i].mlp.down_proj.weight.requires_grad = True
    elif isinstance(model, T5ForConditionalGeneration):
        for i in range(front_frozen_layers, len(model.encoder.block)):
            model.encoder.block[i].layer[
                1
            ].DenseReluDense.wi.weight.requires_grad = True
            model.encoder.block[i].layer[
                1
            ].DenseReluDense.wo.weight.requires_grad = True
        for i in range(front_frozen_layers, len(model.decoder.block)):
            model.decoder.block[i].layer[
                2
            ].DenseReluDense.wi.weight.requires_grad = True
            model.decoder.block[i].layer[
                2
            ].DenseReluDense.wo.weight.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")


def freeze_except_router_and_mlp(
    model: PreTrainedModel, exclude_mlm_head: bool, front_frozen_layers: int
) -> None:
    num_params: float = get_trainable_million_params(model)
    for _, param in model.named_parameters():
        param.requires_grad = False
    if isinstance(model, DebertaV2MoEForMaskedLM):
        for i in range(front_frozen_layers, len(model.deberta.encoder.layer)):
            model.deberta.encoder.layer[i].router.weight.requires_grad = True
            for j in range(model.num_experts):
                if not isinstance(
                    model.deberta.encoder.layer[i].experts[j], DebertaV2FFN
                ):
                    continue
                model.deberta.encoder.layer[i].experts[
                    j
                ].intermediate.dense.weight.requires_grad = True
                model.deberta.encoder.layer[i].experts[
                    j
                ].intermediate.dense.bias.requires_grad = True
                model.deberta.encoder.layer[i].experts[
                    j
                ].output.dense.weight.requires_grad = True
                model.deberta.encoder.layer[i].experts[
                    j
                ].output.dense.bias.requires_grad = True
        if exclude_mlm_head:
            model.cls.predictions.bias.requires_grad = True
            model.cls.predictions.decoder.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.weight.requires_grad = True
            model.cls.predictions.transform.dense.bias.requires_grad = True
            model.cls.predictions.transform.dense.weight.requires_grad = True
    elif isinstance(model, LlamaMoEForCausalLM):
        for i in range(front_frozen_layers, len(model.model.layers)):
            model.model.layers[i].router.weight.requires_grad = True
            for j in range(model.num_experts):
                if not isinstance(model.model.layers[i].experts[j], LlamaMLP):
                    continue
                model.model.layers[i].experts[j].gate_proj.weight.requires_grad = True
                model.model.layers[i].experts[j].up_proj.weight.requires_grad = True
                model.model.layers[i].experts[j].down_proj.weight.requires_grad = True
    elif isinstance(model, T5MoEForConditionalGeneration):
        for i in range(front_frozen_layers, len(model.encoder.block)):
            model.encoder.block[i].router.weight.requires_grad = True
            for j in range(model.num_experts):
                if not isinstance(model.encoder.block[i].experts[j], T5LayerFF):
                    continue
                model.encoder.block[i].experts[
                    j
                ].DenseReluDense.wi.weight.requires_grad = True
                model.encoder.block[i].experts[
                    j
                ].DenseReluDense.wo.weight.requires_grad = True
        for i in range(front_frozen_layers, len(model.decoder.block)):
            model.decoder.block[i].router.weight.requires_grad = True
            for j in range(model.num_experts):
                if not isinstance(model.decoder.block[i].experts[j], T5LayerFF):
                    continue
                model.decoder.block[i].experts[
                    j
                ].DenseReluDense.wi.weight.requires_grad = True
                model.decoder.block[i].experts[
                    j
                ].DenseReluDense.wo.weight.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")


def freeze_layers(model: PreTrainedModel, num_freeze_layers: int) -> None:
    num_params: float = get_trainable_million_params(model)
    for _, param in model.named_parameters():
        param.requires_grad = False
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
        for i in range(num_freeze_layers, len(model.deberta.encoder.layer)):
            for _, param in model.deberta.encoder.layer[i].named_parameters():
                param.requires_grad = True
        model.classifier.bias.requires_grad = True
        model.classifier.weight.requires_grad = True
        if isinstance(model, DebertaV2ForSequenceClassification) or isinstance(
            model, DebertaV2MoEForSequenceClassification
        ):
            model.pooler.dense.bias.requires_grad = True
            model.pooler.dense.weight.requires_grad = True
    elif any(
        map(
            lambda x: isinstance(model, x),
            (
                T5ForSequenceClassification,
                T5MoEForSequenceClassification,
                T5ForTokenClassification,
                T5MoEForTokenClassification,
            ),
        )
    ):
        if any(
            map(
                lambda x: isinstance(model, x),
                (T5ForSequenceClassification, T5MoEForSequenceClassification),
            )
        ):
            model.classification_head.dense.bias.requires_grad = True
            model.classification_head.dense.weight.requires_grad = True
            for i in range(num_freeze_layers, len(model.transformer.encoder.block)):
                for _, param in model.transformer.encoder.block[i].named_parameters():
                    param.requires_grad = True
                for _, param in model.transformer.decoder.block[i].named_parameters():
                    param.requires_grad = True
        else:
            model.classifier.bias.requires_grad = True
            model.classifier.weight.requires_grad = True
            for i in range(num_freeze_layers, len(model.transformer.encoder.block)):
                for _, param in model.transformer.encoder.block[i].named_parameters():
                    param.requires_grad = True
    else:
        raise NotImplementedError()
    avail_params: float = get_trainable_million_params(model)
    print(f"{avail_params:.2f}M / {num_params:.2f}M params are trainable")
