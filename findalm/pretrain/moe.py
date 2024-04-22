from transformers import DebertaV2ForMaskedLM, PreTrainedModel

from ..models.moe.deberta_v2 import DebertaV2MoEForMaskedLM


def get_trainable_million_params(m: PreTrainedModel) -> float:
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 10**6


def freeze_except_mlp(model: PreTrainedModel, exclude_mlm_head: bool) -> None:
    num_params: float = get_trainable_million_params(model)
    if isinstance(model, DebertaV2ForMaskedLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(len(model.deberta.encoder.layer)):
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
        avail_params: float = get_trainable_million_params(model)
        print(f"{avail_params:.1f}M / {num_params:.1f}M params are trainable")
    else:
        raise NotImplementedError()


def freeze_except_router(model: PreTrainedModel, exclude_mlm_head: bool) -> None:
    num_params: float = get_trainable_million_params(model)
    if isinstance(model, DebertaV2MoEForMaskedLM):
        for _, param in model.named_parameters():
            param.requires_grad = False
        for i in range(len(model.deberta.encoder.layer)):
            model.deberta.encoder.layer[i].router.weight.requires_grad = True
        if exclude_mlm_head:
            model.cls.predictions.bias.requires_grad = True
            model.cls.predictions.decoder.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.bias.requires_grad = True
            model.cls.predictions.transform.LayerNorm.weight.requires_grad = True
            model.cls.predictions.transform.dense.bias.requires_grad = True
            model.cls.predictions.transform.dense.weight.requires_grad = True
        avail_params: float = get_trainable_million_params(model)
        print(f"{avail_params:.1f}M / {num_params:.1f}M params are trainable")
    else:
        raise NotImplementedError()
