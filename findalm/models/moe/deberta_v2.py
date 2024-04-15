import itertools
import re
from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers import PreTrainedModel
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    ConvLayer,
    DebertaV2Attention,
    DebertaV2Embeddings,
    DebertaV2Encoder,
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Intermediate,
    DebertaV2Model,
    DebertaV2OnlyMLMHead,
    DebertaV2Output,
    StableDropout,
)

from ... import get_logger
from .base import MOE_TYPES, confirm_same_weights
from .router import Router, Skipper

logger = get_logger()


def load_pretrained_deberta_v2_into_moe(
    cls,
    moe_type: str,
    model_names: list[str],
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    base_cls = {
        DebertaV2MoEForMaskedLM: DebertaV2ForMaskedLM,
        DebertaV2MoEForSequenceClassification: DebertaV2ForSequenceClassification,
        DebertaV2MoEForTokenClassification: DebertaV2ForTokenClassification,
    }[cls]
    lst_models: list[PreTrainedModel] = [
        base_cls.from_pretrained(x, torch_dtype=torch_dtype) for x in model_names
    ]
    # assert other weight is same
    for m1, m2 in itertools.combinations(lst_models, 2):
        assert confirm_same_weights(m1.deberta.embeddings, m2.deberta.embeddings)
        for i in range(max(m1.config.num_hidden_layers, m1.config.num_hidden_layers)):
            assert confirm_same_weights(
                m1.deberta.encoder.layer[i].attention,
                m2.deberta.encoder.layer[i].attention,
            )
    config = lst_models[0].config
    config.num_experts = len(lst_models)
    if moe_type == "top2-skip":
        config.num_experts += 1
    config.moe_type = moe_type
    model: PreTrainedModel = cls(config)
    # copy base weight
    dct_params = dict(model.state_dict())
    init_params: list[str] = list(model.state_dict().keys())
    for name, param in dict(lst_models[0].state_dict()).items():
        if name in model.state_dict():
            dct_params[name].data.copy_(param.data)
            init_params.remove(name)
    for i, mm in enumerate(lst_models):
        for name, param in dict(mm.state_dict()).items():
            # for j in range(model.config.num_hidden_layers):
            m: Optional[re.Match] = re.match(
                r"deberta\.encoder\.layer\.(\d).(intermediate|output)(.+)$", name
            )
            if m is not None:
                param_name: str = (
                    f"deberta.encoder.layer.{m.group(1)}.experts.{i}.{m.group(2)}{m.group(3)}"
                )
                dct_params[param_name].data.copy_(param.data)
                init_params.remove(param_name)
    model.load_state_dict(dct_params)
    logger.info(f"Following params are not loaded and initialized: {init_params}")
    return model


class DebertaV2FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DebertaV2MoELayer(nn.Module):
    def __init__(self, config):
        assert config.moe_type in MOE_TYPES
        self.num_experts = config.num_experts
        super().__init__()
        self.attention = DebertaV2Attention(config)
        if config.moe_type == "top2-skip":
            self.moe_type = "top2"
            modules = [DebertaV2FFN(config) for _ in range(self.num_experts - 1)]
            modules.append(Skipper())
        else:
            self.moe_type = config.moe_type
            modules = [DebertaV2FFN(config) for _ in range(self.num_experts)]
        self.experts = nn.ModuleList(modules)
        self.router = Router(self.moe_type, config.hidden_size, config.num_experts)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        moe_weight = self.router(attention_output)
        layer_outputs = torch.stack(
            [ex(attention_output) for ex in self.experts], dim=2
        )
        layer_output = torch.sum(moe_weight.unsqueeze(-1) * layer_outputs, dim=2)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class DebertaV2MoEEncoder(DebertaV2Encoder):
    def __init__(self, config):
        super(DebertaV2Encoder, self).__init__()
        self.layer = nn.ModuleList(
            [DebertaV2MoELayer(config) for _ in range(config.num_hidden_layers)]
        )
        # copied from DebertaV2Encoder
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(
                config.hidden_size, config.layer_norm_eps, elementwise_affine=True
            )

        self.conv = (
            ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        )
        self.gradient_checkpointing = False


class DebertaV2MoEModel(DebertaV2Model):
    def __init__(self, config):
        super(DebertaV2Model, self).__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2MoEEncoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()


class DebertaV2MoEForMaskedLM(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super(DebertaV2ForMaskedLM, self).__init__(config)

        self.deberta = DebertaV2MoEModel(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


class DebertaV2MoEForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super(DebertaV2ForSequenceClassification, self).__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2MoEModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()


class DebertaV2MoEForTokenClassification(DebertaV2ForTokenClassification):
    def __init__(self, config):
        super(DebertaV2ForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2MoEModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
