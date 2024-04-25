import itertools
import re
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
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
from .base import (
    MOE_TYPES,
    ROUTER_AUX_LOSS_COEF,
    Skipper,
    confirm_same_weights,
    load_balancing_loss_func,
)

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
    if moe_type == "dense":
        config.use_router_loss = False
    else:
        config.use_router_loss = True
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
            m: Optional[re.Match] = re.match(
                r"deberta\.encoder\.layer\.(\d+).(intermediate|output)(.+)$", name
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
        if re.match(r"top\d+", config.moe_type):
            if re.match(r"top\d+-skip", config.moe_type):
                modules = [DebertaV2FFN(config) for _ in range(self.num_experts - 1)]
                modules.append(Skipper())
            else:
                modules = [DebertaV2FFN(config) for _ in range(self.num_experts)]
            self.moe_type = "topk"
            self.top_k = int(re.match(r"top(\d+)", config.moe_type).group(1))
        elif config.moe_type == "dense":
            self.moe_type = config.moe_type
            modules = [DebertaV2FFN(config) for _ in range(self.num_experts)]
        else:
            raise ValueError()
        self.experts = nn.ModuleList(modules)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

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
        batch_size, sequence_length, hidden_dim = attention_output.shape
        if self.moe_type == "topk":
            hidden_states = attention_output.view(-1, hidden_dim)
            router_logits = self.router(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_expert = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            layer_output = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(
                selected_expert, num_classes=self.num_experts
            ).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                if top_x.shape[0] == 0:
                    continue

                # in torch it is faster to index using lists than torch tensors
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state)
                    * routing_weights[top_x_list, idx_list, None]
                )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                layer_output.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            layer_output = layer_output.reshape(batch_size, sequence_length, hidden_dim)
        else:
            router_logits = self.router(attention_output)
            routing_weights = F.softmax(router_logits, dim=2)
            layer_outputs = torch.stack(
                [ex(attention_output) for ex in self.experts], dim=2
            )
            layer_output = torch.sum(
                routing_weights.unsqueeze(-1) * layer_outputs, dim=2
            )
        if output_attentions:
            return (layer_output, router_logits, att_matrix)
        else:
            return (layer_output, router_logits)


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

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        # return_dict=True,
    ) -> Tuple:
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_router_logits = ()
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, router_logits, att_m = output_states
            else:
                output_states, router_logits = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)
            all_router_logits = all_router_logits + (router_logits,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        # if not return_dict:
        # return tuple(v for v in [output_states, all_router_logits, all_hidden_states, all_attentions] if v is not None)
        return (output_states, all_router_logits, all_hidden_states, all_attentions)
        # return BaseModelOutput(
        #     last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        # )


class DebertaV2MoEModel(DebertaV2Model):
    def __init__(self, config):
        super(DebertaV2Model, self).__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2MoEEncoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        router_logits = encoder_outputs[1]
        encoded_layers = encoder_outputs[2]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]
        return (sequence_output, router_logits) + encoder_outputs[2:]
        # return (sequence_output, router_logits) + encoder_outputs[(2 if output_hidden_states else 3) :]


class DebertaV2MoEForMaskedLM(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super(DebertaV2ForMaskedLM, self).__init__(config)

        self.deberta = DebertaV2MoEModel(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        self.num_experts = config.num_experts
        self.use_router_loss = config.use_router_loss
        assert config.moe_type in MOE_TYPES
        assert not self.use_router_loss or config.moe_type != "dense"
        m = re.match(r"top(\d+)", config.moe_type)
        if m is not None:
            self.top_k = int(m.group(1))
            self.router_aux_loss_coef = ROUTER_AUX_LOSS_COEF

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            if self.use_router_loss:
                aux_loss = load_balancing_loss_func(
                    outputs[1], self.num_experts, self.top_k, attention_mask
                )
                masked_lm_loss += self.router_aux_loss_coef * aux_loss.to(
                    masked_lm_loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs[2] if output_hidden_states else None,
            attentions=outputs[3] if output_attentions else None,
        )


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

        self.num_experts = config.num_experts
        self.use_router_loss = config.use_router_loss
        assert config.moe_type in MOE_TYPES
        assert not self.use_router_loss or config.moe_type != "dense"
        m = re.match(r"top(\d+)", config.moe_type)
        if m is not None:
            self.top_k = int(m.group(1))
            self.router_aux_loss_coef = ROUTER_AUX_LOSS_COEF

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits,
                            0,
                            label_index.expand(label_index.size(0), logits.size(1)),
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            labeled_logits.view(-1, self.num_labels).float(),
                            labels.view(-1),
                        )
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            if self.use_router_loss:
                aux_loss = load_balancing_loss_func(
                    outputs[1], self.num_experts, self.top_k, attention_mask
                )
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[2] if output_hidden_states else None,
            attentions=outputs[3] if output_attentions else None,
        )


class DebertaV2MoEForTokenClassification(DebertaV2ForTokenClassification):
    def __init__(self, config):
        super(DebertaV2ForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2MoEModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.num_experts = config.num_experts
        self.use_router_loss = config.use_router_loss
        assert config.moe_type in MOE_TYPES
        assert not self.use_router_loss or config.moe_type != "dense"
        m = re.match(r"top(\d+)", config.moe_type)
        if m is not None:
            self.top_k = int(m.group(1))
            self.router_aux_loss_coef = ROUTER_AUX_LOSS_COEF

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.use_router_loss:
                aux_loss = load_balancing_loss_func(
                    outputs[1], self.num_experts, self.top_k, attention_mask
                )
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[2] if output_hidden_states else None,
            attentions=outputs[3] if output_attentions else None,
        )
