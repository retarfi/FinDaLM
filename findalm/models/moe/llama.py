import itertools
import re
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
)

from ... import get_logger
from .base import (
    MOE_TYPES,
    ROUTER_AUX_LOSS_COEF,
    confirm_same_weights,
    load_balancing_loss_func,
    Skipper,
)

logger = get_logger()


def load_pretrained_llama_into_moe(
    cls,
    moe_type: str,
    model_names: list[str],
    front_frozen_layers: int = 0,
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    base_cls = {LlamaMoEForCausalLM: LlamaForCausalLM}[cls]
    lst_models: list[PreTrainedModel] = [
        base_cls.from_pretrained(x, torch_dtype=torch_dtype) for x in model_names
    ]
    # assert other weight is same
    for m1, m2 in itertools.combinations(lst_models, 2):
        assert confirm_same_weights(m1.model.embed_tokens, m2.model.embed_tokens)
        for i in range(max(m1.config.num_hidden_layers, m1.config.num_hidden_layers)):
            assert confirm_same_weights(
                m1.model.layers[i].self_attn, m2.model.layers[i].self_attn
            )
            if i < front_frozen_layers:
                assert confirm_same_weights(
                    m1.model.layers[i].mlp, m2.model.layers[i].mlp
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
    config.front_frozen_layers = front_frozen_layers
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
            m: Optional[re.Match] = re.match(r"model\.layers\.(\d+).mlp(.+)$", name)
            if m is not None and front_frozen_layers <= int(m.group(1)):
                param_name: str = f"model.layers.{m.group(1)}.experts.{i}{m.group(2)}"
                dct_params[param_name].data.copy_(param.data)
                init_params.remove(param_name)
    model.load_state_dict(dct_params)
    logger.info(f"Following params are not loaded and initialized: {init_params}")
    return model


class LlamaMoEDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        assert config.moe_type in MOE_TYPES
        self.num_experts = config.num_experts
        if re.match(r"top\d+", config.moe_type):
            if re.match(r"top\d+-skip", config.moe_type):
                modules = [LlamaMLP(config) for _ in range(self.num_experts - 1)]
                modules.append(Skipper())
            else:
                modules = [LlamaMLP(config) for _ in range(self.num_experts)]
            self.moe_type = "topk"
            self.top_k = int(re.match(r"top(\d+)", config.moe_type).group(1))
        elif config.moe_type == "dense":
            self.moe_type = config.moe_type
            modules = [LlamaMLP(config) for _ in range(self.num_experts)]
        else:
            raise ValueError()
        self.experts = nn.ModuleList(modules)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.moe_type == "topk":
            hidden_states = hidden_states.view(-1, hidden_dim)
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
            router_logits = self.router(hidden_states)
            routing_weights = F.softmax(router_logits, dim=2)
            layer_outputs = torch.stack(
                [ex(hidden_states) for ex in self.experts], dim=2
            )
            layer_output = torch.sum(
                routing_weights.unsqueeze(-1) * layer_outputs, dim=2
            )

        hidden_states = residual + layer_output

        outputs = (hidden_states, router_logits)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaMoEPreTrainedModel(LlamaPreTrainedModel):
    _no_split_modules = ["LlamaMoEDecoderLayer"]


class LlamaMoEModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.front_frozen_layers = config.front_frozen_layers
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(self.front_frozen_layers)
            ]
            + [
                LlamaMoEDecoderLayer(config, layer_idx)
                for layer_idx in range(
                    self.front_frozen_layers, config.num_hidden_layers
                )
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "cache_position is a required argument when using StaticCache."
                )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = ()
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if i < self.front_frozen_layers:
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            else:
                router_logits = layer_outputs[1]

                if use_cache:
                    next_decoder_cache = layer_outputs[3 if output_attentions else 2]

                if output_attentions:
                    all_self_attns += (layer_outputs[2],)

                all_router_logits = all_router_logits + (router_logits,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        # if not return_dict:
        return (
            hidden_states,
            all_router_logits,
            next_cache,
            all_hidden_states,
            all_self_attns,
        )
        # return tuple(
        #     v
        #     for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
        #     if v is not None
        # )
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )


class LlamaMoEForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        router_logits = outputs[1]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if self.use_router_loss:
                aux_loss = load_balancing_loss_func(
                    router_logits, self.num_experts, self.top_k, attention_mask
                )
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs[2] if use_cache else None,
            hidden_states=outputs[3] if output_hidden_states else None,
            attentions=outputs[4] if output_attentions else None,
        )
