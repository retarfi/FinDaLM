from transformers import LlamaPreTrainedModel, LlamaTokenizerFast


def set_pad_token_to_tokenizer(tokenizer: LlamaTokenizerFast) -> None:
    if tokenizer.pad_token is not None:
        pass
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_250|>"})
    tokenizer.padding_side = "left"


def set_pad_token_to_model(
    model: LlamaPreTrainedModel, tokenizer: LlamaTokenizerFast
) -> None:
    model.config.pad_token_id = tokenizer.pad_token_id
