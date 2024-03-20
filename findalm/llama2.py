from transformers import LlamaPreTrainedModel, LlamaTokenizerFast


def add_pad_token(tokenizer: LlamaTokenizerFast) -> None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})


def set_pad_token_to_model(
    model: LlamaPreTrainedModel, tokenizer: LlamaTokenizerFast
) -> None:
    model.config.pad_token_id = tokenizer.pad_token_id
