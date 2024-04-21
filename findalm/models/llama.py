from transformers import LlamaPreTrainedModel, LlamaTokenizerFast


def set_pad_token_to_tokenizer(tokenizer: LlamaTokenizerFast) -> None:
    tokenizer.pad_token_id = tokenizer.unk_token_id


def set_pad_token_to_model(
    model: LlamaPreTrainedModel, tokenizer: LlamaTokenizerFast
) -> None:
    model.config.pad_token_id = tokenizer.pad_token_id
