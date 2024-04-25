from transformers import AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerBase

from .llama import set_pad_token_to_tokenizer


def load_tokenizer(model_name: str, is_llama: bool) -> PreTrainedTokenizerBase:
    if is_llama:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        set_pad_token_to_tokenizer(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
