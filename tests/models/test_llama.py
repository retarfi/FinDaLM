from transformers import LlamaForCausalLM, LlamaTokenizerFast

from findalm.models.llama import set_pad_token_to_model, set_pad_token_to_tokenizer

LLAMA_MODEL_NAME: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM"


def test_add_pad_token() -> None:
    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    set_pad_token_to_tokenizer(tokenizer)


def test_set_pad_token_to_model() -> None:
    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    set_pad_token_to_tokenizer(tokenizer)
    model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME)
    set_pad_token_to_model(model=model, tokenizer=tokenizer)
