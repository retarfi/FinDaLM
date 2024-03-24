from findalm.models.llama import add_pad_token, set_pad_token_to_model
from transformers import LlamaForCausalLM, LlamaTokenizerFast

LLAMA_MODEL_NAME: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM"


def test_add_pad_token() -> None:
    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    add_pad_token(tokenizer)


def test_set_pad_token_to_model() -> None:
    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    add_pad_token(tokenizer)
    model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME)
    set_pad_token_to_model(model=model, tokenizer=tokenizer)
