import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(model_id: str, config: dict,
               tokenizer: str = None) -> tuple:
    bnb_config = BitsAndBytesConfig(**config)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    if tokenizer is not None:
        print(f'Loading custom tokenizer: {tokenizer}')
        model_id = tokenizer

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    model_tokenizer.pad_token = model_tokenizer.eos_token
    return model, model_tokenizer