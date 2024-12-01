from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(model_name: str,
             model: AutoModelForCausalLM,
             text: str) -> None:

    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
    )

    model_input = eval_tokenizer(text, return_tensors="pt").to("cuda")

    model.eval() # required for doing inference, not training.
    with torch.no_grad():
        response = eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True)

    return response
