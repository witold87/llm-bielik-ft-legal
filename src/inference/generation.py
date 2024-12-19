from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(model_name: str,
             model: AutoModelForCausalLM,
             text: str, generation_params: dict) -> None:

    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
    )

    max_new_tokens = generation_params.get('max_new_tokens', 256)
    sampling = generation_params.get('sampling', False)
    n_beams = generation_params.get('n_beams', 1) # do_sample=True and num_beans = 1 -> multinomial sampling
    top_k = generation_params.get('top_k', 50)
    model_input = eval_tokenizer(text, return_tensors="pt").to("cuda")

    model.eval() # required for doing inference, not training.
    with torch.no_grad():
        response = eval_tokenizer.decode(model.generate(**model_input,
                                                        do_sample=sampling,
                                                        top_k=top_k,
                                                        num_beams=n_beams,
                                                        max_new_tokens=max_new_tokens,
                                                        repetition_penalty=1.15)[0], skip_special_tokens=True)

    return response
