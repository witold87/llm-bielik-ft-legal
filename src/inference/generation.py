from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import  json


def generate(model_name: str,
             model: AutoModelForCausalLM,
             text: str, generation_params: dict) -> None:

    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
    )

    max_new_tokens = generation_params.get('max_new_tokens', 256)
    sampling = generation_params.get('sampling', False)
    n_beams = generation_params.get('n_beams', 1)
    top_k = generation_params.get('top_k', 50)
    temperature = generation_params.get('temperature', None)
    top_p = generation_params.get('top_p', 0.95)
    num_return_sequences = generation_params.get('num_return_sequences', 1)
    repetition_penalty = generation_params.get('repetition_penalty', 1.15)


    model_input = eval_tokenizer(text, return_tensors="pt").to("cuda")

    model.eval() # required for doing inference, not training.

    with torch.no_grad():
        response = eval_tokenizer.decode(model.generate(**model_input,
                                                        do_sample=sampling,
                                                        top_k=top_k,
                                                        num_beams=n_beams,
                                                        top_p=top_p,
                                                        temperature=temperature,
                                                        num_return_sequences=num_return_sequences,
                                                        repetition_penalty=repetition_penalty,
                                                        max_new_tokens=max_new_tokens)[0], skip_special_tokens=True)

    return response


def generate_with_chat_template(question, eval_model, eval_tokenizer):
    template_text = [
        {"role": "system",
         "content": "Jesteś asystentem prawniczym. Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
        {"role": "user", "content": f"{question}"}
    ]

    device = "cuda"
    input_ids = eval_tokenizer.apply_chat_template(template_text, return_tensors="pt")
    model_inputs = input_ids.to(device)
    generated_ids = eval_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = eval_tokenizer.batch_decode(generated_ids)
    return decoded[0]


def pretty_output(raw_answer: str) -> str:
    searched_token = '<|im_end|>'
    last = raw_answer.rfind(searched_token)
    raw_answer = raw_answer[:last]
    second_last = raw_answer.rfind(searched_token)
    return raw_answer[second_last+len(searched_token):]


def get_score(row: str) -> int:
    result = json.loads(row)
    score = result.get('score')
    return score

def get_explanation(row: str) -> str:
    result = json.loads(row)
    explanation = result.get('explanation')
    return explanation