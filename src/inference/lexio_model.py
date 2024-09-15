import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def build_from_checkpoint(model_name: str = 'speakleash/Bielik-7B-v0.1',
                          checkpoint: str = '',
                          return_base: bool = False):
    bnb_config = BitsAndBytesConfig(
        # llm_int8_enable_fp32_cpu_offload=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True)

    if not return_base:
        # "bielik_v2-legal-finetune/checkpoint-500"
        model = PeftModel.from_pretrained(model, checkpoint)

    return model, eval_tokenizer


model = build_from_checkpoint(checkpoint='src/bielik-legal-finetune/checkpoint-400')


class LexioModel:

    @staticmethod
    def get_model():
        return model

    @staticmethod
    def build_prompt(context: list, question: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": "Jesteś prawniczym asystentem, ktory precyzyjnie odpowiada na pytania użytkowników",
            },
            {
                "role": "user",
                "content": f"Mając poniższy kontekst {context} odpowiedz proszę na pytanie: {question}"
            }

        ]
        return messages

    def run_inference(self, query: str, context: list):
        loaded_model, tokenizer = self.get_model()
        model_input = tokenizer.apply_chat_template(self.build_prompt(question=query, context=context), return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = loaded_model.generate(model_input, max_new_tokens=100, do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id)

            decoded = tokenizer.batch_decode(generated_ids)


        # Evaluated model
        looked_word = '[/INST]'
        inst_end = decoded[0].index(looked_word)
        answer = decoded[0][inst_end + len(looked_word):]
        return answer