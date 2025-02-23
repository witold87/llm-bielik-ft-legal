import datasets
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt


def format_for_chat_template(data: pd.DataFrame) -> list:
    messages = [
        {"role": "system", "content": "Jesteś asystentem prawniczym. Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
        {"role": "user", "content": data['question']},
        {"role": "assistant", "content": data['answer']}
    ]
    return messages

def generate_with_template(tokenizer, prompt):
    return tokenizer.apply_chat_template(format_for_chat_template(prompt), return_dict=True)

def generate_and_tokenize_with_template(length: int, tokenizer, prompt):
    result = tokenizer.apply_chat_template(format_for_chat_template(prompt),
        truncation=True,
        max_length=length,
        return_dict=True,
        padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result


def load_data(path: str,
              type: str = 'train',
              format: str = 'json') -> datasets.Dataset:
    _dataset = load_dataset(format, data_files=path, split=type)
    return _dataset

def plot_hist(tokenized_train_dataset, tokenized_eval_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_eval_dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='cyan')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()