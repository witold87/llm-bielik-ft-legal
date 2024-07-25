import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv('openai_api_key'))


# gpt-3.5-turbo"
class OpenAIReq:

    def __init__(self, model: str, parameters: dict) -> None:
        self.model: Optional[str] = model
        self.parameters: dict = parameters

    def generate_questions_based_on_text(self, text: str, n_questions: int) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            temperature=0.2,
            top_p=1,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are question curious lawyer who want to get insights about law',
                },
                {
                    'role': 'user',
                    'content': f'Based on given text here: {text} please generate a {n_questions} questions. Use polish language. Start with "-". '
                }
            ]
        )
        response = completion.choices[0].message.content
        return response

    def get_answers_based_on_text_and_questions(self, text: str, question: str) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            temperature=0.2,
            top_p=1,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are question curious lawyer who want to get insights about law',
                },
                {
                    'role': 'user',
                    'content': f'With given text here: {text} and {question} please give me the answer to the given question. Use polish language.'
                }
            ]
        )
        response = completion.choices[0].message.content
        return response

