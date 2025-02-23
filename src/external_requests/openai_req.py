import os
from typing import Union
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv('openai_api_key'))


class OpenAIReq:
    """
    Base support class for making a calls to OpenAI API.
    """

    def __init__(self, model: str = None) -> None:
        self.model: Union[str, None] = 'gpt-3.5-turbo' if model is None else model

    def call_api(self, prompt: list, **model_params) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            max_tokens=model_params.get('max_tokens', 4096),
            temperature=model_params.get('temperature', 0.2),
            top_p=model_params.get('top_p', 1),
            messages=prompt
        )
        response = completion.choices[0].message.content
        return response

    def call_api_formatted(self, prompt: list, **model_params) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            max_tokens=model_params.get('max_tokens', 4096),
            temperature=model_params.get('temperature', 0.2),
            top_p=model_params.get('top_p', 1),
            messages=prompt
        )
        response = completion.choices[0].message.content
        return response