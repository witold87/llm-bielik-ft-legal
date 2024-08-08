from src.external_requests.openai_req import OpenAIReq
from src.external_requests.prompting import PromptBuilder


class LLMBasedEvaluator:

    def __init__(self, openai_req: OpenAIReq):
        self.openai_req = openai_req

    def evaluate_coverage(self, question: str, context: str) -> str:
        prompt = PromptBuilder.get_coverage(question=question, context=context)
        return self.openai_req.call_api(prompt=prompt)

    def evaluate_coherence(self, question: str) -> str:
        prompt = PromptBuilder.get_coherence(question=question)
        return self.openai_req.call_api(prompt=prompt)
