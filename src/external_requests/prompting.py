class PromptBuilder:
    @staticmethod
    def generate_questions(n_questions: int, text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are question curious lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Based on given text here: {text} please generate a {n_questions} questions. Use polish language. Start with "-". '
            }
        ]

    @staticmethod
    def get_answers_based_on_text_and_questions(text: str, question: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are question curious lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'With given text here: {text} and {question} please give me the answer to the given question. Use polish language.'
            }
        ]

    @staticmethod
    def get_coherence(question: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 the coherence (fluency) of the question and provide explanation. Here is the question: {question}. Return only value of the rating and explanation in tuple format: rating | explanation'
            }
        ]

    @staticmethod
    def get_coverage(question: str, context: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 if the question can be extracted from the context. Here is the context {context}, question: {question}. Return only value of the rating.'
            }
        ]
