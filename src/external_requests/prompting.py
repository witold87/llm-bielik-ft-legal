class PromptBuilder:
    @staticmethod
    def generate_questions(n_questions: int, text: str, tokens: int) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are question curious lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Based on given text here: {text} please generate a {n_questions} questions. Use polish language. Start with "-". Use up to {tokens} tokens. '
            }
        ]

    @staticmethod
    def get_answers_based_on_text_and_questions(text: str, question: str, tokens: int) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are question curious lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'With given text here: {text} and {question} please give me the answer to the given question. Use polish language. Use up to {tokens} tokens.'
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
    def get_coherence_pl(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś prawnikiem, który stara sie zrozumieć zagadnienia i niuanse prawa i umów.',
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 spójność tekstu i wyjaśnić dlaczego tak oceniłeś. Oto text {text}. Zwróć JSONA z polami: score oraz explanation. Odpowiedź ma być po polsku.'
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

    @staticmethod
    def get_accuracy(question: str, answer: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czy dana odpowiedź jest trafna i precyzyjna wzgledem zadanego pytania, gdzie 5 oznacza ze jest wysoce precyzyjna a 1 znaczy, że w ogóle nie jest precyzyjna.'
                           f'Oto pytanie: {question} a to jest {answer}. Zwróć JSONA z polami: score oraz explanation. Odpowiedź ma być po polsku. '
            }
        ]

    @staticmethod
    def get_relevance_pl(question: str, context: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś prawnikiem, który stara sie zrozumieć zagadnienia i niuanse prawa i umów.'
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 if the provided question could be asked by a lawyer then rate 5, and rate 1 if not based on the question and context. '
                           f'Here is the question: {question} and context {context}. Return only value of the rating.'
            }
        ]

    @staticmethod
    def get_global_relevance(question: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 if the provided question could be asked by a lawyer then rate 5, and rate 1 if not based on the question. Here is the question: {question}. Return only value of the rating.'
            }
        ]

    @staticmethod
    def get_clarity_pl(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś prawnikiem, który stara sie zrozumieć zagadnienia i niuanse prawa i umów.',
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czytelność, jasność tekstu i wyjaśnić dlaczego tak oceniłeś. Oto text {text}. Zwróć JSONA z polami: score oraz explanation. Odpowiedź ma być po polsku.'
            }
        ]