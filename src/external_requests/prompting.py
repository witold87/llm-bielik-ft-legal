class PromptBuilder:
    @staticmethod
    def generate_questions(n_questions: int, text: str, tokens: int) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś dociekliwym asystentem prawniczym chcącym poznać szczegóły ustawy',
            },
            {
                'role': 'user',
                'content': f'Bazując na podanym tekście: {text} wygeneruj proszę {n_questions} pytań. Użyj języka polskiego. Rozpocznij pytania od "-". Użyj maksymalnie {tokens} tokenów. '
            }
        ]

    @staticmethod
    def get_answers_based_on_text_and_questions(text: str, question: str, tokens: int) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś pomocnym asystentem prawniczym, który konkretnie i zwięźle odpowiada na zadane pytania.',
            },
            {
                'role': 'user',
                'content': f'Bazując na podanym tekście: {text} odpowiedz proszę na zadane pytanie: {question}. Użyj języka polskiego. Użyj maksymalnie {tokens} tokenów.'
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
                'content': 'Jesteś krytykiem prawniczych tekstów, dokładnie analizujesz tekst i wyciągasz wnioski.',
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czy pytanie związane jest z podanym kontekstem. Oto kontekst: {context}, a tutaj pytanie: {question}. Zwróc tylko wartość oceny.'
            }
        ]

    @staticmethod
    def get_accuracy_pl(question: str, answer: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś krytykiem prawniczych tekstów, dokładnie analizujesz tekst i wyciągasz wnioski.',
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
                'content': 'Jesteś krytykiem prawniczych tekstów, dokładnie analizujesz tekst i wyciągasz wnioski.'
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czy podane pytanie: {question} mogłoby być zadane przez prawnika biorąc pod uwagę kontekst {context}. Zwróć JSONA z polami: score oraz explanation.'
            }
        ]

    @staticmethod
    def get_global_relevance(question: str) -> list:
        return [
            {
                'role': 'system',
                'content': 'Jesteś krytykiem prawniczych tekstów, dokładnie analizujesz tekst i wyciągasz wnioski.',
            },
            {
                'role': 'user',
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czy podane pytanie: {question} mogłoby być zadane przez prawnika. Zwróc tylko wartość oceny.'
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
                'content': f'Twoim zadaniem jest ocenić w skali od 1 do 5 czytelność, jasność tekstu i wyjaśnić dlaczego tak oceniłeś. Oto tekst {text}. Zwróć JSONA z polami: score oraz explanation. Odpowiedź ma być po polsku.'
            }
        ]