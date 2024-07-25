


def get_covarage_by_gpt(question, context):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=200,
        temperature=0.0,
        top_p=1,
        messages=[
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 if the question can be extracted from the context. Here is the context {context}, question: {question}. Return only value of the rating.'
            }
        ]
    )
    response = completion.choices[0].message.content
    return response

def get_coherence(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=200,
        temperature=0.0,
        top_p=1,
        messages=[
            {
                'role': 'system',
                'content': 'You are an expert lawyer who want to get insights about law',
            },
            {
                'role': 'user',
                'content': f'Your task is to rate from 1 to 5 the coherence (fluency) of the question and provide explanation. Here is the question: {question}. Return only value of the rating and explanation in tuple format: rating | explanation'
            }
        ]
    )
    response = completion.choices[0].message.content
    return response