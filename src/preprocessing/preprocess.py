def clean_text(x) -> str:
    text = x.replace('\n', '')
    return text


def remove_stopwords(sentence):
    with open('utils/polish.stopwords.txt') as file:
        stopwords = file.readlines()
    stopwords = [word.replace('\n', '') for word in stopwords]
    return [w for w in sentence.lower().split() if w not in stopwords]