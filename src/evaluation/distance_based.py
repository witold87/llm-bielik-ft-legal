from typing import Union

from gensim.models import KeyedVectors


class DistanceEvaluator:
    def __init__(self, model_path: Union[str, None] = None):
        self.model_path = 'models/word2vec/word2vec_100_3_polish.bin' if model_path is None else model_path
        self.model = KeyedVectors.load(self.model_path)

    def get_diversity_by_wmd(self, value_1: str, value_2: str) -> float:
        return self.model.wmdistance(value_1, value_2)
