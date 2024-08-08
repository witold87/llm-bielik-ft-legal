import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.distance_based import DistanceEvaluator


class Reporting:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_similarity_matrix_for_wmd(self, on_column: str):
        distance_evaluator = DistanceEvaluator()
        similarities = {}
        for index2, row2 in tqdm(self.data.iterrows()):
            for index1, row1 in self.data.iterrows():
                sim_score = distance_evaluator.get_diversity_by_wmd(value_1=row2[on_column], value_2=row1[on_column])
                q1 = row2[on_column]
                q2 = row1[on_column]
                questions = f'{q1} | {q2}'
                similarities[questions] = {f'wmd_{on_column}': sim_score}
        sim_df = pd.DataFrame.from_dict(similarities)
        results = sim_df.T.reset_index().rename(columns={'index': 'question'})
        results[['question', 'compared_to_question']] = results['question'].str.split('|', expand=True)
        sim_heatmap = results.pivot(index='question', columns='compared_to_question', values=f'wmd_{on_column}')
        return sim_heatmap

    @staticmethod
    def get_summary_for_wmd(data: pd.DataFrame) -> tuple:
        ltri = np.tril(data.values, -1)
        ltri = ltri[np.nonzero(ltri)]
        return ltri.std(), ltri.mean()
