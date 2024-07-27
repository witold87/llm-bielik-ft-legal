import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.distance_based import DistanceEvaluator as de


class Reporting:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_similarity_matrix_for_wmd(self, on_column: str):
        similarities = {}
        for index2, row2 in tqdm(self.data.iterrows()):
            for index1, row1 in self.data.iterrows():
                sim_score = de.get_diversity_by_wmd(row2[on_column], row1[on_column])
                q1 = row2[on_column]
                q2 = row1[on_column]
                questions = f'{q1} | {q2}'
                similarities[questions] = {'wmd': sim_score}
        sim_df = pd.DataFrame.from_dict(similarities)
        results = sim_df.T.reset_index().rename(columns={'index': 'question'})
        results[['question', 'compared_to_question']] = results['question'].str.split('|', expand=True)
        sim_heatmap = results.pivot(index='question', columns='compared_to_question', values='wmd')
        return sim_heatmap

    @staticmethod
    def get_summary(data: pd.DataFrame) -> tuple:
        ltri = np.tril(data.values, -1)
        ltri = ltri[np.nonzero(ltri)]
        return ltri.std(), ltri.mean()
