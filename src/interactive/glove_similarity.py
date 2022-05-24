#%%
from abc import ABC
from dataclasses import dataclass

import gensim
import pandas as pd


class GloveExperiment:
    def __init__(self) -> None:
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            "~/Downloads/glove.6B.300d.txt", binary=False, no_header=True
        )

    def run(self):

        demo_words = ["stock", "buy", "money", "company"]

        data = {}

        for word in demo_words:
            most_similar = [t[0] for t in self.word_vectors.similar_by_key(word)[:5]]
            data.update({word: most_similar})

        for k, v in data.items():
            print(f"\\textbf{{{k}}} & {' & '.join(v)}" + "\\\\")


class Word2VecExperiment:
    def __init__(self) -> None:
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            "~/Downloads/GoogleNews-vectors-negative300.bin", binary=True
        )

    def get_closest(self, word: str, n: int = 5) -> list[str]:
        return [t[0] for t in self.word_vectors.similar_by_key(word)[:n]]

    def run(self):

        demo_words = ["stock", "buy", "money", "company"]

        data = {}

        for word in demo_words:
            most_similar = [t[0] for t in self.word_vectors.similar_by_key(word)[:5]]
            data.update({word: most_similar})
        print(data)
        for k, v in data.items():
            print(f"\\textbf{{{k}}} & {' & '.join(v)}" + "\\\\")


#%%
word2vec = Word2VecExperiment()

#%%
word2vec.run()
