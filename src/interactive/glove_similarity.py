#%%
import gensim
import pandas as pd

#%%
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
    "~/Downloads/glove.6B.300d.txt", binary=False, no_header=True
)

#%%
demo_words = ["stock", "buy", "money", "company"]

data = {}

for word in demo_words:
    most_similar = [t[0] for t in word_vectors.similar_by_key(word)[:5]]
    data.update({word: most_similar})


#%%
for k, v in data.items():
    print(f"\\textbf{{{k}}} & {' & '.join(v)}" + "\\\\")
