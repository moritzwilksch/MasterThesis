#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = ["researcher in academia", "academic research"]
word_cv = CountVectorizer()
subword_cv = CountVectorizer(analyzer="char_wb", ngram_range=(7, 7))


def demo(cv: CountVectorizer, texts: list):
    print(f"{' Matrix ':-^40}")
    mtx = cv.fit_transform(texts).toarray()
    vocab = cv.vocabulary_
    vocab = {v: k for k, v in vocab.items()}
    print([vocab.get(idx) for idx in range(len(vocab))])
    print(mtx)
    print(f"{' LaTeX ':-^40}")
    print(" & ".join([str(s) for s in mtx[0, :]]))
    print(" & ".join([str(s) for s in mtx[1, :]]))
    print(f"{' Cosine Sim ':-^40}")
    print(cosine_similarity(mtx[0, :].reshape(1, -1), mtx[1, :].reshape(1, -1)))

    print("=" * 40)


demo(word_cv, texts)
demo(subword_cv, texts)

#%%
