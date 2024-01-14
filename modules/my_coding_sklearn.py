from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

import gensim
from gensim.models import doc2vec


class Doc2VecCoding(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=10, min_count=2, epochs=30):
        self.epochs = epochs
        self.vector_size = vector_size
        self.min_count = min_count
        self.d2v_model = doc2vec.Doc2Vec(vector_size=self.vector_size,  # длина результирующего вектора
                                         min_count=self.min_count,  # min кол-во встречания слова в прпедложении для учета
                                         epochs=self.epochs,  # количество эпох
                                         )

    def tagged_document(self, list_of_ListOfWords):
        for num, text in enumerate(list_of_ListOfWords):
            yield doc2vec.TaggedDocument(str(text).split(), [num])

    def fit(self, X, y=None):
        X = list(self.tagged_document(X))
        self.d2v_model.build_vocab(X)
        self.d2v_model.train(X,
                             total_examples=self.d2v_model.corpus_count,
                             epochs=self.epochs,
                             )
        return self

    def transform(self, X):
        Xprim = np.array([self.d2v_model.infer_vector([text_x]) for text_x in X])
        return Xprim


if __name__ == '__main__':
    # df = pd.read_csv('./data/sentences.csv')['sentence'].iloc[:200]
    # transf = Doc2VecCoding()
    # a = transf.fit_transform(df)
    # print(df[0], '\n', a[0])
    df = pd.read_csv('../data/sentences.csv')
    print("Все")