import numpy as np
from joblib import Memory
from keras import backend as K

cache = Memory('cache').cache


@cache
def get_embedding_dim(embedding_path):
    with open(embedding_path, 'rb') as f:
        return len(f.readline().split()) - 1


@cache
def get_embedding_matrix(vocab, embedding_path):
    word2ind = {w: i for i, w in enumerate(vocab)}
    embedding_dim = get_embedding_dim(embedding_path)
    embeddings = np.random.normal(size=(len(vocab), embedding_dim))

    with open(embedding_path, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2ind:
                i = word2ind[word]
                vec = np.array([float(x) for x in parts[1:]])
                embeddings[i] = vec
    return embeddings


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
