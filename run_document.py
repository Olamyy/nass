import numpy as np
import pandas as pd

from keras_models.fchollet_cnn import FCholletCNN
from keras_models.mlp import MLP
from sklearn_models import MultNB, BernNB, SVM, RandomForest, LinearSVM
from benchmarks import benchmark

epoch = 50

models = [
    (FCholletCNN, {'embedding_dim': 200, 'units': 400, 'epochs': epoch}, "CNN 37D"),
    (FCholletCNN, {'epochs': epoch, 'units': 400,
                   'embeddings_path': '/Users/Olamilekan/Desktop/Machine Learning/OpenSource/nass-ai/models/glove/glove.6B.300d.txt'}, "CNN GloVe"),
    (MLP, {'layers': 1, 'units': 360, 'epochs': epoch, 'max_vocab_size': 22000}, "MLP 1x360"),
    (MLP, {'layers': 2, 'units': 180, 'epochs': epoch, 'max_vocab_size': 22000}, "MLP 2x180"),
    (MLP, {'layers': 3,  'epochs': epoch}, "MLP 3x512"),
    (MultNB, {'tfidf': True}, "MNB tfidf"),
    (MultNB, {'tfidf': True, 'ngram_n': 2}, "MNB tfidf 2-gr"),
    (MultNB, {'tfidf': True, 'ngram_n': 3}, "MNB tfidf 3-gr"),
    (BernNB, {'tfidf': True}, "BNB tfidf"),
    (MultNB, {'tfidf': False}, "MNB"),
    (MultNB, {'tfidf': False, 'ngram_n': 2}, "MNB 2-gr"),
    (BernNB, {'tfidf': False}, "BNB"),
    (SVM, {'tfidf': True, 'kernel': 'linear'}, "SVM tfidf"),
    (SVM, {'tfidf': True, 'kernel': 'linear', 'ngram_n': 2}, "SVM tfidf 2-gr"),
    (SVM, {'tfidf': False, 'kernel': 'linear'}, "SVM"),
    (SVM, {'tfidf': False, 'kernel': 'linear', 'ngram_n': 2}, "SVM 2-gr"),
    (RandomForest, {'tfidf': True, 'ngram_n': 2}, "Random"),
    (RandomForest, {'tfidf': False, 'ngram_n': 1}, "Random TF"),
    (LinearSVM, {'tfidf': True, 'ngram_n': 2}, "LinearSVM TF"),
    (LinearSVM, {'tfidf': False, 'ngram_n': 2}, "LinearSVM"),
]

results_path = 'document_results.csv'

if __name__ == '__main__':
    records = []
    for model_class, params, model_name in models:
        scores, times = benchmark(model_class, params, name=model_name)
        model_str = str(model_class(**params))
        print('F1 %.3f' % np.mean(scores), model_str)
        records.append({
            'model': model_str,
            'f1': scores,
            'model_name': model_name
        })

    pd.DataFrame(records).to_csv(results_path, index=False)
