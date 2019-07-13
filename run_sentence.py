import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from keras_models.blstm_2dcnn import BLSTM2DCNN
from keras_models.lstm import LSTMClassifier
from keras_models.mlp import MLP
from keras_models.ykim_cnn import YKimCNN
from sklearn_models import MultNB, SVM
from stacking_classifier import StackingTextClassifier
from benchmarks import benchmark

epoch = 50

models = [
    (MLP, {'layers': 1, 'units': 360, 'epochs': 50, 'max_vocab_size': 50000}, "MLP 1x360"),
    (MLP, {'layers': 2, 'units': 180, 'epochs': 50, 'max_vocab_size': 50000}, "MLP 2x180"),
    (MLP, {'layers': 3, 'epochs': epoch}, "MLP 3x512"),
    (LSTMClassifier, {
        'max_seq_len': 1000,
        'layers': 3,
        'rec_dropout_rate': 0.35,
        'optimizer': 'adam',
        'embedding_dim': 24,
        'epochs': epoch,
        'bidirectional': False,
        'units': 250
    }, "LSTM 24D"),
    (LSTMClassifier, {
        'max_seq_len': 1000,
        'layers': 2,
        'rec_dropout_rate': 0.4,
        'optimizer': 'rmsprop',
        'embedding_dim': 12,
        'epochs': epoch,
        'bidirectional': False,
        'units': 80
    }, "LSTM 12D"),
    (LSTMClassifier, {
        'max_seq_len': 1000,
        'layers': 2,
        'rec_dropout_rate': 0.5,
        'optimizer': 'rmsprop',
        'embeddings_path': '/home/lekan/nass/glove/glove.6B.300d.txt',
        'epochs': 42,
        'bidirectional': True,
        'units': epoch
    }, "BLSTM GloVe"),
    (LSTMClassifier, {
        'max_seq_len': 1000,
        'layers': 2,
        'rec_dropout_rate': 0.5,
        'optimizer': 'rmsprop',
        'embeddings_path': '/home/lekan/nass/glove/glove.6B.300d.txt',
        'epochs': epoch,
        'bidirectional': False,
        'units': 32
    }, "LSTM GloVe"),
    (YKimCNN, {
        'max_seq_len': 1000,
        'filter_sizes': (3, 5, 7),
        'num_filters': 5,
        'embedding_dim': 45,
        'dropout_rates': (0.25, 0.47),
        'units': epoch,
        'epochs': 53,
        'batch_size': 128
    }, "CNN 45D"),
    (YKimCNN, {
        'max_seq_len': 1000,
        'filter_sizes': (3, 5),
        'num_filters': 75,
        'embeddings_path': '/home/lekan/nass/glove/glove.6B.300d.txt',
        'dropout_rates': (0.2, 0.8),
        'units': epoch,
        'epochs': 33,
        'batch_size': 128
    }, "CNN GloVe"),
    (BLSTM2DCNN, {
        'max_seq_len': 1000,
        'rec_dropout_rate': 0.88,
        'optimizer': 'rmsprop',
        'embeddings_path': '/home/lekan/nass/glove/glove.6B.300d.txt',
        'units': 8,
        'conv_filters': 32,
        'epochs': epoch,
        'batch_size': 64
    }, "BLSTM2DCNN GloVe"),
    (BLSTM2DCNN, {
        'max_seq_len': 1000,
        'rec_dropout_rate': 0.75,
        'optimizer': 'adam',
        'embedding_dim': 15,
        'units': 162,
        'conv_filters': 32,
        'epochs': epoch,
        'batch_size': 128
    }, "BLSTM2DCNN 15D"),
    (MultNB, {'tfidf': True}, "MNB tfidf"),
    (MultNB, {'tfidf': True, 'ngram_n': 2}, "MNB tfidf 2-gr"),
    (MultNB, {'tfidf': True, 'ngram_n': 3}, "MNB tfidf 3-gr"),
    (MultNB, {'tfidf': False}, "MNB"),
    (MultNB, {'tfidf': False, 'ngram_n': 2}, "MNB 2-gr"),
    (SVM, {'tfidf': True, 'kernel': 'linear'}, "SVM tfidf"),
    (SVM, {'tfidf': True, 'kernel': 'linear', 'ngram_n': 2}, "SVM tfidf 2-gr"),
    (SVM, {'tfidf': False, 'kernel': 'linear'}, "SVM")
]


logreg_stacker = (StackingTextClassifier, {
    'stacker': (LogisticRegression, {}),
    'base_classifiers': [
        (m, params)
        for m, params, _ in models[:-3]
    ] + [
        (m, dict(list(params.items()) + [('probability', True)]))
        for m, params, _ in models[-3:]
    ],
    'use_proba': True,
    'folds': 5
}, "Stacker LogReg")

xgb_stacker = (StackingTextClassifier, {
    'stacker': (XGBClassifier, {}),
    'base_classifiers': [(m, p) for m, p, _ in models],
    'use_proba': False,
    'folds': 5
}, "Stacker XGB")

models.append(logreg_stacker)
models.append(xgb_stacker)

results_path = 'sentence_results.csv'

if __name__ == '__main__':
    records = []
    for model_class, params, model_name in models:
            scores = benchmark(model_class, params, name=model_name)
            model_str = str(model_class(**params))
            print('F1 : %.3f' % np.mean(scores), model_str)
            records.append({
                    'model': model_str,
                    'f1': scores,
                    'model_name': model_name
                })

    pd.DataFrame(records).to_csv(results_path, index=False)
