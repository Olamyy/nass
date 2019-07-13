from collections import Counter

import numpy
import numpy as np
from time import time

import pandas
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from joblib import Memory
from sklearn.preprocessing import LabelEncoder


def star(f):
    return lambda args: f(*args)


def prepare_data():
    # data = pandas.read_csv('/home/lekan/nass/clean_data.csv')
    data = pandas.read_csv('/Users/Olamilekan/Desktop/Machine Learning/OpenSource/nass-ai/data/clean_data.csv')
    text = data.clean_text
    labels = data.bill_class
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    word_counts = Counter(w for text in text for w in text)
    vocab = [''] + [w for (w, _) in sorted(word_counts.items(), key=star(lambda _, c: -c))]
    word2ind = {w: i for i, w in enumerate(vocab)}
    X = np.array([[word2ind[w] for w in tokens] for tokens in text])
    return X, labels, vocab, label_encoder


cache = Memory('cache').cache


@cache
def benchmark(model_class, model_params=None, name=None):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :param iters: how many times to benchmark
    :param return_time: if true, returns list of running times in addition to scores
    :return: tuple (accuracy scores, running times)
    """

    def run_validation(clf, train, y_train):
        scoring = {'f1_micro': 'f1_macro'}
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        accuracies = cross_validate(clf, train, y_train, scoring=scoring, cv=cv)
        print(accuracies)
        return True

    if model_params is None:
        model_params = {}

    X, y, vocab, label_encoder = prepare_data()
    class_count = len(label_encoder.classes_)
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab
    model_params['class_count'] = class_count

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_class(**model_params)
    preds = model.fit(X_train, y_train).predict(X_test)
    score = f1_score(preds, y_test, average='macro')
    run_validation(model, X_train, y_train)
    numpy.savez_compressed('test_and_pred_{name}.npz', test=X_test, predictions=preds)
    return score


@cache
def benchmark_with_early_stopping(model_class, model_params=None):
    """same as benchmark but fits with validation data to allow the model to do early stopping
    Works with all models from keras_models
    :param model_class: class of the model to instantiate, must have fit(X, y, validation_data)
        method and 'history' attribute
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :return: best_loss, best_score, best_epoch
    """
    if model_params is None:
        model_params = {}

    X, y, vocab, label_encoder = prepare_data()
    class_count = len(label_encoder.classes_)
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab
    model_params['class_count'] = class_count
    model = model_class(**model_params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model.fit(X_train, y_train, validation_data=[X_test, y_test])
    best_loss = np.min(model.history.history['val_loss'])
    best_acc = np.max(model.history.history['val_acc'])
    best_epoch = np.argmin(model.history.history['val_loss']) + 1

    print(model, "acc", best_acc, "loss", best_loss, "epochs", best_epoch)
    return best_loss, best_acc, best_epoch
