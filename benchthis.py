import numpy
import numpy as np

import pandas
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from keras_models.utils import MeanEmbeddingVectorizer

GLOVE = '/home/lekan/nass/glove/glove.6B.300d.txt'
# GLOVE = '/Users/Olamilekan/Desktop/Machine Learning/OpenSource/nass-ai/models/glove/glove.6B.300d.txt'


def star(f):
    return lambda args: f(*args)


def prepare_data():
    data = pandas.read_csv('/home/lekan/nass/clean_data.csv')
    # data = pandas.read_csv('/Users/Olamilekan/Desktop/Machine Learning/OpenSource/nass-ai/data/clean_data.csv')
    text = data.clean_text
    labels = data.bill_class
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    tok = Tokenizer(num_words=30000)
    tok.fit_on_texts(text)
    word_counts = tok.word_counts
    vocab = [''] + [w for (w, _) in sorted(word_counts.items(), key=star(lambda _, c: -c))]

    embeddings_index = {}
    with open(GLOVE) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = numpy.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    return text, labels, vocab, embeddings_index


cache = Memory('cache').cache


def show_report(y_test, y_pred):
    print(metrics.classification_report(y_test, y_pred))
    print()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    print("Average Accuracy : {}".format(accuracy))
    print("Average F1 : {}".format(f1))

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

    if model_params is None:
        model_params = {}

    print(model_params)
    text, labels, vocab, embeddings_index = prepare_data()
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab

    X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42)
    vectorizer_class = TfidfVectorizer
    vectorizer = vectorizer_class(
        tokenizer=lambda x: x,
        min_df=0.2
    )
    model = SVC(C=10, gamma=0.001, kernel='linear')
    clf = Pipeline([('vectorizer', MeanEmbeddingVectorizer(word2vec=embeddings_index, dim=300)), ('model', model)])
    print(clf.steps)
    # model = model_class(**model_params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = f1_score(preds, y_test, average='macro')
    # numpy.savez_compressed('test_and_pred_{0}.npz'.format(name), test=X_test, predictions=preds)
    show_report(y_test, preds)

    return score


