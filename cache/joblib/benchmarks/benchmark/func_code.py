# first line: 33
@cache
def benchmark(model_class, model_params=None, iters=1):
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

    X, y, vocab, label_encoder = prepare_data()
    class_count = len(label_encoder.classes_)
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab
    model_params['class_count'] = class_count

    scores = []
    times = []
    for i in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model = model_class(**model_params)
        start = time()
        preds = model.fit(X_train, y_train).predict(X_test)
        end = time()
        scores.append(accuracy_score(preds, y_test))
        times.append(end - start)
    return scores, times
