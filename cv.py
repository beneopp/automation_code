import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import copy


def make_random_chunks(l, n):
    np.random.shuffle(l)

    grouped_l = []
    quotient = len(l) // n
    remainder = len(l) % n
    i = 0
    group = 1

    while i < len(l):
        if group <= remainder:
            grouped_l.append(l[i:i+quotient+1])
            i += quotient+1
        else:
            grouped_l.append(l[i:i+quotient])
            i += quotient
        group += 1

    return grouped_l


def generate_model(train_set, cv=False):
    X = train_set[:, 0:2]
    y = train_set[:, 2]

    if cv and len(X) > 10:
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X, y)
    return clf


def get_prediction(clf, test_set):
    X = test_set[:, 0:2]
    model_pred = clf.predict(X)
    return model_pred


def get_accuracy(test_set, train_set):
    clf = generate_model(train_set, cv=True)
    model_pred = get_prediction(clf, test_set)
    y = test_set[:, 2]
    agreements = np.sum(model_pred == y)
    return agreements


def get_cv_result(train_data, n):
    chunks = make_random_chunks(train_data, n)
    results = np.zeros(n)

    for i in range(n):
        test_set = chunks[i]

        train_set = copy.deepcopy(chunks)
        del train_set[i]
        train_set = np.concatenate(train_set)

        result = get_accuracy(test_set, train_set)
        results[i] = result

    output = sum(results) / train_data.shape[0]

    return output








