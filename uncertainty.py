import numpy as np
from scipy.spatial import KDTree
import random


def get_weights(class_assignments, indices, c):

    c_ind = np.arange(len(class_assignments))[class_assignments == c+1]
    indicator = np.isin(indices, c_ind)

    proportions = np.sum(indicator, axis=1) / indicator.shape[1]

    return proportions


def get_neighbors(observed_data, unobserved_data):

    observed_data = observed_data[:, 0:2]
    unobserved_data = unobserved_data[:, 0:2]

    tree = KDTree(observed_data)
    _, indices = tree.query(unobserved_data, k=5)

    return indices


def get_proportions(observed_data, unobserved_data):
    indices = get_neighbors(observed_data, unobserved_data)

    proportions = np.zeros((unobserved_data.shape[0], 2))
    for c in range(2):
        proportions[:, c] = get_weights(observed_data[:, 2], indices, c)

    return proportions


def get_most_uncertain(observed_data, unobserved_data, mellow):

    proportions = get_proportions(observed_data, unobserved_data)
    diffs = np.abs(proportions[:, 0] - proportions[:, 1])

    if mellow:
        index = diffs != 1

        if not np.any(index):
            index = np.repeat(True, len(diffs))

    else:
        min_diff = np.min(diffs)
        index = diffs == min_diff

    index = np.arange(len(diffs))[index]
    most_uncertain_point_ind = random.choice(index)

    return most_uncertain_point_ind

