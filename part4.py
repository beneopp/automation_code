import numpy as np
from sklearn.linear_model import LogisticRegression
import cv
import random
import matplotlib.pyplot as plt


def calculate_accuracy(pred, actual):
    y = actual[:, 2]
    accuracy = np.sum(pred == y) / len(y)
    return accuracy


def initialize_arrays(unobserved_array):
    """
    :param unobserved_array: array of input data
    :return: training_data: data used for training
    :return: unobserved_array: data used for testing
    """
    
    initial_obs_ind = calculate_initial_obs_ind(unobserved_array)
    training_data = unobserved_array[initial_obs_ind, :]
    unobserved_array = np.delete(unobserved_array, initial_obs_ind, axis=0)
    return training_data, unobserved_array


def update_arrays(unobserved_array, training_data, ind):
    """
    :param unobserved_array:
    :param training_data:
    :param ind: index of data to incorporate to data
    """

    new_obs = unobserved_array[ind, :]
    new_obs = new_obs.reshape((1,-1))
    training_data = np.append(training_data, new_obs, axis=0)
    unobserved_array = np.delete(unobserved_array, ind, axis=0)
    return training_data, unobserved_array


def calculate_initial_obs_ind(unobserved_array):
    """
    :param unobserved_array: data used for training
    :return: initial_obs_ind: index of 5 samples to use for training data
    """

    initial_obs_ind = random.sample(range(len(unobserved_array)), 5)

    y = unobserved_array[initial_obs_ind, 2]
    while np.all(y == 0) or np.all(y == 1) or np.all(y == 0):
        initial_obs_ind = random.sample(range(len(unobserved_array)), 5)
        y = unobserved_array[initial_obs_ind, 2]

    return initial_obs_ind


def generate_model(train_set):
    X = train_set[:, 0:2]
    y = train_set[:, 2]

    model = LogisticRegression(random_state=0, penalty="none").fit(X, y)
    model.fit(X, y)

    return model


def get_test_accuracy(training_data, unobserved_array):
    model = generate_model(training_data)
    pred = cv.get_prediction(model, unobserved_array)
    accuracy = calculate_accuracy(pred, unobserved_array)
    return accuracy, model


def get_most_uncertain(model, unobserved_array):
    X = unobserved_array[:, 0:2]
    probs = model.predict_proba(X)
    probs = np.max(probs, axis=1)
    most_uncertain_point_ind = np.argmin(probs)
    return most_uncertain_point_ind


def get_most_dense(unobserved_array):

    num_of_samples = len(unobserved_array)
    X = unobserved_array[:, 0:2]

    a1 = np.reshape(X, (num_of_samples, 1, 2))
    a1 = np.repeat(a1, num_of_samples, axis=1)

    a2 = np.reshape(X, (1, num_of_samples, 2))
    a2 = np.repeat(a2, num_of_samples, axis=0)

    dists = np.linalg.norm(a1 - a2, axis=2)

    avg = np.sum(dists, axis=0) / num_of_samples

    most_dense = np.argmin(avg)

    return most_dense


def single_run(unobserved_array,next_sample_method):
    """
    :param unobserved_array: array of input data
    :return: rounds: array containing accuracy information
    """

    training_data, unobserved_array = initialize_arrays(unobserved_array)
    accuracy, model = get_test_accuracy(training_data, unobserved_array)
    rounds = [accuracy]

    while unobserved_array.shape[0] >= 50:

        if next_sample_method == "uncertainty":
            next_obs_index = get_most_uncertain(model, unobserved_array)
        elif next_sample_method == "density":
            next_obs_index = get_most_dense(unobserved_array)

        training_data, unobserved_array = update_arrays(unobserved_array, training_data, next_obs_index)
        accuracy, model = get_test_accuracy(training_data, unobserved_array)
        rounds.append(accuracy)

    return rounds, training_data, unobserved_array


def plot_results(train_means, train_sds, dataset_name, next_sample_method):
    plt.clf()

    # plot train data mean
    plt.plot(range(1, len(train_means)+1), train_means, color="red")
    # plot train data standard deviation
    plt.plot(range(1, len(train_means)+1), train_means + train_sds, "--", color="red")
    plt.plot(range(1, len(train_means)+1), train_means - train_sds, "--", color="red")

    if next_sample_method == "uncertainty":
        title = "Test Accuraccy with Uncertainty Method for Dataset " + str(dataset_name)
        f = "output/test_results_uncertainty_dataset" + str(dataset_name) + ".png"
    else:
        title = "Test Accuraccy with Density Method for Dataset " + str(dataset_name)
        f = "output/test_results_density_dataset" + str(dataset_name) + ".png"

    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy")

    plt.title(title)
    plt.savefig(f)


def multiple_runs(seeds, next_sample_method, dataset_name):

    start_num = 5
    end_num = 50
    num_of_rounds = end_num - start_num + 2
    rounds = np.zeros((10, num_of_rounds))

    for i, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)

        # get results for each round
        rounds[i], training_data, unobserved_array = single_run(random_dataset, next_sample_method)

        if i == 0:
            plot_sample_data(training_data, unobserved_array, next_sample_method, dataset_name, point_type=True)
            if next_sample_method == "uncertainty":
                # arbitrary choice so it wasn't called twice
                plot_sample_data(training_data, unobserved_array, next_sample_method, dataset_name, point_type=False)

    train_means = np.mean(rounds, axis=0)
    train_sds = np.std(rounds, axis=0)

    plot_results(train_means, train_sds, dataset_name, next_sample_method)


def plot_point_type(array, fmt, data_type):
    zero_index = array[:, 2] == 0
    zero_points = array[zero_index]
    one_points = array[np.logical_not(zero_index)]
    label0 = data_type + " from class 0"
    plt.scatter(zero_points[:, 0], zero_points[:, 1], color="blue", marker=fmt, label=label0)
    label1 = data_type + " from class 1"
    plt.scatter(one_points[:, 0], one_points[:, 1], color="red", marker=fmt, label=label1)


def normal_plot(array):
    zero_index = array[:, 2] == 0
    zero_points = array[zero_index]
    one_points = array[np.logical_not(zero_index)]
    plt.scatter(zero_points[:, 0], zero_points[:, 1], color="blue", label="class 0")
    plt.scatter(one_points[:, 0], one_points[:, 1], color="red", label="class 1")


def plot_sample_data(training_data, unobserved_array, next_sample_method, dataset_name, point_type=False):
    plt.clf()

    if point_type:
        plot_point_type(training_data, "x", "labeled")
        plot_point_type(unobserved_array, "o", "unlabeled")
        title = "Dataset " + str(dataset_name) + " Using " + next_sample_method.capitalize() + " Method"
        f = "output/sample_plot_for_dataset_" + str(dataset_name) + "_" + next_sample_method + ".png"

    else:
        array = np.append(training_data, unobserved_array, axis=0)
        normal_plot(array)
        title = "Dataset " + str(dataset_name)
        f = "output/sample_plot_for_dataset_" + str(dataset_name) + ".png"

    plt.title(title)
    plt.legend()

    plt.savefig(f)


random.seed(180)
np.random.seed(180)

# create dataset 1 where uncertainty is best
random_dataset = np.zeros((100, 3))

random_dataset[:, 0:2] = np.random.uniform(low=1, high=2, size=(100, 2))
random_dataset[49, 0] = np.random.uniform(low=-0.5, high=0, size=1)
random_dataset[49, 1] = np.random.uniform(low=2, high=2.5, size=1)

random_dataset[50:, 0] = np.random.uniform(low=-2, high=-1, size=50)
random_dataset[50:, 2] = 1

# run dataset 1
seeds = [random.randint(0, 500) for i in range(10)]

print("uncertainty for dataset 1")
multiple_runs(seeds, "uncertainty", 1)

print("density for dataset 1")
multiple_runs(seeds, "density", 1)

# run dataset 2
random_dataset = np.zeros((100, 3))

random_dataset[0:40, 0] = np.random.uniform(low=2, high=6, size=40)
random_dataset[0:40, 1] = np.random.uniform(low=0, high=3.5, size=40)

random_dataset[40:50, 0] = np.random.uniform(low=8.5, high=9.5, size=10)
random_dataset[40:50, 1] = np.random.uniform(low=1, high=2, size=10)

random_dataset[50:, 2] = 1

random_dataset[50:98, 0] = np.random.uniform(low=1, high=4, size=48)
random_dataset[50:98, 1] = np.random.uniform(low=2, high=5, size=48)

random_dataset[98:, 0] = np.random.uniform(low=6, high=9, size=2)
random_dataset[98:, 1] = np.random.uniform(low=8, high=10, size=2)

print("uncertainty for dataset 2")
multiple_runs(seeds, "uncertainty", 2)

print("density for dataset 2")
multiple_runs(seeds, "density", 2)



