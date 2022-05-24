import numpy as np
import random
import matplotlib.pyplot as plt
import cv
import uncertainty


random.seed(36)

f = "../../data/classification.csv"
classification_array = np.loadtxt(f, skiprows=1, delimiter=",")


class Round:
    def __init__(self, training_data, unobserved_array):

        self.train_accuracy = cv.get_cv_result(training_data, 5)
        self.test_accuracy = get_test_accuracy(training_data, unobserved_array)


class Results:
    def __init__(self, train_data, test_data):
        self.train_means, self.train_sd = calculate_stats(train_data)
        self.test_means, self.test_sd = calculate_stats(test_data)


def get_test_accuracy(training_data, unobserved_array):
    model = cv.generate_model(training_data)
    pred = cv.get_prediction(model, unobserved_array)
    accuracy = calculate_accuracy(pred, unobserved_array)
    return accuracy


def calculate_accuracy(pred, actual):
    y = actual[:, 2]
    accuracy = np.sum(pred == y) / len(y)
    return accuracy


def calculate_initial_obs_ind(unobserved_array):
    """
    :param unobserved_array: data used for training
    :return: initial_obs_ind: index of 5 samples to use for training data
    """
    initial_obs_ind = random.sample(range(len(unobserved_array)), 5)

    y = unobserved_array[initial_obs_ind, 2]
    while np.all(y == 0) or np.all(y == 1):
        initial_obs_ind = random.sample(range(len(unobserved_array)), 5)
        y = unobserved_array[initial_obs_ind, 2]

    return initial_obs_ind


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


def single_run(unobserved_array, mellow=False):
    """
    :param unobserved_array: array of input data
    :param mellow: whether or not a mellow method is used
    :return: rounds: array containing accuracy information
    """

    training_data, unobserved_array = initialize_arrays(unobserved_array)
    rounds = [Round(training_data, unobserved_array)]

    while unobserved_array.shape[0] >= 50:
        next_obs_index = uncertainty.get_most_uncertain(training_data, unobserved_array, mellow)
        training_data, unobserved_array = update_arrays(unobserved_array, training_data, next_obs_index)
        rounds.append(Round(training_data, unobserved_array))

    return rounds


def calculate_stats(array):
    """
    :param array: array of
    :return:
    """
    average = np.mean(array, axis=0)
    sd = np.std(array, axis=0)
    return average, sd


def plot_results(results, mellow=False):
    plt.clf()

    # plot train data mean
    plt.plot(range(0, len(results.train_means)), results.train_means, color="red")
    # plot train data standard deviation
    plt.plot(range(0, len(results.train_sd)), results.train_means + results.train_sd, "--", color="red")
    plt.plot(range(0, len(results.train_sd)), results.train_means - results.train_sd, "--", color="red")

    if mellow:
        plt.title("Train Accuracy of Mellow Method")
    else:
        plt.title("Train Accuracy")

    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.1])

    if mellow:
        plt.savefig("output/train_results_mellow.png")
    else:
        plt.savefig("output/train_results.png")

    plt.clf()
    # plot test data mean
    plt.plot(range(1, len(results.test_means)+1), results.test_means, color="blue")
    # plot test data standard deviation
    plt.plot(range(1, len(results.test_sd)+1), results.test_means + results.test_sd, "--", color="blue")
    plt.plot(range(1, len(results.test_sd)+1), results.test_means - results.test_sd, "--", color="blue")

    if mellow:
        plt.title("Test Accuracy of Mellow Method")
    else:
        plt.title("Test Accuracy")

    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.1])

    if mellow:
        plt.savefig("output/test_results_mellow.png")
    else:
        plt.savefig("output/test_results.png")


# initialize arrays
start_num = 5
end_num = 50
num_of_rounds = end_num - start_num + 2

train_data = np.zeros((10, num_of_rounds))
test_data = np.zeros((10, num_of_rounds))

seeds = [random.randint(0, 500) for i in range(10)]
for i, seed in enumerate(seeds):
    random.seed(seed)
    np.random.seed(seed)

    # get results for each round
    rounds = single_run(classification_array)

    train_data[i] += [r.train_accuracy for r in rounds]
    test_data[i] += [r.test_accuracy for r in rounds]

results = Results(train_data, test_data)
plot_results(results)

# mellow method

train_data = np.zeros((10, num_of_rounds))
test_data = np.zeros((10, num_of_rounds))

for i, seed in enumerate(seeds):
    random.seed(seed)
    np.random.seed(seed)

    # get results for each round
    rounds = single_run(classification_array, mellow=True)

    train_data[i] += [r.train_accuracy for r in rounds]
    test_data[i] += [r.test_accuracy for r in rounds]

results = Results(train_data, test_data)

plot_results(results, mellow=True)
