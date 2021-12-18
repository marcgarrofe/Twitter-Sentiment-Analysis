import numpy as np


def count_words(x_train: np.ndarray, y_train: np.ndarray):
    """
    Given the training input and output values, generate the dict() for further probabilities
    :param x_train: Input training data
    :param y_train: Output training data
    :return: Initialized dictionary
    """
    vocabulary = dict()
    for index, row in enumerate(x_train):
        for word in row.split():
            row_class = y_train[index]
            if word not in vocabulary.keys():
                vocabulary[word] = [0, 0]  # vocabulary[word] = np.array([0, 0], dtype=float) Numpy és menys eficient
                vocabulary[word][row_class] = 1
            else:
                vocabulary[word][row_class] += 1

    return vocabulary


def laplace_smoothing(vocabulary: dict, zero_prob: int, one_prob: int, alpha=1, minimum_appearances=-1):
    """
    Calculates the probabilities of each word applying Laplace Smoothing
    :param vocabulary: Dictionary to be modified containing the counter classification for each word
    :param zero_prob: Total probability of 0
    :param one_prob: Total probability of 1
    :param alpha: Additive (Laplace) smoothing parameter (0 for no smoothing)
    :param minimum_appearances: Minimum times a words needs to appear to be fitted in the dictionary.
        If -1, all words are fitted
    :return: Dictionary with the probabilities of each word
    """
    laplace_l = alpha
    laplace_r = 2  # Al ser binary = 2

    for word in vocabulary:
        if minimum_appearances == -1 or vocabulary[word][0] + vocabulary[word][1] >= minimum_appearances:
            vocabulary[word][0] = (vocabulary[word][0] + laplace_l) / (zero_prob + laplace_l * laplace_r)
            vocabulary[word][1] = (vocabulary[word][1] + laplace_l) / (one_prob + laplace_l * laplace_r)
        else:
            del vocabulary[word]

    return vocabulary


def learn_naive_bayes_text(x_train: np.ndarray, y_train: np.ndarray, alpha=1, minimum_appearances=-1):
    """
    Generates the probabilities dictionary of the model
    :param x_train: Input training data
    :param y_train: Output training data
    :param alpha: Additive (Laplace) smoothing parameter (0 for no smoothing)
    :param minimum_appearances: Minimum times a words needs to appear to be fitted in the dictionary.
        If -1, all words are fitted.
    :return: Final dictionary of the model
    """
    vocabulary = count_words(x_train, y_train)
    zero_counter, one_counter = count_class(y_train)
    return laplace_smoothing(vocabulary, zero_counter, one_counter, alpha, minimum_appearances)


def classify_naive_bayes_text(vocabulary, x_test: np.ndarray, total_zero_prob: float, total_one_prob: float):
    """
    Generates a classification for the input data
    :param vocabulary: Input probabilities dictionary of the model
    :param x_test: Input data to be classified
    :param total_zero_prob: Total probability of zero class
    :param total_one_prob: Total probability of one class
    :return: List with the classification. Example : list([0, 0, 1, 0, 0])
    """
    classification = list()

    for row in x_test:
        zero_prob = 0.0
        one_prob = 0.0

        for word in row.split():
            # Si la paraula esta al dictionary llavors es pot calcular la seva probabilitat
            if word in vocabulary.keys():
                zero_prob += np.log(vocabulary[word][0])
                one_prob += np.log(vocabulary[word][1])

        if zero_prob * total_zero_prob > one_prob * total_one_prob:
            classification.append(0)
        else:
            classification.append(1)

    return np.asarray(classification, dtype=int)


def count_class(y: np.ndarray):
    """
    Calculates the number of times a class has appeared
    :param y: Array of the dataset output class
    :return: Number of times that 0 and 1 values appear. Example : 19, 8
    """
    zero_counter = 0
    one_counter = 0
    for classification in y:
        if classification == 0:
            zero_counter += 1
        else:
            one_counter += 1
    return zero_counter, one_counter
