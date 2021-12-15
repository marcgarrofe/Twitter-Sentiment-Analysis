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
                # vocabulary[word] = np.array([0, 0], dtype=float) # Numpy és menys eficient
                vocabulary[word] = [0, 0]
                vocabulary[word][row_class] = 1
            else:
                vocabulary[word][row_class] += 1

    return vocabulary


def laplace_smoothing(vocabulary: dict, zero_prob: int, one_prob: int):
    """
    Calculates the probabilities of each word applying Laplace Smoothing
    :param vocabulary: Dictionary to be modified containing the counter classification for each word
    :param zero_prob: Total probability of 0
    :param one_prob: Total probability of 1
    :return: Dictionary with the probabilities of each word
    """
    laplace_l = 1
    laplace_r = 2  # Al ser binary = 2

    for word in vocabulary:
        vocabulary[word][0] = (vocabulary[word][0] + laplace_l) / (zero_prob + laplace_l * laplace_r)
        vocabulary[word][1] = (vocabulary[word][1] + laplace_l) / (one_prob + laplace_l * laplace_r)

    return vocabulary


def learn_naive_bayes_text(x_train: np.ndarray, y_train: np.ndarray):
    """
    Generates the probabilities dictionary of the model
    :param x_train: Input training data
    :param y_train: Output training data
    :return: Final dictionary of the model
    """
    vocabulary = count_words(x_train, y_train)
    train_size = x_train.size
    return laplace_smoothing(vocabulary, train_size/2, train_size/2)


def classify_naive_bayes_text(vocabulary, x_test: np.ndarray):
    """
    Generates a classification for the input data
    :param vocabulary: Input probabilities dictionary of the model
    :param x_test: Input data to be classified
    :return: List with the classification. Example : list([0, 0, 1, 0, 0])
    """
    classfication = list()

    for row in x_test:
        zero_prob = 0.0
        one_prob = 0.0

        for word in row.split():
            # Si la paraula esta al dictionary llavors es pot calcular la seva probabilitat
            if word in vocabulary.keys():
                zero_prob += np.log(vocabulary[word][0])
                one_prob += np.log(vocabulary[word][1])

        if zero_prob >= one_prob:
            classfication.append(0)
        else:
            classfication.append(1)

    return np.asarray(classfication, dtype=int)
