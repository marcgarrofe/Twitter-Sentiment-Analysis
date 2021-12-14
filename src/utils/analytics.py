import pandas as pd
import numpy as np


def count_class_type(dataset: pd.DataFrame, column_label: str):
    """
    Counts de class type of the dataset
    :param dataset: Input data
    :param column_label: Name of the column to be analyzed
    """
    assert dataset.size > 0, 'count_class_type() : dataset empty'
    assert column_label in dataset.columns, 'count_class_type() : column_label not in dataset'

    zero_count = 0
    one_count = 0

    for value in dataset[column_label]:
        if value == 0:
            zero_count += 1
        elif value == 1:
            one_count += 1

    print('Class Counter :')
    print('  0 appeared ', zero_count, ' times')
    print('  1 appeared ', one_count, ' times\n')


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray):
    """
    Given the actual classification (or true classification) and the predicted by the algorithm,
    returns the confusion matrix
    :param actual: True classification
    :param predicted: Classification of the implemented algorithm. Example [[27, 2],[5, 22]]
    :return:
    """
    assert actual.size == predicted.size, 'confusion_matrix() : actual and predicted not the same size'
    assert actual.size > 0, 'confusion_matrix() : actual empty'

    matrix = np.array([[0, 0], [0, 0]])

    for i in range(actual.size):
        if actual[i] == predicted[i]:
            if actual[i] == 1:
                matrix[0][0] += 1
            else:
                matrix[1][1] += 1
        else:
            if actual[i] == 1:
                matrix[1][0] += 1
            else:
                matrix[0][1] += 1

    return matrix


def accuracy(conf_matrix: np.ndarray):
    """
    Accuracy = ( TP + TN ) / ( TP + FP + FN + TN )
    :param conf_matrix: Input confusion matrix
    :return: accuracy calculated from the confusion matrix
    """
    return (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1]
                                                      + conf_matrix[1][0] + conf_matrix[1][1])


def precision(conf_matrix: np.ndarray):
    """
    Precision = TP / ( TP + FP )
    :param conf_matrix: Input confusion matrix
    :return: precision calculated from the confusion matrix
    """
    return conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])


def recall(conf_matrix: np.ndarray):
    """
    Recall = TP / ( TP + FN )
    :param conf_matrix: Input confusion matrix
    :return: recall calculated from the confusion matrix
    """
    return conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
