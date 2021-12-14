import numpy as np
from src.algorithms.naive_bayes_algorithms import learn_naive_bayes_text, classify_naive_bayes_text
import time
import sklearn


class NB:
    """
    def __init__(self):
        self.vocabulary = dict()

    def get_params(self, deep=False):
        return {
            'vocabulary': self.vocabulary,
        }
    """

    def get_params(self, deep=False):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Trains the Naïve Bayes fucntion
        :param x: Input training data
        :param y: Output training data
        """
        start = time.time()
        self.vocabulary = learn_naive_bayes_text(x, y)
        end = time.time()
        print("::-> fit() Time = ", end - start)

    def predict(self, x: np.ndarray):
        """
        Given an input data, predicts the classfication
        :param x: Input testing data
        :return: list() of the classfication
        """
        start = time.time()
        classfication = classify_naive_bayes_text(self.vocabulary, x)
        end = time.time()
        print("::-> fit() Time = ", end - start)
        return classfication

    def score(self, x, y_true):
        """
        Given a input data and
        :param x:
        :param y_true:
        :return:
        """
        good_classification = 0
        y_pred = self.predict(x)
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                good_classification += 1
        return good_classification / y_true.size


