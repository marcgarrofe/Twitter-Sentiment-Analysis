import numpy as np
from src.algorithms.naive_bayes_algorithms import learn_naive_bayes_text, classify_naive_bayes_text
import time


class NB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_params(self, deep=False):
        return {
            'alpha': self.alpha
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray, verbose=0):
        """
        Trains the Naïve Bayes function
        :param x: Input training data
        :param y: Output training data
        :param verbose: If higher than 0, prints more info
        """
        start = time.time()
        self.vocabulary = learn_naive_bayes_text(x, y)
        end = time.time()
        if verbose == 1:
            print("::-> fit() Time = ", end - start)

    def predict(self, x: np.ndarray, verbose=0):
        """
        Given an input data, predicts the classification
        :param x: Input testing data
        :return: list() of the classification
        :param verbose: If higher than 0, prints more info
        """
        start = time.time()
        classification = classify_naive_bayes_text(self.vocabulary, x)
        end = time.time()
        if verbose == 1:
            print("::-> predict() Time = ", end - start)
        return classification

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
