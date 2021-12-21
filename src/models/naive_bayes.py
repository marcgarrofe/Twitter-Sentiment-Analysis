import numpy as np
from src.algorithms.naive_bayes_algorithms import learn_naive_bayes_text, classify_naive_bayes_text, count_class
import time


class NB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocabulary = dict()
        self.total_zero_prob = 0.0
        self.total_one_prob = 0.0
        self.minimum_appearances = -1

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
        Trains the Naive Bayes function
        :param x: Input training data
        :param y: Output training data
        :param verbose: If higher than 0, prints more info
        """
        start = time.time()
        self.total_zero_prob, self.total_one_prob = count_class(y)
        self.total_zero_prob = self.total_zero_prob / y.size
        self.total_one_prob = self.total_one_prob / y.size
        self.vocabulary = learn_naive_bayes_text(x, y, self.alpha, minimum_appearances=self.minimum_appearances)
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
        classification = classify_naive_bayes_text(self.vocabulary, x, self.total_zero_prob, self.total_one_prob)
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
