
import random
import unittest

import numpy as np

from skratch.datasets import load_iris
from skratch.datasets import load_wine
from skratch.datasets import load_breast_cancer

from skratch.garden import RandomGardenClassifier


class ParameterssTest(unittest.TestCase):

    def test_max_depth(self):

        """
        Test that the model fits and predicts with different parameter values
        for max_depth=1,2,3,5,10,20.
        """

        for i in [1,2,3,5,10,20]:
            X = load_iris().data
            y = load_iris().target
            clf = RandomGardenClassifier(max_depth=i).fit(X, y)


    def test_min_samples(self):

        """
        Test that the model fits and predicts with different parameter values
        for min_samples_leaf=1,2,3,5,10.
        """

        for i in [1,2,3,5,10]:
            X = load_iris().data
            y = load_iris().target
            clf = RandomGardenClassifier(min_samples_leaf=i).fit(X, y)


    def test_min_samples(self):

        """
        Test that the model fits and predicts with different parameter values
        for min_samples_leaf=1,2,3,5,10.
        """

        for i in [3,5,10,13]:
            X = load_wine().data
            y = load_wine().target
            clf = RandomGardenClassifier(max_features=i).fit(X, y)


class TransformationTest(unittest.TestCase):

    def test_random(self):

        """
        Test that the model does not learn when the target are randomized.
        """

        random.seed(28)
        np.random.seed(28)

        X = np.random.rand(500,5)
        y = np.random.randint(2, size=(500))

        X_train, X_test = X[0:400,:], X[400:,:]
        y_train, y_test = y[0:400], y[400:]

        clf = RandomGardenClassifier().fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = np.sum(y_pred==y_test)/len(y_test)

        assert acc > 0.4 and acc <0.6



class ScoreTest(unittest.TestCase):

    def test_iris_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_iris().data
        y = load_iris().target

        clf = RandomGardenClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9


    def test_wine_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_wine().data
        y = load_wine().target

        clf = RandomGardenClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9


    def test_breast_cancer_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_breast_cancer().data
        y = load_breast_cancer().target

        clf = RandomGardenClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9
