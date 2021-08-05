
import random
import unittest

import numpy as np

from skratch.datasets import load_iris
from skratch.datasets import load_wine
from skratch.datasets import load_breast_cancer

from skratch.bonsai import DecisionBonsaiClassifier


class ParameterssTest(unittest.TestCase):

    def test_max_depth(self):

        """
        Test that the model fits and predicts with different parameter values
        for max_depth=1,2,3,5,10,20.
        """

        for i in [1,2,3,5,10,20]:
            X = load_iris().data
            y = load_iris().target
            clf = DecisionBonsaiClassifier(max_depth=i).fit(X, y)


    def test_min_samples(self):

        """
        Test that the model fits and predicts with different parameter values
        for min_samples_leaf=1,2,3,5,10.
        """

        for i in [1,2,3,5,10]:
            X = load_iris().data
            y = load_iris().target
            clf = DecisionBonsaiClassifier(min_samples_leaf=i).fit(X, y)


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

        clf = DecisionBonsaiClassifier().fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = np.sum(y_pred==y_test)/len(y_test)

        assert acc > 0.4 and acc <0.6


    def test_permutation(self):

        """
        Test that de model does not change under feature permutation.
        """

        X = load_iris().data
        y = load_iris().target

        clf1 = DecisionBonsaiClassifier().fit(X, y)
        y_pred1 = clf1.predict(X)

        X = X.T
        np.random.shuffle(X)
        X = X.T

        clf2 = DecisionBonsaiClassifier().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==y_pred2).all()


    def test_inversion(self):

        """
        Test that the model does not change the sign of the features is
        inverted.
        """

        X = load_iris().data
        y = load_iris().target
        clf1 = DecisionBonsaiClassifier().fit(X, y)
        y_pred1 = clf1.predict(X)

        y = -y

        clf2 = DecisionBonsaiClassifier().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==-y_pred2).all()


    def test_dilatation(self):

        """
        Test that the model does not change under feature dilatation.
        """

        X = load_iris().data
        y = load_iris().target

        clf1 = DecisionBonsaiClassifier().fit(X, y)
        y_pred1 = clf1.predict(X)

        X = X * np.random.randint(1,10,size=X.shape[1])

        clf2 = DecisionBonsaiClassifier().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==y_pred2).all()


class ScoreTest(unittest.TestCase):

    def test_iris_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_iris().data
        y = load_iris().target

        clf = DecisionBonsaiClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9


    def test_wine_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_wine().data
        y = load_wine().target

        clf = DecisionBonsaiClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9


    def test_breast_cancer_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_breast_cancer().data
        y = load_breast_cancer().target

        clf = DecisionBonsaiClassifier().fit(X, y)
        y_pred = clf.predict(X)

        assert np.sum(y_pred==y)/len(y) > 0.9
