
import random
import unittest

import numpy as np

from skratch.datasets import load_boston
from skratch.datasets import load_diabetes

from skratch.bonsai import DecisionBonsaiRegressor


def r2_score(y_true, y_pred):

    """
    Very similar to the Sklearn implementation but without weights and single
    output.
    """

    numerator = ((y_true-y_pred)**2).sum(axis=0,dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)
                    )** 2).sum(axis=0,dtype=np.float64)
    score = 1 - (numerator/ denominator)

    return score


class ParameterssTest(unittest.TestCase):

    def test_max_depth(self):

        """
        Test that the model fits and predicts with different parameter values
        for max_depth=1,2,3,5,10,20.
        """

        for i in [1,2,3,5,10,20]:
            X = load_boston().data
            y = load_boston().target
            DecisionBonsaiRegressor(max_depth=i).fit(X, y)


    def test_min_samples(self):

        """
        Test that the model fits and predicts with different parameter values
        for min_samples_leaf=1,2,3,5,10.
        """

        for i in [1,2,3,5,10]:
            X = load_boston().data
            y = load_boston().target
            DecisionBonsaiRegressor(min_samples_leaf=i).fit(X, y)


class TransformationTest(unittest.TestCase):

    def test_random(self):

        """
        Test that the model does not learn when the target are randomized.
        """

        random.seed(28)
        np.random.seed(28)

        X = np.random.rand(500,5)
        y = np.random.rand(500)

        X_train, X_test = X[0:400,:], X[400:,:]
        y_train, y_test = y[0:400], y[400:]

        clf = DecisionBonsaiRegressor().fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        assert r2_score(y_test,y_pred)<0.3


    def test_permutation(self):

        """
        Test that de model does not change under feature permutation.
        """

        X = load_boston().data
        y = load_boston().target

        clf1 = DecisionBonsaiRegressor().fit(X, y)
        y_pred1 = clf1.predict(X)

        X = X.T
        np.random.shuffle(X)
        X = X.T

        clf2 = DecisionBonsaiRegressor().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==y_pred2).all()


    def test_inversion(self):

        """
        Test that the model does not change the sign of the features is
        inverted.
        """

        X = load_boston().data
        y = load_boston().target
        clf1 = DecisionBonsaiRegressor().fit(X, y)
        y_pred1 = clf1.predict(X)

        y = -y

        clf2 = DecisionBonsaiRegressor().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==-y_pred2).all()


    def test_dilatation(self):

        """
        Test that the model does not change under feature dilatation.
        """

        X = load_boston().data
        y = load_boston().target

        clf1 = DecisionBonsaiRegressor().fit(X, y)
        y_pred1 = clf1.predict(X)

        X = X * np.random.randint(1,10,size=X.shape[1])

        clf2 = DecisionBonsaiRegressor().fit(X, y)
        y_pred2 = clf2.predict(X)

        assert (y_pred1==y_pred2).all()


class ScoreTest(unittest.TestCase):

    def test_boston_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_boston().data
        y = load_boston().target
        clf = DecisionBonsaiRegressor().fit(X, y)
        y_pred = clf.predict(X)

        assert r2_score(y, y_pred)>0.8


    def test_diabetes_fitting(self):

        """
        Test if the model is tested with the training data, the adjustment is
        perfect.
        """

        X = load_diabetes().data
        y = load_diabetes().target
        clf = DecisionBonsaiRegressor(max_depth=10).fit(X, y)
        y_pred = clf.predict(X)

        assert r2_score(y, y_pred)>0.8
