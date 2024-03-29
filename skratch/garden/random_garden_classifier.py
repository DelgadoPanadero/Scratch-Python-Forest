#!/usr/bin/python3

#  Copyright (c) 2020 Angel Delgado Panadero

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in 
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

import copy
import numpy as np

from ..bonsai import DecisionBonsaiClassifier


class RandomGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    base_estimator_ = DecisionBonsaiClassifier

    def __init__(self,
                 max_depth=5,
                 max_samples=None,
                 max_features=None,
                 n_estimators=10,
                 min_samples_leaf=1):

        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators_ = n_estimators
        self.min_samples_leaf = min_samples_leaf


    def _make_estimator(self):

        """
        Make and configure a copy of the `base_estimator_` attribute.
        """

        estimator = copy.deepcopy(self.base_estimator_)

        return estimator(max_depth = self.max_depth,
                         max_features = self.max_features,
                         min_samples_leaf = self.min_samples_leaf)


    def predict(self, X):

        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, dense matrix} of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,) The predicted classes.
        """

        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


    def predict_proba(self, X):

        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, dense matrix} of shape (n_samples, n_features)

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes) The class probabilities
            of the input samples.
        """

        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        for estimator in self.estimators_:
            prediction = estimator.predict(X)
            prediction_one_hot = np.eye(self.n_classes_)[prediction]
            all_proba += prediction_one_hot

        all_proba /= len(self.estimators_)

        return all_proba


    def fit(self, X, y):

        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, dense matrix} of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) The target values

        Returns
        -------
        self : object
        """

        n_samples = X.shape[0]
        n_samples_bootstrap = n_samples
        self.n_features_= X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        if self.max_samples:
            n_samples_bootstrap=self.max_samples

        self.estimators_ = []
        for i in range(self.n_estimators_):
            self.estimators_.append(self._make_estimator())

        random = np.random.RandomState()
        for estimator in self.estimators_: #TODO class balanced sample
            sample_indices = random.randint(0, n_samples, n_samples_bootstrap)
            estimator.fit(X[sample_indices,:], y[sample_indices])

        return self
