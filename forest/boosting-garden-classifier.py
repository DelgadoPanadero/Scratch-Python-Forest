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

from tree import DecisionBonsaiRegressor


class MultinomialDevianceLoss():

    """
    Multinomial deviance loss function for multi-class classification.
    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.

    Parameters
    ----------
    K : int The number of regression trees to be induced.
    """

    def __init__(self, n_classes=2):
        self.K = n_classes


    def negative_gradient(self, y, pred, k=0):

        """
        Compute negative gradient for the ``k``-th class.

        Parameters
        ----------
        y : np.ndarray, shape=(n,) The target labels.
        y_pred : np.ndarray, shape=(n,): The predictions.
        """
        pred_T = np.rollaxis(pred, 1)
        vmax = pred_T.max(axis=0)
        logsumexp = np.log(np.sum(np.exp(pred_T-vmax), axis=0)) + vmax

        return y - np.nan_to_num(np.exp(pred[:,k]-logsumexp))


    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learning_rate=1.0, k=0):

        """
        Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree The tree object.
        X : np.ndarray, shape=(n, m) The data array.
        y : np.ndarray, shape=(n,) The target labels.
        residual : np.ndarray, shape=(n,) The residuals (negative gradient).
        y_pred : np.ndarray, shape=(n,) The predictions.
        """

        # compute leaf for each sample in ``X``.
        terminal_regions = [tree.apply(row) for row in X[~sample_mask,:]]

        # Get the unique regions from all terminal regions
        unique_region_pointers = set([id(d) for d in terminal_regions])

        # update each leaf (= perform line search)
        for leaf in unique_region_pointers:
            self._update_terminal_region(tree, terminal_regions, leaf,
                                                    X[~sample_mask  ],
                                                    y[~sample_mask  ],
                                             residual[~sample_mask  ],
                                               y_pred[~sample_mask,k])

        # update predictions (both in-bag and out-of-bag)
        terminal_values=[tree.apply(row)["value"] for row in X[~sample_mask,:]]
        y_pred[:, k] += learning_rate * np.array(terminal_values)

        return y_pred


    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):

        """
        Make a single Newton-Raphson step. Our node estimate is given by:

            sum(y - prob) / sum(prob * (1 - prob))

        we take advantage that: y - prob = residual
        """

        # Pointer filter mask
        region_mask = [id(x) == leaf for x in terminal_regions]

        # Filter terminal region data
        y = y[region_mask]
        residual = residual[region_mask]

        # Compute new value
        numerator = residual.sum()*(self.K-1)/self.K
        denominator = np.sum((y-residual)*(1.0-y+residual))
        new_value = numerator/denominator if denominator!=0.0 else 0.0

        # Substitute the value
        tree.apply(X[region_mask,:][0,:])["value"] = new_value


class BoostingGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    loss_ = MultinomialDevianceLoss()
    base_estimator_ = DecisionBonsaiRegressor

    def __init__(self,
                 n_estimators=100,
                 estimator_params=dict(),
                 max_samples=None,
                 max_depth=5,
                 min_samples_leaf=1,
                 learning_rate=1.0,
                 subsample=1):

        self.subsample = subsample
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.n_estimators_ = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.estimator_params = estimator_params


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

        score = self.decision_function(X)

        score_T = np.rollaxis(score, 1)
        vmax = score_T.max(axis=0)

        proba = np.ones((score.shape[0], self.loss_.K), dtype=np.float64)
        logsumexp = np.log(np.sum(np.exp(score_T-vmax), axis=0)) + vmax

        proba = np.nan_to_num(np.exp(score - logsumexp[:, np.newaxis]))

        return proba


    def decision_function(self, X):

        """
        Perform the decision

        Parameters
        ----------
        X : {array-like, dense matrix} of shape (n_samples, n_features)
        """

        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.loss_.K))

        for i in range(self.n_estimators_):
            for k in range(self.loss_.K):
                if self.estimators_[i, k]:

                    pred = self.estimators_[i,k].predict(X)
                    out[:, k] += self.learning_rate*pred

        return out


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
        self.classes_ = np.unique(y)
        self.loss_.K = self.classes_.shape[0]
        mask_size = int(self.subsample*n_samples)

        # Init splitter and criterion
        self.estimators_ = np.empty((self.n_estimators_, self.loss_.K),
                                    dtype=type(self.base_estimator_))

        # Init prior probabilities for the first iteration
        y_pred = np.empty((n_samples, self.loss_.K), dtype=np.float64)
        y_pred[:] = np.bincount(y) / float(y.shape[0])

        # Perform boosting iterations
        for i in range(self.n_estimators_):

            # Data sample
            sample_mask = np.random.choice(n_samples, mask_size, replace=False)

            # predict for the actual state
            y_pred = self.decision_function(X) if i>0 else y_pred

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_mask)

            # no need to fancy index w/ no subsampling
            #self.train_score_[i] = self.loss_(y, y_pred)

        return self


    def _fit_stage(self, i, X, y, y_pred, sample_mask):

        """
        Fit another stage of ``n_classes_`` trees to the boosting model.
        """

        original_y = y
        for k in range(self.loss_.K):

            # Binary tarjet
            y = np.array(original_y == k, dtype=np.int32)

            # Compute residuals
            residual = self.loss_.negative_gradient(y, y_pred, k=k)

            # Induce regression tree on residuals
            bonsai = DecisionBonsaiRegressor(
                                      max_depth = self.max_depth,
                                      min_samples_leaf = self.min_samples_leaf
                                      )

            # Train the tree
            bonsai.fit(X[~sample_mask,:], residual[~sample_mask])

            # Update tree leaves
            y_pred = self.loss_.update_terminal_regions(bonsai.bonsai_, X, y,
                                                        residual, y_pred,
                                                        sample_mask,
                                                        self.learning_rate, k)
            # Add tree to ensemble
            self.estimators_[i, k] = bonsai

        return y_pred


if __name__=="__main__":

    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix
    from pprint import pprint

    iris = load_iris()
    X = iris.data
    y = iris.target

    classifier = BoostingGardenClassifier(n_estimators=100)
    classifier = classifier.fit(X, y)

    print("\n\nCONFUSION MATRIX\n")
    prediction = classifier.predict(iris.data)
    print(confusion_matrix(y,prediction))
