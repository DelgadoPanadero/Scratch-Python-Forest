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

from tree import DecisionBonsaiClassifier


class MultinomialDevianceLoss():

    """
    Multinomial deviance loss function for multi-class classification.
    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.

    Parameters
    ----------
    K : int The number of regression trees to be induced.
    """

    def __init__(self, n_classes):
        self.K = n_classes


    def negative_gradient(self, y, pred, k=0):

        """
        Compute negative gradient for the ``k``-th class.

        Parameters
        ----------
        y : np.ndarray, shape=(n,) The target labels.
        y_pred : np.ndarray, shape=(n,): The predictions.
        """

        logsumexp = np.log(np.sum(np.exp(pred)))

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
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k])

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))


    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):

        """
        Make a single Newton-Raphson step. Our node estimate is given by:

            sum(y - prob) / sum(prob * (1 - prob))

        we take advantage that: y - prob = residual
        """

        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)

        y = y.take(terminal_region, axis=0)

        numerator = residual.sum()
        numerator *= (self.K - 1) / self.K

        denominator = np.sum((y - residual) * (1.0 - y + residual))

        if denominator == 0.0:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator


class BoostingGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    base_estimator_ = DecisionBonsaiClassifier()

    def __init__(self,
                 n_estimators=100,
                 estimator_params=dict(),
                 max_samples=None):

        self.max_samples = max_samples
        self.n_estimators_ = n_estimators
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

        pass

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

        pass

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

        random = np.random.RandomState()

        # Init prior probabilities for the first iteration
        #TODO

        # Init splitter and criterion
        #TODO

        # Perform boosting iterations
        for i in range(begin_at_stage, self.n_estimators):

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_mask)

            # no need to fancy index w/ no subsampling
            self.train_score_[i] = loss_(y, y_pred)

        return self


    def _fit_stage(self, i, X, y, y_pred, sample_mask):

        """
        Fit another stage of ``n_classes_`` trees to the boosting model.
        """

        for k in range(self.loss.K):

            y = np.array(y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, y_pred, k=k)

            # induce regression tree on residuals
            tree = DecisionBonsaiClassifier(
                      max_depth=self.max_depth,
                      min_samples_leaf=self.min_samples_leaf,
                      max_features=self.max_features)

            tree.fit(X, residual, sample_weight=sample_weight)

            # update tree leaves
            loss.update_terminal_regions(tree.tree_, X, y, residual, y_pred,
                                         sample_mask, self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return y_pred
