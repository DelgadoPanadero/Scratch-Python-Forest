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
        terminal_regions = [tree.apply(row) for row in X[~sample_mask,:]]

        # update each leaf (= perform line search)
        for leaf in set(terminal_regions):
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

        #TODO sub "==" for "is"
        region_mask = [x == leaf for x in terminal_regions]

        # Filter terminal region data
        y = y[region_mask]
        residual = residual[region_mask]

        # Compute new value
        numerator = residual.sum()*(self.K-1)/self.K
        denominator = np.sum((y-residual)*(1.0-y+residual))

        # Substitute the value
        tree.apply(X[region_mask,:][0,:])["value"] = numerator/denominator



class BoostingGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    loss_ = MultinomialDevianceLoss()
    base_estimator_ = DecisionBonsaiClassifier()

    def __init__(self,
                 n_estimators=100,
                 estimator_params=dict(),
                 max_samples=None,
                 learning_rate=1.0,
                 subsample=1):


        self.subsample = subsample
        self.max_samples = max_samples
        self.n_estimators_ = n_estimators
        self.learning_rate = learning_rate
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

        proba = np.ones((score.shape[0], self.n_classes_), dtype=np.float64)

        score = self.decision_function(X)

        logsumexp = np.log(np.sum(np.exp(score)))

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
                for j in range(n_samples):

                    node = self.estimators_[i, k].bonsai_

                    while node.get('left_child') and node.get('right_child'):

                        if X[j, node.get('feature')] <= node.get('threshold'):
                            node = node['left_child' ]
                        else:
                            node = node['right_child']

                    out[j, k] += self.learning_rate * node.get('value',0.0)

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
        mask_size = int(self.subsample*n_samples)

        # Data sample
        random = np.random.RandomState()
        sample_mask = np.random.choice(n_samples, mask_size, replace=False)

        # Init prior probabilities for the first iteration
        self.loss_.K = self.classes_.shape[0]

        # Init splitter and criterion
        #TODO

        # Perform boosting iterations
        for i in range(self.n_estimators_):

            # predict for the actual state
            y_pred = self.decision_function(X)

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_mask)

            # no need to fancy index w/ no subsampling
            self.train_score_[i] = self.loss_(y, y_pred)

        return self


    def _fit_stage(self, i, X, y, y_pred, sample_mask):

        """
        Fit another stage of ``n_classes_`` trees to the boosting model.
        """

        for k in range(self.loss_.K):

            y = np.array(y == k, dtype=np.float64)

            # Compute residuals
            residual = self.loss_.negative_gradient(y, y_pred, k=k)

            # Residuals to binary
            residual = np.int(1/(1 + np.exp(-residual))>0.5)

            # induce regression tree on residuals
            bonsai = DecisionBonsaiClassifier(
                      max_depth=self.max_depth,
                      min_samples_leaf=self.min_samples_leaf)

            bonsai.fit(X, residual, sample_weight=sample_weight)

            # update tree leaves
            y_pred = self.loss.update_terminal_regions(bonsai.bonsai_, X, y,
                                                       residual, y_pred,
                                                       sample_mask,
                                                       self.learning_rate, k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return y_pred
