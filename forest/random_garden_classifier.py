import copy
import numpy as np

from tree import DecisionBonsaiClassifier


class RandomGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 n_estimators=100,
                 estimator_params=dict(),
                 bootstrap=False,
                 max_samples=None):

        self.base_estimator_ = DecisionBonsaiClassifier()
        self.n_estimators_ = n_estimators
        self.estimator_params=estimator_params
        self.bootstrap = bootstrap
        self.max_samples = max_samples


    def _make_estimator(self):

        """
        Make and configure a copy of the `base_estimator_` attribute.
        """

        estimator = copy.deepcopy(self.base_estimator_)
        [setattr(estimator, p, v) for p, v in self.estimator_params.items()]

        return estimator


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
        self.n_features_= X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        n_samples_bootstrap = self.max_samples if self.max_samples else n_samples

        self.estimators_ = [self._make_estimator() for i in range(self.n_estimators_)]

        random = np.random.RandomState()
        for estimator in self.estimators_: #TODO class balanced sample
            sample_indices = random.randint(0, n_samples, n_samples_bootstrap)
            estimator.fit(X[sample_indices,:], y[sample_indices])

        return self
