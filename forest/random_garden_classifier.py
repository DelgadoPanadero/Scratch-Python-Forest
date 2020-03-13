import copy
import numpy as np

from tree import DecisionBonsaiClassifier


class RandomGardenClassifier():

    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    base_estimator_ = DecisionBonsaiClassifier()

    def __init__(self,
                 n_estimators=100,
                 estimator_params=dict(),
                 max_samples=None,
                 max_features=None):

        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators_ = n_estimators
        self.estimator_params = estimator_params

        self.estimator_params["max_features"] = max_features


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
        n_samples_bootstrap = n_samples
        self.n_features_= X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        if self.max_features is None:
            self.max_features = X.shape[1]

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


if __name__=="__main__":


    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix
    from pprint import pprint

    iris = load_iris()
    X = iris.data
    y = iris.target

    estimator_params = {
        "max_features": X.shape[1]-2,
        "max_samples" : X.shape[0]/0.6
    }

    classifier = RandomGardenClassifier(n_estimators=100,
                                        estimator_params=dict())

    m = classifier.fit(X, y)
    print(m)

    print("\n\nCONFUSION MATRIX\n")
    prediction = classifier.predict(iris.data)
    print(confusion_matrix(y,prediction))
