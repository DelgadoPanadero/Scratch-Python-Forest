import numpy as np

class RandomGardenClassifier():
    """
    Base class for forest of trees-based classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 max_samples=None):

        self.base_estimator_ base_estimator
        self.n_estimators_ = n_estimators
        self.estimator_params=estimator_params
        self.bootstrap = bootstrap
        self.obb_score = obb_score
        self.max_samples = max_samples


    def _make_estimator(self):

        """
        Make and configure a copy of the `base_estimator_` attribute.
        """

        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        return estimator


    def _set_oob_score(self, X, y):

        """
        Compute out-of-bag score.
        """

        n_classes_ = len(set(y))
        n_samples = y.shape[0]

        oob_score = 0.0
        oob_decision_function = []
        predictions = np.zeros((n_samples, n_classes_))

        n_samples_bootstrap = self.max_samples if self.max_samples else n_samples

        for estimator in self.estimators_:
            random = np.random.RandomState()
            sample_indices = random.randint(0, n_samples, n_samples_bootstrap)
            #unsampled_mask = 0==np.bincount(sample_indices, minlength=n_samples)
            #unsampled_indices = np.arange(n_samples)[unsampled_mask]
            p_estimator = estimator.predict_proba(X[sampled_indices, :])

            predictions[unsampled_indices, :] += p_estimator

        decision = predictions/predictions.sum(axis=1)[:, np.newaxis]
        self.oob_decision_function_=decision
        self.oob_score += np.mean(y == np.argmax(predictions, axis=1)) #, axis=0)


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
            prediction = estimator.predict_proba(X)
            all_proba += prediction

        all_proba /= len(self.estimators_)

        return all_proba


    def apply(self, X):

        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, dense matrix} of shape (n_samples, n_features).

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """

        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer="threads"))(
            delayed(tree.apply)(X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T


    def fit(self, X, y, sample_weight=None):
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
        n_samples_bootstrap = self.max_samples if self.max_samples else n_samples


        self.estimators_ = [self._make_estimator() for i in range(n_estimators)]

        random = np.random.RandomState()
        for estimator in self.estimators_: #TODO class balanced sample
            sample_indices = random.randint(0, n_samples, n_samples_bootstrap)
            sample_counts = np.bincount(sample_indices, minlength=n_samples)
            estimator.fit(X[sample_counts,:], y[sample_counts])

        self._set_oob_score(X, y) if self.oob_score else None

        return self
