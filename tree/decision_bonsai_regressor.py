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

import numpy as np


class Criterion():

    """
    This class compute the metric value used to decide to split a node or not.

    """

    def __init__(self,criterion):

        self.criterion = "mse"


    def node_impurity(self, sample):

        """
        Returns entropy or gini coefficient from a given sample.

        Parameters
        ----------
        sample: list, data sample of values.
        """

        if self.criterion == "mse":
            return self.node_mse(sample)


    def node_mse(self, sample):

        """
        Returns entropy of a divided group of data (may have multiple classes).

        Parameters
        ----------
        sample: list, data sample of values.
        """

        return np.sum(np.square(np.array(sample)-np.mean(sample)))


class Splitter():

    """
    Class compute the splitting feature and threshold given a node sample.
    It computes the "best split" algorithm.

    Parameters
    ---------
    criterion: Criterion.
    max_features: int. Maximum number of features to look for the split.
    min_samples_leaf: int default=5. Minimum number of register in a leaf node
    """

    def __init__(self, criterion, max_features, min_samples_leaf):

        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf


    def feature_sampling(self, X):

        """
        It performs the sample feature selection.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        """

        n_features = X.shape[1]
        self.sample_features = [i for i in range(n_features)]

        if self.max_features is None:
            self.max_features = n_features

        if self.max_features<n_features:
            random = np.random.RandomState()
            self.sample_features = random.choice(n_features,
                                                 self.max_features,
                                                 replace=False)


    def find_best_split(self, X, y):

        """
        Compute the feature and value which performs the best split given
        a dataset.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers.
        """

        index = None
        threshold = None
        min_impurity = np.inf

        for feature_index, feature_values in enumerate(X.T):
            if feature_index in self.sample_features:
                impurity, cutoff = self.find_best_value(feature_values, y)

                if impurity <= min_impurity:
                    min_impurity = impurity
                    index = feature_index
                    threshold = cutoff

        return index, threshold, min_impurity


    def find_best_value(self, feature_values, y):

        """
        Given a feature column, which minimize the criterion of the target y.

        Parameters
        ----------
        feature_values : list, array-like of shape (n_samples,). Column feature
        y : list, array-like (n_samples,). The target values as integers.
        """

        threshold = np.inf
        min_impurity = np.inf

        for val in np.unique(feature_values):
            y_predict = feature_values < val
            impurity  = self.criterion.node_impurity(y[~y_predict])
            impurity += self.criterion.node_impurity(y[ y_predict])

            if impurity <= min_impurity:
                min_impurity = impurity
                threshold = val

        return min_impurity, threshold


class Bonsai():

    """
    Base Tree though to work similarly as a lite version of the sklearn Tree
    class (that is why it is called bonsai). Given a built model, this class
    walk through it to perform predictions.
    """


    def predict(self, X):

        """
        Given a dataset, return an array with the class predicted for each data
        row.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        """

        results = np.zeros(X.shape[0])
        for row_index, row_value in enumerate(X):
            leaf_node = self.apply(row_value)
            results[row_index] = leaf_node.get('value')

        return results


    def apply(self, row):

        """
        Given a dataset row, it crawls through the bonsai model to return
        its appropriate leaf value.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        """

        current_node=self.graph
        while current_node.get('threshold'):
            if row[current_node['feature']] < current_node['threshold']:
                current_node = current_node['left_node']
            else:
                current_node = current_node['right_node']
        else:
            return current_node


class Builder():

    """
    Factory of bonsais. It performs the "depth-fashion" algorithm.
    """

    def __init__(self,splitter, max_depth, min_samples_leaf):

        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build(self, bonsai, X, y):

        """
        Given an new of Bonsai instance, it builds the bonsai graph

        Parameters
        ----------
        bonsai: Bonsai.
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers
        """

        bonsai.graph = self._add_split_node(X,y)


    def _add_split_node(self, X, y, depth=0):

        """
        Given a data set, it calls the splitter recursively to get all
        the nodes.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers.
        depth: int, default=0. Current depth of the node.
        """

        self.splitter.feature_sampling(X)

        node = {'value': np.mean(y)}

        if len(y)<=self.min_samples_leaf
            return node

        if depth >= self.max_depth:
            return node

        if len(np.unique(y))==1:
            return node

        feature, threshold, impurity = self.splitter.find_best_split(X, y)

        y_left = y[X[:,feature] < threshold]
        y_right =y[X[:,feature] >=threshold]

        if (len(y_left) >= self.min_samples_leaf and
            len(y_right)>= self.min_samples_leaf ):

            X_left  = X[X[:,feature] < threshold]
            X_right = X[X[:,feature] >=threshold]

            node[ 'feature'  ] = feature
            node['threshold' ] = threshold
            node['left_node' ] = self._add_split_node(X_left, y_left, depth+1)
            node['right_node'] = self._add_split_node(X_right,y_right,depth+1)

        return node


class DecisionBonsaiRegressor():

    """
    A decision tree regressor. This object is expected to work as a lite
    version of the DecisionTreeRegressor from Scikit-Learn (That is why
    it is called Bonsai!).

    Parameters
    ----------
    criterion : {"mse"}
    splitter : "depth-first"
    max_depth : int, default=5. Maximum tree depth
    min_samples_leaf: int default=5. Minimum number of register in a leaf node
    """

    def __init__(self,
                 criterion="mse",
                 max_depth=7,
                 min_samples_leaf=5,
                 max_features=None):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitter = "depth-first"
        self.max_features = max_features


    def load_trained_model(self, graph):

        """
        Load a previous bonsai graph

        Parameters
        ----------
        graph: dict, bonsai graph with json format.
        """

        self.bonsai_.graph = graph


    def fit(self, X, y):

        """
        Train the model.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers
        """

        self.bonsai_ = Bonsai()
        self.criterion = Criterion(self.criterion)
        self.splitter = Splitter(self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf)
        self.builder = Builder(self.splitter,
                               self.max_depth,
                               self.min_samples_leaf)

        self.builder.build(self.bonsai_,X,y)

        return self


    def predict(self, X):

        """
        Perform prediction from a given dataset.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        """

        return self.bonsai_.predict(X)


if __name__=="__main__":

    from sklearn.metrics import r2_score
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    reg = DecisionBonsaiRegressor(max_depth=15)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print(r2_score(y_test, y_pred))
