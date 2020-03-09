import math
import numpy as np


class Criterion():

    """
    This class compute the metric value used to decide to split a node or not.
    """

    def __init__(self,criterion):

        if criterion == "gini":
            self.node_impurity = self.node_gini

        if criterion == "entropy":
            self.node_impurity = self.node_entropy


    def node_entropy(self, sample):
        """
        Returns entropy of a divided group of data
        Data may have multiple classes
        """
        entropy = 0
        n_samples = len(sample)
        classes = set(sample)
        for c in classes:   # for each class, get impurity
            count_class = sum(sample==c)

            if count_class>0:
                prob = count_class/n_samples
                entropy = -(prob)*math.log(prob, 2)
                entropy += prob*entropy

        return entropy


    def node_gini(self, sample):
        """
        Returns gini of a divided group of data
        Data may have multiple classes
        """
        mad = np.abs(np.subtract.outer(sample, sample)).mean()
        gini = mad/np.mean(sample)/2

        return gini


class Splitter():

    def __init__(self, criterion):

        self.criterion = criterion

    def find_best_split(self, X, y):

        index = None
        min_impurity = 1
        threshold = None

        for col_index, col_values in enumerate(X.T):
            impurity, cutoff = self.find_best_value(col_values, y)

            if impurity <= min_impurity:
                min_impurity = impurity
                index = col_index
                threshold = cutoff

        return index, threshold, min_impurity


    def find_best_value(self, col_values, y):

        min_impurity = 10

        for value in set(col_values):
            y_predict = col_values < value
            impurity = self.criterion.node_impurity(y[~y_predict])
            impurity += self.criterion.node_impurity(y[y_predict])

            if impurity <= min_impurity:
                min_impurity = impurity
                threshold = value

        return impurity, threshold


class Tree():

    def predict(self, X):

        results = np.zeros(X.shape[0])
        for row_index, row_value in enumerate(X):
            results[row_index] = self.apply(row_value)

        return results


    def apply(self, row, current_node):

        current_node = self.trees

        while current_node.get('cutoff'):
            if row[current_node['index_col']] < current_node['cutoff']:
                current_node = current_node['left']
            else:
                current_node = current_node['right']
        else:
            return current_node.get('val')


class Builder():

    def __init__(self,splitter, max_depth, min_samples_leaf):

        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build(self, tree, X, y):

        if all(val == y[0] for val in y):
            return {'val':y[0]}

        tree.node = self._add_split_node(X,y)


    def _add_split_node(self, X, y, node={}, depth=0):

        if node is None or len(y)==0 or depth >= self.max_depth:
            return None


        node = {'val': np.round(np.mean(y))}

        if len(y)>=self.min_samples_leaf:
            feature, threshold, impurity = self.splitter.find_best_split(X, y)

            y_left = y[X[:, feature] < threshold]
            y_right = y[X[:,feature] >= threshold]

            X_left = X[X[:,feature] < threshold]
            X_right = X[X[:,feature] >= threshold]

            node['feature'] = feature,
            node['threshold'] = threshold
            node['left']=self._add_split_node(X_left, y_left, {}, depth+1),
            node['right']=self._add_split_node(X_right, y_right, {}, depth+1)


        self.trees = node

        return node


class DecisionTreeClassifier():

    """A decision tree classifier.

    This object is expected to work as the the DecisionTreeClassifier from
    Scikit-Learn.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
    splitter : "best"
    max_depth : int, default=5
    min_samples_leaf: int default=5
    """

    def __init__(self, max_depth=5, min_samples_leaf=5, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.splitter = "best"


    def load_trained_model(self, node):
        self.tree_ = node
        return 1


    def fit(self, X, y):

        self.tree_ = Tree()
        self.criterion = Criterion(self.criterion)
        self.splitter = Splitter(self.criterion)
        self.builder = Builder(self.splitter,
                               self.max_depth,
                               self.min_samples_leaf)

        self.builder.build(self.tree_,X,y)
        return 1

    def predict(self, X):
        self.tree_.predict(X)


if __name__=="__main__":

    from sklearn.datasets import load_iris
    from pprint import pprint

    iris = load_iris()

    X = iris.data
    y = iris.target

    classifier = DecisionTreeClassifier(max_depth=7)
    m = classifier.fit(X, y)

    pprint(classifier.tree_.node)
