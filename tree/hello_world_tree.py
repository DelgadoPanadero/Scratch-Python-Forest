import math
import numpy as np



def entropy_function(count_class, n_samples):
    """
    The math formula
    """
    return -(count_class/n_samples)*math.log(count_class/n_samples, 2)
#    entropy=-(count_class/n_samples)*math.log(count_class/n_samples, 2)
#    entropy-=(1-count_class/n_samples)*math.log(1-count_class/n_samples, 2)
#    return entropy


# get the entropy of one big circle showing above
def node_impurity(sample):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    entropy = 0
    n_samples = len(sample)
    classes = set(sample)

    for c in classes:   # for each class, get entropy
        count_class = sum(sample==c)

        if count_class>0:
            weight = count_class/n_samples
            entropy += weight*entropy_function(count_class, n_samples)

    return entropy, n_samples


# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/False, and y_true can be multi class
    """

    n_samples = len(y_real)
    s_true, n_true = node_impurity(y_real[y_predict]) # left hand side entropy
    s_false, n_false = node_impurity(y_real[~y_predict]) # right hand side entropy
    s = n_true/n_samples * s_true + n_false/n_samples * s_false

    return s


class DecisionTreeClassifier():

    """A decision tree classifier.

    This object is expected to work as the the DecisionTreeClassifier from
    Scikit-Learn.

    https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
    splitter : {"best", "random"}, default="best"
    max_depth : int, default=None
    min_samples_split : int or float, default=2
    min_samples_leaf : int or float, default=1
    min_weight_fraction_leaf : float, default=0.0
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
    random_state : int, RandomState instance, default=None
    max_leaf_nodes : int, default=None
    min_impurity_decrease : float, default=0.0
    min_impurity_split : float, default=0
    class_weight : dict, list of dict or "balanced", default=None
    presort : deprecated, default='deprecated'
    ccp_alpha : non-negative float, default=0.0

    """

    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth

    def fit(self, X, y, par_node={}, depth=0):

        if par_node is None or len(y) == 0 or depth >= self.max_depth:
            return None

#        if all(val == y[0] for val in y):
#            return {'val':y[0]}

        col_index, cutoff, entropy = self.find_best_split(X, y)

        y_left = y[X[:, col_index] < cutoff]
        y_right = y[X[:, col_index] >= cutoff]

        par_node = {'col': col_index, #X.feature_names[col_index],
                    'col_index':col_index,
                    'cutoff':cutoff,
                    'val': np.round(np.mean(y))}

        par_node['left'] = self.fit(X[X[:, col_index] < cutoff], y_left, {}, depth+1)
        par_node['right'] = self.fit(X[X[:, col_index] >= cutoff], y_right, {}, depth+1)

        self.depth += 1
        self.trees = par_node

        return par_node

    def load_trained_model(self, par_node):
        self.trees = par_node
        return 1

    def find_best_split(self, X, y):

        index = None
        min_entropy = 1
        cutoff = None

        for col_index, col_values in enumerate(X.T):
            entropy, cur_cutoff = self.find_best_value(col_values, y)

#            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
#                return index, cur_cutoff, entropy

            if entropy <= min_entropy:
                min_entropy = entropy
                index = col_index
                cutoff = cur_cutoff

        return index, cutoff, min_entropy

    def find_best_value(self, col_values, y):

        min_entropy = 10

        for value in set(col_values):
            y_predict = col_values < value

            entropy = get_entropy(y_predict, y)

            if entropy <= min_entropy:
                min_entropy = entropy
                cutoff = value

        return entropy, cutoff


    def predict(self, X):

        """
        """

        results = np.zeros(len(X)) #np.array([0]*len(X))

        for row_index, row_value in enumerate(X):
            results[row_index] = self._get_prediction(row_value)

        return results


    def _get_prediction(self, row):

        """
        """

        current_node = self.trees

        while current_node.get('cutoff'):
            if row[current_node['index_col']] < current_node['cutoff']:
                current_node = current_node['left']
            else:
                current_node = current_node['right']
        else:
            return cur_layer.get('val')


if __name__=="__main__":

    from sklearn.datasets import load_iris
    from pprint import pprint

    iris = load_iris()

    X = iris.data
    y = iris.target

    classifier = DecisionTreeClassifier(max_depth=7)
    m = classifier.fit(X, y)

    pprint(m)
