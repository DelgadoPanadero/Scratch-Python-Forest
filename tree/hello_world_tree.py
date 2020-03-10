import numpy as np


class Criterion():

    """
    This class compute the metric value used to decide to split a node or not.

    Parameters
    ----------
    criterion: str, {"gini", "entropy"}.
    """

    def __init__(self,criterion):

        self.criterion = criterion


    def node_impurity(self, sample):

        """
        Returns entropy or gini coefficient from a given sample.

        Parameters
        ----------
        sample: list, data sample of values.
        """

        if self.criterion == "gini":
            return self.node_gini(sample)

        if self.criterion == "entropy":
            return self.node_entropy(sample)


    def node_entropy(self, sample):

        """
        Returns entropy of a divided group of data (may have multiple classes).

        Parameters
        ----------
        sample: list, data sample of values.
        """

        entropy = 0.0
        n_classes = set(sample)
        weighted_n_node_samples = len(sample)

        for c in n_classes:
            count_k = sum(sample==c)
            if 0<count_k:
                count_k /= weighted_n_node_samples
                entropy -= count_k*np.log2(count_k)

        return entropy


    
    def node_gini(self, sample):

        """
        Returns gini of a divided group of data (may have multiple classes).

        Parameters
        ----------
        sample: list, data sample of values.
        """
        
        sq_count = 0
        n_classes = set(sample)
        weighted_n_node_samples = len(sample)

        for c in range(n_classes):
            count_k = sum(sample==c)
            sq_count += count_k**2

        gini = 1.0-sq_count/weighted_n_node_samples**2

        return gini




class Splitter():

    """
    Class compute the splitting feature and threshold given a node sample.
    It computes the "best split" algorithm.

    Parameters
    ---------
    criterion: Criterion.
    """

    def __init__(self, criterion):

        self.criterion = criterion


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
        min_impurity = 1
        threshold = None

        for feature_index, feature_values in enumerate(X.T):
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
        y : list, array-like (n_samples,). The target values as integers
        """

        min_impurity = 10
        for value in set(feature_values):
            y_predict = feature_values < value
            impurity = self.criterion.node_impurity(y[~y_predict])
            impurity += self.criterion.node_impurity(y[y_predict])

            if impurity <= min_impurity:
                min_impurity = impurity
                threshold = value

        return impurity, threshold


class Tree():

    """
    Base tree. Given a built tree model, this class walk through it to perform
    predictions.
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
            results[row_index] = self.apply(row_value)

        return results


    def apply(self, row, current_node):

        """
        Given a dataset row, it crawls through the tree model to return
        its appropriate leaf value.

        Parameters
        ----------
        X : dense matrix, The training input samples.
        """

        current_node = self.trees

        while current_node.get('cutoff'):
            if row[current_node['index_col']] < current_node['cutoff']:
                current_node = current_node['left']
            else:
                current_node = current_node['right']
        else:
            return current_node.get('val')


class Builder():

    """
    Factory of trees. It performs the "depth-fashion" algorithm.
    """

    def __init__(self,splitter, max_depth, min_samples_leaf):

        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build(self, tree, X, y):

        """
        Given an new of Tree, it builds the tree graph

        Parameters
        ----------
        tree: Tree.
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers
        """

        if all(val == y[0] for val in y):
            return {'val':y[0]}

        tree.node = self._add_split_node(X,y)


    def _add_split_node(self, X, y, node={}, depth=0):

        """
        Given a data set, it calls the splitter recursively to get all
        the nodes
        tree: Tree.
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers.
        node: dict, node values.
        depth: current depth of the node.
        """

        if node is None or len(y)==0 or depth >= self.max_depth:
            return None

        node = {'val': np.round(np.mean(y))}

        if len(y)>=self.min_samples_leaf:
            feature, threshold, impurity = self.splitter.find_best_split(X, y)

            y_left = y[X[:, feature] < threshold]
            y_right = y[X[:,feature] >= threshold]

            X_left = X[X[:,feature] < threshold]
            X_right = X[X[:,feature] >= threshold]

            node['feature'] = feature
            node['threshold'] = threshold
            node['left']=self._add_split_node(X_left, y_left, {}, depth+1)
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
    splitter : "depth-first"
    max_depth : int, default=5
    min_samples_leaf: int default=5
    """

    def __init__(self, max_depth=5, min_samples_leaf=5, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.splitter = "depth-first"


    def load_trained_model(self, node):

        """
        Load a previous tree graph

        Parameters
        ----------
        node: dict, tree graph with json format.
        """

        self.tree_ = node


    def fit(self, X, y):

        """
        Train the model.

        Parameters
        ----------
        tree: Tree.
        X : dense matrix, The training input samples.
        y : list, array-like (n_samples,). The target values as integers
        """

        self.tree_ = Tree()
        self.criterion = Criterion(self.criterion)
        self.splitter = Splitter(self.criterion)
        self.builder = Builder(self.splitter,
                               self.max_depth,
                               self.min_samples_leaf)

        self.builder.build(self.tree_,X,y)


    def predict(self, X):

        """
        Perform prediction from a given dataset.

        Parameters
        ----------
        tree: Tree.
        X : dense matrix, The training input samples.
        """

        self.tree_.predict(X)


if __name__=="__main__":

    from sklearn.datasets import load_iris
    from pprint import pprint

    iris = load_iris()

    X = iris.data
    y = iris.target
    print(type(X),type(y))
    classifier = DecisionTreeClassifier(max_depth=7)
    m = classifier.fit(X, y)

    pprint(classifier.tree_.node)
