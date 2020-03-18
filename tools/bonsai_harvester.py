import copy
import numpy as np


class BonsaiHarvester():

    """
    This class convert a Bonsai json model into a list of numpy array boxes.
    Each of this boxes is a numpy array with shape (n_features,2) and it stores
    the feature space dimension of each leaf node.

    Parameters
    ----------
    X : {array-like, dense matrix} of shape (n_samples, n_features). Tran data.
    """

    def __init__(self, X):

        self.leaf_boxes = []
        self.bonsai_box = X


    @property
    def bonsai_box(self):

        """
        Numpy array with the dimensaions of the training data domain in the
        feature space.
        """

        return self._bonsai_box


    @bonsai_box.setter
    def bonsai_box(self, X):

        """
        Given the data, the bonsai_box property is set automatically in the
        __init__().

        Parameters
        ----------
        X : {array-like, dense matrix} shape (n_samples, n_features). Tran data.
        """

        bonsai_box = np.zeros((X.shape[1],2), dtype=float)
        bonsai_box[:,0] = np.amin(X, axis=0)
        bonsai_box[:,1] = np.amax(X, axis=0)

        self._bonsai_box = bonsai_box


    def harvest_bonsai(self, bonsai_graph):

        """
        Given the Bonsai json graph, compute the leafs space domain.

        Parameters
        ----------
        node: Bonsai graph or Bonsai graph node.
        """

        self._divide_box_node(bonsai_graph, self.bonsai_box)


    def harvest_garden(self, garden_classifier):

        """
        Given the Bonsai Garden classifier. Get the leafs of all the bonsais.

        Parameters
        ----------
        node: Bonsai graph or Bonsai graph node.
        """

        for estimator in garden_classifier.estimators_:
            bonsai = estimator.bonsai_.graph
            harvester.harvest_bonsai(bonsai)


    def _divide_box_node(self, node, parent_box):

        """
        Recursive method to parse the Bonsai graph until the leafs and obtain
        its feature space domain.

        Parameters
        ----------
        node: Bonsai graph or Bonsai graph node.
        parent_box: {array-like, dense matrix} shape (n_features,2). Domain
            of the parent node.
        """

        if not node.get('threshold'):
            self.leaf_boxes.append({"box":parent_box, "value":node["value"]})

        else:
            left_node = node.get( 'left_node', {})
            right_node =node.get('right_node', {})

            left_box = copy.deepcopy(parent_box)
            right_box= copy.deepcopy(parent_box)

            left_box[node["feature"],1] = node['threshold']
            right_box[node["feature"],0] = node['threshold']

            self._divide_box_node(left_node, left_box )
            self._divide_box_node(right_node,right_box)


if __name__=="__main__":

    from pprint import pprint
    from sklearn.datasets import load_iris
    from tree import DecisionBonsaiClassifier
    from forest import RandomGardenClassifier

    iris = load_iris()
    X = iris.data
    y = iris.target

    classifier = DecisionBonsaiClassifier(max_depth=7)
    m = classifier.fit(X, y)

    bonsai = classifier.bonsai_.graph
    print("\n\nBONSAI GRAPH\n")
    pprint(bonsai)

    harvester = BonsaiHarvester(X)
    print("\n\nBONSAI BOX\n")
    print(harvester.bonsai_box)

    harvester.harvest_bonsai(bonsai)
    print("\n\HAVERST TREE\n")
    pprint(harvester.leaf_boxes)

    classifier = RandomGardenClassifier(n_estimators=10)
    m = classifier.fit(X, y)

    harvester.harvest_garden(classifier)
    print("\n\HAVERST GARDEN\n")
    pprint(harvester.leaf_boxes)
