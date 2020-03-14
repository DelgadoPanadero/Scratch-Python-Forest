import copy
import numpy as np


class BonsaiSquareBaler():

    """
    This class convert a Bonsai json model into a list of numpy array boxes.
    Each of this boxes is a numpy array with shape (n_features,2) and it stores
    the feature space dimension of each leaf node.
    """

    def __init__(self, X):
        self.boxes = []
        self.global_box = X


    @property
    def global_box(self):
        return self._global_box


    @global_box.setter
    def global_box(self, X):

        global_box = np.zeros((X.shape[1],2), dtype=float)
        global_box[:,0] = np.amin(X, axis=0)
        global_box[:,1] = np.amax(X, axis=0)

        self._global_box = global_box


    def bale_bonsai(self, bonsai_graph):

        self._divide_box_node(bonsai_graph, self.global_box)


    def _divide_box_node(self, node, parent_box):

        if not node.get('threshold'):
            self.boxes.append(parent_box)

        else:
            left_node = node.get(' left_node', {})
            right_node =node.get('right_node', {})

            left_box = copy.deepcopy(parent_box)
            right_box= copy.deepcopy(parent_box)

            left_box[node["feature"],1] = node['threshold']
            right_box[node["feature"],0] = node['threshold']

            self._divide_box_node(left_node, left_box )
            self._divide_box_node(right_node,right_box)
