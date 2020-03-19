import numpy as np


class MultidimensionalIOU():

    def _area(self, box):

        """
        Compute the area of a box.

        Parameters
        ----------
        """

        differences = box[:,1]-box[:,0]

        return np.prod(differences)


    def _intersection(self, box_1, box_2):

        """
        Compute the intersection of two n-dimensional boxes.

        Parameters
        ----------
        box_1: narray of shape (n_feature,2)
        box_2: narray of shape (n_feature,2)
        """
        max_coord = np.maximum(box_1[:,0], box_2[:,0])
        min_coord = np.minimum(box_1[:,1], box_2[:,1])

        differences = max_coord-min_coord

        differences *= -1 if all(differences<0) else 1

        zeros_mask = np.zeros(differences.shape[0])

        differences = np.maximum(differences, zeros_mask)

        intersection = np.prod(differences)

        return intersection


    def compute_iou(self, box_1, box_2):

        """
        Compute the Intersection over Union of two n-dimensional boxes.

        Parameters
        ----------
        box_1: narray of shape (n_feature,2)
        box_2: narray of shape (n_feature,2)
        """

        intersection = self._intersection(box_1,box_2)
        union = self._area(box_1)+self._area(box_2)-intersection
        return intersection/union


class LeafBoxesNMS():

    selector = MultidimensionalIOU()

    def __init__(self):
        self.true_leaves = []


    def filter(self, garden_leaves):

        garden_size = len(garden_leaves)

        for i in range(garden_size):

            matches = [i]
            for j in range(garden_size):

                box_1 = garden_leaves[i]["box"]
                box_2 = garden_leaves[j]["box"]

                if self.selector.compute_iou(box_1,box_2)>0.5:
                    matches.append(j)

            if len(matches)>1:
                values = [garden_leaves[i]["value"] for i in matches]
                probs = [values.count(value)/len(values) for value in set(values)]

                if any([prob>0.9 for prob in probs]):
                    self.true_leaves.append(min(matches))

        self.true_leaves = [garden_leaves[i] for i in set(self.true_leaves)]
