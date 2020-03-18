import numpy as np


class MultidimensionalNMS():

    def _area(self, box):
        """
        Compute the area of a box
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
