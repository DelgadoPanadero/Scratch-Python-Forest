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
#  THE SOFTWARE.import numpy as np


class MultidimensionalIOU():

    """
    This class is a helper object aimed just to compute the IOU score
    """

    def _area(self, box):

        """
        Compute the area of a box.
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


    def compute_ioa(self, box_1, box_2):

        """
        Compute the Intersection of box_1 with box_2 over its area

        Parameters
        ----------
        box_1: narray of shape (n_feature,2)
        box_2: narray of shape (n_feature,2)
        """

        intersection = self._intersection(box_1,box_2)
        return intersection/self._area(box_1)


class LeafBoxesNMS():

    """
    This class filter the distinct leafs from the forest using the
    Non Maximum Supression algorithm.

    Parameters
    ----------
    iou_threshold: float interval (0, 1)
    confidence_threshold: float interval (0, 1)
    """

    selector = MultidimensionalIOU()

    def __init__(self, iou_threshold=0.5, confidence_threshold=0.9):

        self.true_leaves = []
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold


    def _iou_similarity_filter(self, leaf_id, garden_leaves):

        """
        Given a leaf box id, filter all the leaves from the garden that
        math with the iou score.


        Parameters
        ----------
        leaf_id: int with the current leaf box id.
        garden_leaves: list of leaves boxes returned by a BonsaiHarvester.
        """

        matches = [leaf_id]
        for j in range(len(garden_leaves)):
            if j>leaf_id:

                box_1 = garden_leaves[leaf_id]["box"]
                box_2 = garden_leaves[   j   ]["box"]

                if self.selector.compute_iou(box_1,box_2)>self.iou_threshold:
                    matches.append(j)

        return matches

    def _category_confidence_filter(self, matches, garden_leaves):

        """
        Given a list of id, compute the confidence the caterory.


        Parameters
        ----------
        matches: list of int, with the indexes of the garden leaves.
        garden_leaves: list of leaves boxes returned by a BonsaiHarvester.
        """

        if len(matches)>1:
            values = [garden_leaves[i]["value"] for i in matches]
            probs = [values.count(value)/len(values) for value in set(values)]

            if any([prob>self.confidence_threshold for prob in probs]):
                return matches


    def _filter_smallest_leaves(self, true_leaves):

        """
        Filter the smallest leaves.
        """

        matches = []
        for leaf_1 in true_leaves:
            for leaf_2 in true_leaves:
                box_1 = leaf_1["box"]
                box_2 = leaf_2["box"]
                if self.selector.compute_ioa(box_1,box_2)>0.9 and box_1 is not box_2:
                    matches.append(box_1)

        return matches


    def filter(self, garden_leaves):


        """
        Filter the distinct leaves following the NMS algorithm.

        Parameters
        ----------
        garden_leaves: list of leaves boxes returned by a BonsaiHarvester.
        """

        matches_list=[]
        for leaf_id in range(len(garden_leaves)):

            matches = self._iou_similarity_filter(leaf_id, garden_leaves)
            matches = self._category_confidence_filter(matches, garden_leaves)
            matches_list.append(matches)

        matches = [id for matches in matches_list if matches for id in matches]
        matches = [id for id in matches if matches.count(id)==1]

        self.true_leaves = [garden_leaves[id] for id in matches]
        self.true_leaves = self._filter_smallest_leaves(self.true_leaves)

        return self.true_leaves


if __name__=="__main__":

    from pprint import pprint
    from skratch.datasets import load_iris

    from tools import BonsaiHarvester
    from tree import DecisionBonsaiClassifier
    from forest import RandomGardenClassifier

    iris = load_iris()
    X = iris.data
    y = iris.target

    classifier = RandomGardenClassifier(n_estimators=100)
    m = classifier.fit(X, y)

    harvester = BonsaiHarvester(X)
    harvester.harvest_garden(classifier)

    leaf_boxes_nms = LeafBoxesNMS()
    leaf_boxes_nms.filter(harvester.leaf_boxes)

    print(len(harvester.leaf_boxes))
    print(len(leaf_boxes_nms.true_leaves))
