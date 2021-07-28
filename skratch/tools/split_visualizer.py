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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class BonsaiPathsExtractor():

    """
    Extracts all the decision paths from a Bonsai object. This object is used
    by SplitVisualizer object.
    """

    def _recurse_crawl(self, graph, depth=0, path=[]):

        """
        A recursive function parses a graph from a trained Bonsai by parsing
        from each node from the first node and calling again the fuction for
        its sons.

        Parameters
        ----------
        graph: Nested dictionary. Bonsai graph or Bonsai graph node.
        depht: integer with the current graph depth.
        path: list of decisions made for the current node
        """

        if 'left_node' in graph and 'right_node' in graph:

            feature   = graph['feature']
            threshold = graph['threshold']

            cond = feature in self._features
            path_left  = path+[[feature,'=<',threshold]] if cond else path
            path_rigth = path+[[feature,'>',threshold]]  if cond else path

            self._recurse_crawl(graph['left_node' ], depth+1, path_left )
            self._recurse_crawl(graph['right_node'], depth+1, path_rigth)

        else:
            if path and path not in self._paths:
                self._paths.append(path)


    def _prune_paths(self):

        """
        While extracting all the posible paths, some paths are not complete
        (they do not end in a leaf node). This function removes the paths that
        are no complete.
        """

        is_full_path = [True for i in self._paths]
        for i, path_i in enumerate(self._paths):
            for j, path_j in enumerate(self._paths):
                if i!=j and all(step in path_j for step in path_i):
                    is_full_path[i]=False

        self._paths = [self._paths[i] for i,j in enumerate(is_full_path) if j]


    def get_paths(self, bonsai, features):

        """
        Given a trained Bonsai and a list of features, this function returns
        all the decisions used in any node of the Bonsai that use any of those
        features.

        Parameters
        ----------
        bonsai: Bonsai object.
        features: list of features from the training data.
        """

        self._paths = []
        self._features = features
        graph = bonsai.bonsai_.graph

        self._recurse_crawl(graph)
        self._prune_paths()

        return self._paths



class SplitVisualizer():

    """
    An implementation to extract all the splits made by a Bonsai and visually
    plot them in a set of images.
    """


    _path_extractor = BonsaiPathsExtractor()
    _labels_colors = list(mcolors.BASE_COLORS)
    _splits_colors = list(mcolors.BASE_COLORS)


    def _path_to_segments(self, paths, X, pair):

        """
        Given a list of paths extracted by the BonsaiPathsExtractor object,
        where each path is a decision. This function return a list of pairs of
        points, where each pair of points define a segment to plot each
        decision.

        Parameters
        ----------
        paths: list of lists with all the decisions from the Bonsai.
        X: np.array with the training data.
        pair: list of the two features used to extract the segments.
        """

        borders_x = X[:, pair[0]].min(), X[:, pair[0]].max()
        borders_y = X[:, pair[1]].min(), X[:, pair[1]].max()

        segments = []
        for path in paths:

            x_min, x_max = borders_x
            y_min, y_max = borders_y

            for step in path:

                if step[0] == pair[0]:
                    segments.append([(step[2], step[2]),(y_min, y_max)])
                    x_min = step[2] if '>' in step[1] else x_min
                    x_max = step[2] if '<' in step[1] else x_max

                if step[0] == pair[1]:
                    segments.append([(x_min, x_max),(step[2], step[2])])
                    y_min = step[2] if '>' in step[1] else y_min
                    y_max = step[2] if '<' in step[1] else y_max

        segments = [list(x) for x in set(tuple(x) for x in segments)]

        return segments


    def _get_pair_variables(self, bonsai, X):

        """
        This function returns a list of pair features for all the features to
        generate all the decision splits.

        Parameters
        ----------
        bonsai: Bonsai object.
        X: np.array with the training data.
        """

        n_features = X.shape[-1]
        features = [i for i in range(n_features)]
        pairs = [(i,j) for i,j in zip(features[:-1],features[1:])]

        return pairs


    def plot(self, bonsai, X, y):

        """
        Plot all the decision splits from a trained Bonsai along with the
        training dataset points.

        Parameters
        ----------
        bonsai: Bonsai object.
        X: np.array with the training data.
        y: np.array with the label data.
        """

        self._n_classes = np.unique(y).shape[0]
        self._n_columns = X.shape[1]

        pairs = self._get_pair_variables(bonsai,X)

        for pair in pairs:

            paths = self._path_extractor.get_paths(bonsai, pair)
            segments = self._path_to_segments(paths, X, pair)

            if segments:
                for i, segment in enumerate(segments):
                        x_points, y_points = segment
                        plt.plot(x_points,
                                 y_points,
                                 color = self._splits_colors[i],
                                 label = str(i))

            plt.legend()

            for i, color in zip(range(self._n_classes), self._labels_colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, pair[0]],
                            X[idx, pair[1]],
                            c=color,
                            label= [i for i in range(X.shape[1])],
                            cmap=plt.cm.RdYlBu,
                            edgecolor='black',
                            s=15)

            plt.xlabel("Feature " + str(pair[0]))
            plt.ylabel("Feature " + str(pair[1]))
            plt.savefig(f'bonsai_splits_feature{pair[0]}_feature{pair[1]}.png')
            plt.clf()
