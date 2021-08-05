import os
import csv
import numpy as np


class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def load_data(data_file_name, target_type):


    module_path = os.path.dirname(__file__)
    file_path = os.path.join(module_path,'data',data_file_name)

    with open(file_path) as csv_file:
        n_samples = sum(1 for line in csv_file)-1

    with open(file_path) as csv_file:

        data_file = csv.reader(csv_file)

        feature_names = next(data_file)
        n_features = len(feature_names)-1

        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=target_type)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=target_type)

    return Bunch(data=data, target=target, feature_names=feature_names)


def load_iris():

    """
    Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    """

    return load_data('iris.csv',target_type=np.int)


def load_wine():

    """
    Load and return the wine dataset (classification).

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============
    """

    return load_data('wine_data.csv',target_type=np.int)


def load_breast_cancer():

    """
    Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============
    """

    return load_data('breast_cancer.csv',target_type=np.int)


def load_boston():

    """
    Load and return the diabetes dataset (regression).

    ==============   ==================
    Samples total    442
    Dimensionality   10
    Features         real, -.2 < x < .2
    Targets          integer 25 - 346
    ==============   ==================
    """

    return load_data('boston_house_prices.csv',target_type=np.float64)


def load_diabetes():

    """
    Load and return the boston house-prices dataset (regression).

    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features         real, positive
    Targets           real 5. - 50.
    ==============   ==============
    """

    return load_data('diabetes_data.csv',target_type=np.float64)
