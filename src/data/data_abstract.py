import pandas as pd
from abc import ABC, abstractmethod
from os.path import isfile
from pickle import load, dump

"""
Standardized DataAbstract for KSU project.
Changes to this class can take place after group discussion to discover dependency
impacts, otherwise override methods in the model classes as needed.
"""


class DataAbstract(ABC):
    def __init__(self, cache_filename=None):
        # Load data from source_data folder
        self.data = pd.read_csv("src/data/source_data/Series3_6.15.17_padel.csv", index_col=0)
        self.cache_filename = cache_filename

    """
    Non-Abstract Methods, override if necessary
    """

    def load_data(self):
        filename = "src/data/cached_data/{}.pkl".format(self.cache_filename)
        if self.cache_filename is not None and isfile(filename):
            print("loading from cache....")
            x_train, x_test, y_train, y_scaler = load(open(filename, "rb"))
        else:
            x_data, y_data = self.clean_data(self.data)
            x_data = self.engineer_features(x_data, y_data)
            x_train, x_test, y_train, y_scaler = self.test_train_split(x_data, y_data)

            if self.cache_filename is not None:
                print("caching....")
                dump((x_train, x_test, y_train, y_scaler), open(filename, "wb"))

        return x_train, x_test, y_train, y_scaler

    """
    Abstract Methods, these must be overriden
    """

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def clean_data(data):
        x_data, y_data = None, None
        return x_data, y_data

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def engineer_features(x_data, y_data=None):
        return x_data

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def test_train_split(x_data, y_data):
        x_train, x_test, y_train, y_scaler = None, None, None, None
        return x_train, x_test, y_train, y_scaler
