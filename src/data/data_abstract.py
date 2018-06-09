import pandas as pd
from abc import ABC, abstractmethod


"""
Standardized DataAbstract for KSU project. 
Changes to this class can take place after group discussion to discover dependency 
impacts, otherwise override methods in the model classes as needed.
"""


class DataAbstract(ABC):
    def __init__(self):
        # Load data from source_data folder
        self.data = pd.read_csv("src/data/source_data/Series3_6.15.17_padel.csv", index_col=0)

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def clean_data(data):
        return

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def engineer_features(x_data):
        return

    @staticmethod  # staticmethod to allow easier sharing
    @abstractmethod
    def test_train_split(x_data, y_data):
        return
