from abc import ABC, abstractmethod

"""
Abstract class for all custom data classes (ex. data_non_linear.py)
"""


class DataAbstract(ABC):

    @staticmethod
    @abstractmethod
    def clean_data(data):
        """
        Example implementation steps:
          * Clean data
          * Split x, y
          * Remove/impute missing data (may need separate method if complex)
          * one-hot-encoding if applicable
          * Check for missing values
        :param data: raw data from source_data folder, static to allow sharing
        :return: x_data, y_data
        """
        return

    @staticmethod
    @abstractmethod
    def engineer_features(x_data):
        """
        Example implementation steps:
          * Perform feature engineering
          * Check for unexpected values
        :param x_data:
        :return: return x_data with new features
        """
        return

    @abstractmethod
    def test_train_split(self, x_data, y_data):
        """
        Example implementation steps:
          * Scale/normalize
          * Split train/test based on missing target variables
        :param x_data:
        :param y_data:
        :return: x_train, x_test, y_train
        """
        return
