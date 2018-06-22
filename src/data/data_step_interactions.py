from src.data.data_abstract import DataAbstract
from src.data.data_non_linear import DataNonLinear
from src.data.data_simple import DataSimple
from src.data.util.interactions import InteractionChecker
import pandas as pd
import numpy as np

"""
Template for data classes in the KSU project.
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class DataStepInteractions(DataAbstract):
    def __init__(self):
        self.cache_filename = "data_step_interactions"  # Optional, swap None with a string for caching capabilities
        DataAbstract.__init__(self, self.cache_filename)

    @staticmethod
    def clean_data(data):
        """
        Example implementation steps:
          * Clean data
          * Split x, y
          * Remove/impute missing data (use additional methods as needed)
          * one-hot-encoding if applicable
          * Check for missing values
        :param data: raw data from source_data folder
        :return: x_data, y_data
        """

        # TODO Implement custom clean_data, or steal
        x_data, y_data = DataSimple.clean_data(data)
        return x_data, y_data

    @staticmethod
    def engineer_features(x_data, y_data=None):
        """
        Example implementation steps:
          * Perform feature engineering (use additional methods as needed, or static file)
          * Check for unexpected values
        :param x_data:
        :param y_data:
        :return: return x_data with new features
        """

        # TODO Implement custom engineer_features, or steal

        feature_df = pd.read_csv("src/models/support/mixed_stepwise_features.csv")
        feature_list = list(np.squeeze(feature_df.values))

        # Find interactions of only features from the mixed stepwise feature list
        ic = InteractionChecker(alpha=.01)
        x_train, x_test, y_train, y_scaler = DataSimple.test_train_split(x_data, y_data)
        ic.fit(x_train.loc[:, feature_list], y_train)
        interactions = ic.transform(x_data.loc[:, feature_list])

        # Combine x_data and interactions
        x_data = pd.merge(x_data, interactions, left_index=True, right_index=True)

        return x_data

    @staticmethod
    def test_train_split(x_data, y_data):
        """
        Example implementation steps:
          * Scale/normalize
          * Split train/test based on missing target variables
        :param x_data:
        :param y_data:
        :return: x_train, x_test, y_train
        """

        # TODO Implement custom test_train_split, or steal
        x_train, x_test, y_train, y_scaler = DataSimple.test_train_split(x_data, y_data)
        return x_train, x_test, y_train, y_scaler
