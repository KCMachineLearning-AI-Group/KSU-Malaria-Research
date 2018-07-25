from src.data.data_abstract import DataAbstract
from src.data.data_simple import DataSimple
from scipy import stats as st

"""
Template for data classes in the KSU project.
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class DataKSFiltered(DataAbstract):
    def __init__(self):
        self.cache_filename = "data_ks_filtered"  # Optional, swap None with a string for caching capabilities
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
        x_data = DataSimple.engineer_features(x_data)
        x_data = x_data.loc[:, x_data.std() > 0]
        x_data = x_data.loc[:, x_data.nunique() > 3]

        x_train, x_test, y_train, y_scaler = DataSimple.test_train_split(x_data, y_data)

        filtered_features = list()
        a = 0.10

        for f in x_data.dropna(axis=1).columns:
            ks = st.ks_2samp(x_test.loc[:, f], x_train.loc[:, f])
            # if p-value > a, add to list
            if ks[1] > a:
                filtered_features.append(f)

        return x_data[filtered_features]

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
