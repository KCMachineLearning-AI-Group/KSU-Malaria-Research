import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale

from src.data.data_abstract import DataAbstract

class DataDimensionReduction(DataAbstract):

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
        x_data, y_data = DataNonLinear.clean_data(data)
        return x_data, y_data

    @staticmethod
    def engineer_features(x_data):
        """Engineer features"""
        # Perform PCA to reduce the dimensions
        x_data = scale(x_data, with_mean=True)
        x_scaler = StandardScaler()
        x_data = x_scaler.fit_transform(x_data)
        print("performing dimension reduction using PCA....\n")
        reduce_dim = PCA(n_components=100)
        x_data_pca = reduce_dim.fit_transform(x_data)

        return pd.DataFrame(data = x_data_pca)

    @staticmethod
    def test_train_split(x_data, y_data):
        y_scaler = StandardScaler()
        # Seperate Series 3 test when IC50 is null
        test_index = y_data.isnull().values
        x_train = x_data.loc[~test_index].copy()
        y_train = y_data.loc[~test_index].copy()
        x_test = x_data.loc[test_index].copy()
        """Return train and test set ready for model"""
        # Normalize
        y_train.loc[:] = np.squeeze(y_scaler.fit_transform(y_train.values.reshape(-1, 1)))
        return x_train, x_test, y_train, y_scaler
