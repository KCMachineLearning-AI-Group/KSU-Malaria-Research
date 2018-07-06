import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale

from src.data.data_abstract import DataAbstract
from src.data.data_non_linear import DataNonLinear
from streamml.streamline.transformation.flow.TransformationStream import TransformationStream

class DataTOPSIS(DataAbstract):

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
    def engineer_features(x_data, y_data):
        """Engineer features"""
        
        # Boxcox
        X = TransformationStream(x_data).flow(["scale","pca","binarize","brbm"],params={'pca__percent_variance':0.9})

        # Lasso

        # RF

        # TOPSIS

        # Top 100 features
        
        return X

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
