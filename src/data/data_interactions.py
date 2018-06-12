from src.data.lib.interactions import InteractionChecker
from src.data.data_abstract import DataAbstract
from src.data.data_non_linear import DataNonLinear
from src.data.lib.dummy_scaler import DummyScaler
import pandas as pd
from numpy import squeeze

class DataInteractions(DataAbstract):
    def __init__(self):
        DataAbstract.__init__(self)
        self.cache = True

    @staticmethod
    def clean_data(data):
        """
        Seperate into x and y data
        """
        # Convert string data to nan
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.loc[:, (data.std() > 0).values]
        # Split x, y
        y_data = data.pop("IC50")
        x_data = data.copy().fillna(0)
        return x_data, y_data

    @staticmethod
    def engineer_features(x_data, y_data=None):
        """
        Example implementation steps:
          * Perform feature engineering (use additional methods as needed, or static file)
          * Check for unexpected values
        :param x_data:
        :return: return x_data with new features
        """
        print("Finding interactions....")
        ic = InteractionChecker(alpha=0.01)
        ic.fit(x_data[~y_data.isna()].fillna(0), y_data[~y_data.isna()])
        interactions = ic.transform(x_data.fillna(0))
        transformations = DataNonLinear().engineer_features(x_data.fillna(0))
        return pd.concat([transformations,interactions],axis=1)

    @staticmethod
    def test_train_split(x_data, y_data):
        """
          * Scale using MinMaxScaler
          * Split train/test based on missing target variables
        :param x_data:
        :param y_data:
        :return: x_train, x_test, y_train
        """

        test_index = y_data.isnull()
        x_train = x_data.loc[~test_index].copy()
        y_train = y_data.loc[~test_index].copy()
        x_test = x_data.loc[test_index].copy()

        y_scaler = DummyScaler()
        return x_train, x_test, y_train, y_scaler
