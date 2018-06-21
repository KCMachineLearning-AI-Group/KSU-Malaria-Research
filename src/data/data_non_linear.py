import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.data_abstract import DataAbstract


class DataNonLinear(DataAbstract):

    @staticmethod
    def clean_data(data):
        """ Clean data """
        # Eliminate features without variance
        data = data.loc[:, (data.std() > 0).values]
        # Split x, y
        y_data = data.pop("IC50")
        # y_class = pd.Series(data=[int(y < 2.1) for y in y_data])
        x_data = data.copy()
        # Remove columns with missing data
        x_data = x_data.dropna(axis=1)
        # Transform discrete with one-hot-encoding
        int_cols = x_data.columns[x_data.dtypes == 'int64']
        float_cols = x_data.columns[x_data.dtypes == 'float64']
        one_hot_df = pd.get_dummies(x_data[int_cols].astype('O'))
        x_data = pd.merge(x_data[float_cols], one_hot_df, left_index=True, right_index=True)
        # Ensure no (+/-) inf or nan due to improper transformation
        x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        assert not sum(x_data.isna().sum()), "Unexpected nulls found"  # TODO add to unit-tests
        return x_data, y_data

    @staticmethod
    def engineer_features(x_data, y_data=None):
        """Engineer features"""
        # Perform feature engineering on float columns
        print("performing non-linear transformations....\n")
        for feat in x_data.columns[x_data.dtypes == 'float64']:
            feature_df = x_data.loc[:, feat]
            if feature_df.min() > 0:  # Avoid 0 or negative
                x_data.loc[:, feat + "_log"] = feature_df.apply(np.log)  # log
                x_data.loc[:, feat + "_log2"] = feature_df.apply(np.log2)  # log2
                x_data.loc[:, feat + "_log10"] = feature_df.apply(np.log10)  # log10
                x_data.loc[:, feat + "_cubert"] = feature_df.apply(
                    lambda x: np.power(x, 1 / 3))  # cube root
                x_data.loc[:, feat + "_sqrt"] = feature_df.apply(np.sqrt)  # square root
            # Avoid extremely large values, keep around 1M max
            if feature_df.max() < 13:
                x_data.loc[:, feat + "_exp"] = feature_df.apply(np.exp)  # exp
            if feature_df.max() < 20:
                x_data.loc[:, feat + "_exp2"] = feature_df.apply(np.exp2)  # exp2
            if feature_df.max() < 100:
                x_data.loc[:, feat + "_cube"] = feature_df.apply(
                    lambda x: np.power(x, 3))  # cube
            if feature_df.max() < 1000:
                x_data.loc[:, feat + "_sq"] = feature_df.apply(np.square)  # square
        return x_data

    @staticmethod
    def test_train_split(x_data, y_data):
        # Create scaler objects
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # Seperate Series 3 test when IC50 is null
        test_index = y_data.isnull()
        x_train = x_data.loc[~test_index].copy()
        y_train = y_data.loc[~test_index].copy()
        x_test = x_data.loc[test_index].copy()
        """Return train and test set ready for model"""
        # Normalize
        x_train.loc[:, :] = x_scaler.fit_transform(x_train)
        x_test.loc[:, :] = x_scaler.transform(x_test)
        y_train.loc[:] = np.squeeze(y_scaler.fit_transform(y_train.values.reshape(-1,1)))
        return x_train, x_test, y_train, y_scaler
