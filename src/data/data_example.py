import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataNonLinear:

    def __init__(self):
        # Load data from source_data folder
        self.train = pd.read_csv("src/data/source_data/Series3_6.15.17_padel.csv", index_col=0)
        self.test = None  # Missing IC50, Series3 examples

    def clean_data(self, train):
        """ Clean data """
        # Eliminate features without variance
        train = train.loc[:, (train.std() > 0).values]
        # Seperate Series 3 test when IC50 is null
        test_index = train.IC50.isnull()
        x_test = train.loc[test_index]  # TODO implement
        train = train.loc[~test_index]
        # Remove columns with missing data
        train = train.dropna(axis=1)
        # Transform discrete with one-hot-encoding
        int_cols = train.columns[train.dtypes == 'int64']
        float_cols = train.columns[train.dtypes == 'float64']
        one_hot_df = pd.get_dummies(train[int_cols].astype('O'))
        train = pd.merge(train[float_cols], one_hot_df, left_index=True, right_index=True)
        # Split x, y
        y_train = train.pop("IC50")
        # y_class = pd.Series(data=[int(y < 2.1) for y in y_data])
        x_train = train.copy()
        # Ensure no (+/-) inf or nan due to improper transformation
        x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        assert not sum(x_train.isna().sum()), "Unexpected nulls found"  # TODO add to unit-tests
        # TODO assert not isinfinite instead of only swapping here ^^

        return x_train, y_train, x_test

    def engineer_features(self, x_train, x_test):
        """Engineer features"""
        # Combine train and test set
        # x_data =
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

    """Return train and test set ready for model"""
    # Normalize
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_data.loc[:, :] = x_scaler.fit_transform(x_data)
    y_data.loc[:] = np.squeeze(y_scaler.fit_transform(y_data.values.reshape(-1, 1)))


