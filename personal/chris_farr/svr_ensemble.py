import pandas as pd
import numpy as np
from src.model_validation import ModelValidation
from src.data.data_best_svr import DataBestSVR
from sklearn.svm import LinearSVR
# Pull robust_features.csv
# Create model from one column
# Make prediction...

# Load data
data_class = DataBestSVR()
x_data, y_data = data_class.clean_data(data_class.data)
# x_data = data_class.engineer_features(x_data, y_data)
# Create the features required
feat_df = pd.read_csv("personal/chris_farr/robust_features.csv", index_col=0).iloc[:, 1:]
feature_list = list(feat_df.iloc[feat_df.iloc[:, -1].astype(bool).values, -1].index)
for feat1, feat2 in [feat.split("*") for feat in feature_list if "*" in feat]:
    x_data[feat1 + "*" + feat2] = x_data[feat1] * x_data[feat2]
x_train, x_test, y_train, y_scaler = data_class.test_train_split(x_data, y_data)
# Analyze the CV prediction for each

x_train = x_train.loc[:, feature_list]

x_train = x_train.loc[:, x_train.std() != 0]


model = LinearSVR(random_state=0)

validation = ModelValidation()

validation.score_regressor(x_train, y_train, model, y_scaler, pos_split=y_scaler.transform([[2.1]]))

# TODO Update robust_features.csv to include only features with >0 STD, narrow to only selected


