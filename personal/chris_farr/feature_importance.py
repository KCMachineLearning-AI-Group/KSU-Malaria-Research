# Combine run files
import pandas as pd
import numpy as np
from src.data.data_ks_filtered import DataKSFiltered
from sklearn.svm import LinearSVR
from src.model_validation import ModelValidation
from matplotlib import pyplot as plt

validation = ModelValidation()
# Pull and combine runs
df = pd.read_csv("personal/chris_farr/features.csv", index_col=0)
# df = pd.concat([feat_df_1, feat_df_2], axis=1)
# Create coef_df for new values
coef_df = pd.DataFrame(index=df.index)

# TODO filter out low performing models (remove outliers?)
# Construct data
data_class = DataKSFiltered()
x_train, x_test, y_train, y_scaler = data_class.load_data()

# Loop through each run
for i in range(df.shape[1]):
    # Extract features
    features = df.loc[df.iloc[:, i].apply(bool)].index.values
    col_name = df.columns[i]
    col_split = col_name.split("_")
    # Assert feature length
    assert len(features) == int(col_split[0]), "Wrong column length"
    # Create model
    model = LinearSVR(random_state=0)
    # Assert model performance matches
    score_dict = validation.score_regressor(x_train[features], y_train, model, y_scaler, pos_split=y_scaler.transform([[2.1]]), verbose=0)
    if np.mean(score_dict["r2_score"]) < .75:
        continue
    assert np.isclose(np.mean(score_dict["root_mean_sq_error"]), float(col_split[2]), rtol=.001), "Wrong score"
    # Train model
    model.fit(x_train[features], y_train)
    # Replace bool with coefficient in df
    model_coef = pd.DataFrame(data=model.coef_, index=features)
    model_coef.rename(columns={0: df.columns[i]}, inplace=True)
    coef_df = pd.merge(coef_df, model_coef, how="left", left_index=True, right_index=True).fillna(0)


# Top 10 by frequency selected
selected_percent = coef_df.apply(lambda x: x != 0, axis=0).sum(1) / coef_df.shape[1]
selected_percent.sort_values(ascending=False, inplace=True)

# Find the top 10 features by average coefficient
selected_mean = coef_df.mean(1)

# Confidence intervals for each feature
# z * 2.576
# 2.576 * (coef_df.std(1) / sqrt(coef_df.shape[1])
max_99_coef = coef_df.mean(1) + (2.576 * (coef_df.std(1) / np.sqrt(coef_df.shape[1])))
min_99_coef = coef_df.mean(1) - (2.576 * (coef_df.std(1) / np.sqrt(coef_df.shape[1])))
summary_df = pd.concat([selected_percent, min_99_coef, selected_mean, max_99_coef], axis=1)
summary_df.columns = ["selected_percent", "min_99_coef", "coef_mean", "max_99_coef"]
# TODO Start here! Finish creating the above and get them ready for a final report
# Are there any in agreement across the runs for pos/neg and low standard deviation?

summary_df.sort_values(by="selected_percent", ascending=False, inplace=True)
summary_df.to_csv("feature_importance.csv")
coef_df.to_csv("feature_coefficients.csv")

