import numpy as np
import pandas as pd
from src.model_validation import ModelValidation
from personal.chris_farr.mixed_stepwise_selection import MixedStepSelect
from src.data.data_ks_filtered import DataKSFiltered
from random import randint
from sklearn.svm import LinearSVR
# Setup
data_class = DataKSFiltered()
x_train, x_test, y_train, y_scaler = data_class.load_data()
validation = ModelValidation()
predictions_file = "personal/chris_farr/predictions.csv"
features_file = "personal/chris_farr/features.csv"
model = LinearSVR(random_state=0)

# TODO Run mixed select
for _ in range(100):
    starting_features = randint(5, int(len(x_train.columns) / 2))
    ms = MixedStepSelect(corr_threshold=.99, data_class=data_class, n_start_feats=starting_features)
    ms.model = model
    ms.run(1000)
    in_features = ms.in_features

    # TODO Store results in CSV

    # TODO randomize attributes with each run

    # Final scoring and storage

    # Final validation
    np.set_printoptions(suppress=True)
    results = validation.score_regressor(x_train.loc[:, ms.in_features], y_train, model, y_scaler,
                                         pos_split=y_scaler.transform([[2.1]]))

    predictions = y_scaler.inverse_transform(
        model.fit(x_train.loc[:, in_features], y_train).predict(x_test.loc[:, in_features]))

    # Save the feature names in a csv
    selected_features = pd.DataFrame(list(in_features.keys()), columns=["features"])
    # selected_features.to_csv("src/models/support/best_features.csv", index=False)

    # TODO run many different times, store the columns select, the test predictions, and the performance scores

    # Set name for round
    round_name = "{}_feats_{:.2f}_rmse".format(len(in_features), np.mean(results["root_mean_sq_error"]))
    # Read files
    new_test_prediction_df = pd.DataFrame(columns=[round_name], data=predictions, index=x_test.index)
    new_selected_features_df = pd.DataFrame(columns=[round_name], data=[1] * len(selected_features), index=selected_features.features)
    try:
        test_prediction_df = pd.read_csv(predictions_file, index_col=0)
        test_prediction_df = pd.merge(test_prediction_df, new_test_prediction_df, how="outer", left_index=True,
                                      right_index=True)
        selected_features_df = pd.read_csv(features_file, index_col=0)
        selected_features_df = pd.merge(selected_features_df, new_selected_features_df, how="outer",
                                        left_index=True, right_index=True).fillna(0).astype(int)
    except FileNotFoundError:
        test_prediction_df = new_test_prediction_df
        selected_features_df = pd.DataFrame(index=x_train.columns)  # For first run
        selected_features_df = pd.merge(selected_features_df, new_selected_features_df, how="outer",
                                        left_index=True, right_index=True).fillna(0).astype(int)

    # Store files
    test_prediction_df.to_csv(predictions_file)
    selected_features_df.to_csv(features_file)
