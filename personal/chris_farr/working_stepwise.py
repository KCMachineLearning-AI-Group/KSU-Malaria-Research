"""
Mixed selection.

Problem Statement: A problem with feature selection is the unexpected change in performance when making multiple moves
at once. However, exhaustive search for the best feature to add or remove is too resource intensive with large datasets
that have thousands of columns.

Algorithm:
Group all available features into buckets by intercorrelation. Use a threshold range from 90%-100%.

As a starting point, randomly select a feature from half of the correlation groups. (or a number close to the suspected
number of final features in the best model).
Using various starting points may help avoid a local maximum.
* The starting random features will have low intercorrelation.

Test from correlation groups with the lowest representation when adding, and with the highest representation when
removing. Also include randomness to avoid getting stuck.


The algorithm should be able to increase or decrease the final number of features based on the performance.

Removal:
* Test the individual removal of a number of features, each from a different correlation group.
* Build a distribution from the tests, stop after x number of tests fall within the statistical average to show it is a
representative sample.
* If any improve the score, remove the one with the largest improvement

Addition:
* Test the addition of a number of features
* Build a distribution from the tests, stop after x number of tests fall within the statistical average to show it is a
representative sample.
* If any improve the score, add the one with the largest improvement

Stop: when loop produces no changes

Every step should increase score. The same CV splits should be used as in model validation (but fewer repeats)

"""
import numpy as np
from sklearn.svm import LinearSVR
import pprint
import pandas as pd
# Load data
from src.data.data_non_linear import DataNonLinear
from src.model_validation import ModelValidation
validation = ModelValidation()
data_class = DataNonLinear()
data = data_class.data
x_data, y_data = data_class.clean_data(data)
x_train, x_test, y_train, y_scaler = data_class.test_train_split(x_data, y_data)
# Group correlated features
corr_threshold = .95
corr_matrix = x_train.corr()
corr_matrix.loc[:, :] = np.tril(corr_matrix, k=-1)

already_in = set()
corr_result = []
for col in corr_matrix:
    correlated = corr_matrix[col][np.abs(corr_matrix[col]) > corr_threshold].index.tolist()
    if correlated and col not in already_in:
        already_in.update(set(correlated))
        correlated.append(col)
        corr_result.append(correlated)
    elif col not in already_in:
        already_in.update(set(col))
        corr_result.append([col])

# selected_feats = set([feats[0] for feats in corr_result])
len(corr_result)
# Create a feature selection dictionary:
# Contains all features, grouped by correlation
# Within each corr group there's an "in" and "out" portion for tracking selection
corr_dict = dict([(i, {"out": set(feats), "in": set([])}) for i, feats in zip(range(len(corr_result)), corr_result)])

# Starting point: randomly select features from half of the correlation groups (or arbitrary number of them)
# Start with 100 features
choices = np.random.choice(range(len(corr_dict)), size=500, replace=False)
for c in choices:
    # if not len(corr_dict[c]["out"]):  # Ensure there are more to add from group
    corr_dict[c]["in"].add(corr_dict[c]["out"].pop())

pprint.pprint(corr_dict)

for _ in range(1000):
    # Extract selected features from corr_dict, create new dict with feat as key and group as value
    in_features = dict([(feat, i) for i, group in corr_dict.items() for feat in group["in"]])

    # Measure model benchmark
    model = LinearSVR(random_state=0)
    validation.score_regressor(x_train.loc[:, in_features.keys()], y_train, model, pos_split=y_scaler.transform([[2.1]]))
    # Remove features
    # Removal:
    # * Test the individual removal of a number of features, each from a different correlation group.
    remove_dict = {}
    for _ in range(10):
        # Pick 10 current feature to remove
        remove_test = np.random.choice(list(in_features.keys()))
        # Evaluate after removal and log results
        test_features = [feat for feat in in_features if feat != remove_test]
        results = validation.score_regressor(x_train.loc[:, test_features], y_train, model, pos_split=y_scaler.transform([[2.1]]))
        new_score = np.mean(results["r2_score"])
        remove_dict[remove_test] = new_score

    # If any improve the score, remove the one with the largest improvement
    # TODO test if any improve score
    final_removal = sorted(remove_dict, key=remove_dict.get)[-1]
    # Remove and update corr dict
    corr_dict[in_features[final_removal]]["out"].add(final_removal)
    corr_dict[in_features[final_removal]]["in"].remove(final_removal)

    # Add feature
    out_features = dict([(feat, i) for i, group in corr_dict.items() for feat in group["out"]])
    # Randomly choose 10 features from out_features
    add_dict = {}
    for _ in range(10):
        # Pick 10 current feature to remove
        add_test = np.random.choice(list(out_features.keys()))
        # Evaluate addition and log results
        test_features = list(in_features) + [add_test]
        results = validation.score_regressor(x_train.loc[:, test_features], y_train, model, pos_split=y_scaler.transform([[2.1]]))
        new_score = np.mean(results["r2_score"])
        add_dict[add_test] = new_score

    # If any improve the score, add the one with the largest improvement

    final_addition = sorted(add_dict, key=add_dict.get)[-1]
    # Add and update corr dict
    corr_dict[out_features[final_addition]]["in"].add(final_addition)
    corr_dict[out_features[final_addition]]["out"].remove(final_addition)

# Test for stop

# TODO number of features needs to drift. Only change if improves.

in_features = dict([(feat, i) for i, group in corr_dict.items() for feat in group["in"]])


in_features.keys()
# Final validation
model = LinearSVR(random_state=0)
results = validation.score_regressor(x_train.loc[:, in_features], y_train, model, pos_split=y_scaler.transform([[2.1]]))
# Save the feature names in a csv
selected_features = pd.DataFrame(list(in_features.keys()), columns=["features"])
selected_features.to_csv("selected_features.csv", index=False)

"""
with 3 splits and 10 repeats
average r2_score: 0.9444898240679163
average rmse: 0.2245742394692992
"""
# TODO create new implementation and use CSV output
