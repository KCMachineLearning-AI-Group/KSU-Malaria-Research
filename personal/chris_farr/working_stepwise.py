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
import random
# Load data
from src.data.data_non_linear import DataNonLinear
from src.model_validation import ModelValidation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

validation = ModelValidation()
data_class = DataNonLinear()
# data = data_class.data
# x_data, y_data = data_class.clean_data(data)
# x_train, x_test, y_train, y_scaler = data_class.test_train_split(x_data, y_data)

x_train, x_test, y_train, y_scaler = data_class.load_data()

# Group correlated features
corr_threshold = .98
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

# # Starting Point A: Read a csv file
# # Upload a starting point from a csv dataframe with no index
# feature_df = pd.read_csv("src/models/support/mixed_stepwise_features.csv")
# feature_list = list(np.squeeze(feature_df.values))
# # Find the dict key for each feature and add to the list
# for feat in feature_list:
#     for group in corr_dict.keys():
#         if feat in corr_dict[group]["out"]:
#             corr_dict[group]["in"].add(feat)
#             corr_dict[group]["out"].remove(feat)
#             break

# Starting Point B: randomly select features from half of the correlation groups (or arbitrary number of them)
# Start with 100 features
choices = np.random.choice(range(len(corr_dict)), size=150, replace=False)
for c in choices:
    # if not len(corr_dict[c]["out"]):  # Ensure there are more to add from group
    corr_dict[c]["in"].add(corr_dict[c]["out"].pop())

# Set model for selection
model = LinearSVR(random_state=0)

# pprint.pprint(corr_dict)

no_improvement_count = 0
last_benchmark = np.inf

for i in range(10000):
    # pass
    # Every other loop add/remove
    # Select for add by correlation group, one from random selection of them
    # Select for removal a random sample up to the size of in_features
    # TODO add parallel if no improve for > 10

    batch_size = 10

    # Extract selected features from corr_dict, create new dict with feat as key and group as value
    in_features = dict([(feat, i) for i, group in corr_dict.items() for feat in group["in"]])

    # Measure model benchmark
    benchmark = validation.score_regressor(x_train.loc[:, in_features.keys()], y_train, model, y_scaler,
                                           pos_split=y_scaler.transform([[2.1]]), verbose=0)
    benchmark = np.mean(benchmark["root_mean_sq_error"])
    if benchmark >= last_benchmark:
        no_improvement_count += 1
    else:
        no_improvement_count = 0
    print("New Benchmark RMSE:", '{0:.2f}'.format(benchmark), " iteration: ", i, " no improve: ", no_improvement_count,
          " feats: ", len(in_features))
    last_benchmark = benchmark

    batch_size += 5 * no_improvement_count
    if no_improvement_count > 200:
        print("Early stopping....")
        break

    if i % 2 != 0:
        # Remove features
        # * Test the individual removal of a number of features, each from a different correlation group.
        # Max this out at the number of features or close to for batch_size min(n_feats, batch_size)
        test_feats_for_removal = dict()
        # Choose testing features randomly
        choices = np.random.choice(range(len(in_features)), min(batch_size, len(in_features)))
        for i_, feat in enumerate(in_features.keys()):
            if i_ in choices:
                test_feats_for_removal[feat] = in_features[feat]

        remove_dict = {}
        for feat, corr_group in test_feats_for_removal.items():  # Loop through dict keys

            # Evaluate after removal and log results
            test_features = [f for f in list(in_features.keys()) if f != feat]
            results = validation.score_regressor(x_train.loc[:, test_features], y_train, model, y_scaler,
                                                 pos_split=y_scaler.transform([[2.1]]), verbose=0)
            new_score = np.mean(results["root_mean_sq_error"])
            remove_dict[feat] = new_score

        # Find the best of those tested
        final_removal = sorted(remove_dict, key=remove_dict.get, reverse=True)[-1]
        # Remove and update corr dict if improves score
        if remove_dict[final_removal] < benchmark:
            corr_dict[test_feats_for_removal[final_removal]]["out"].add(final_removal)
            corr_dict[test_feats_for_removal[final_removal]]["in"].remove(final_removal)

    if i % 2 == 0:
        # Add feature
        # TODO Parallelize this when no improve > 10
        test_feats_for_addition = dict()
        # Pick random group
        choices = np.random.choice(range(len(corr_dict)), batch_size * 10)
        k = 0
        # Test if any are out and pick one randomly
        for c in choices:
            if len(corr_dict[c]["out"]) > 0:
                feat = random.sample(corr_dict[c]["out"], 1)[0]  # Pull random feature
                test_feats_for_addition[feat] = c  # Store the corr group
                k += 1
            if k == batch_size:
                break
        # Randomly choose 10 features from out_features

        # from joblib import Parallel, delayed
        #
        # # TODO par
        # n_jobs = 7
        # par_results = Parallel(n_jobs=n_jobs)(
        #     delayed(validation.score_regressor(x_train.loc[:, list(in_features.values()) + [feat]],
        #                                        y_train, model, y_scaler,
        #                                        pos_split=y_scaler.transform([[2.1]]),
        #                                        verbose=0))(feat) for feat in test_feats_for_addition.items())
        # https://stackoverflow.com/questions/22878743/how-to-split-dictionary-into-multiple-dictionaries-fast
        # How to keep together with feat?
        add_dict = {}
        for feat, corr_group in test_feats_for_addition.items():
            # Evaluate addition and log results
            test_features = list(in_features) + [feat]
            results = validation.score_regressor(x_train.loc[:, test_features], y_train, model, y_scaler,
                                                 pos_split=y_scaler.transform([[2.1]]), verbose=0)
            new_score = np.mean(results["root_mean_sq_error"])
            add_dict[feat] = new_score

            # Find the best of those tested
        final_addition = sorted(add_dict, key=add_dict.get, reverse=True)[-1]
        # Add and update corr dict if improves score
        if add_dict[final_addition] < benchmark:
            corr_dict[test_feats_for_addition[final_addition]]["in"].add(final_addition)
            corr_dict[test_feats_for_addition[final_addition]]["out"].remove(final_addition)

# Test for stop

# TODO number of features needs to drift. Only change if improves.

in_features = dict([(feat, i) for i, group in corr_dict.items() for feat in group["in"]])

in_features.keys()
# Final validation
model = LinearSVR(random_state=0)
results = validation.score_regressor(x_train.loc[:, in_features], y_train, model, y_scaler,
                                     pos_split=y_scaler.transform([[2.1]]))
# Save the feature names in a csv
selected_features = pd.DataFrame(list(in_features.keys()), columns=["features"])
selected_features.to_csv("src/models/support/mixed_stepwise_features_non_linear.csv", index=False)

"""
with 3 splits and 10 repeats
average r2_score: 0.9444898240679163
average root_mean_sq_error: 6.599426503362611
average explained_variance: 0.9529389554386497
average mean_sq_error: 47.53500411504506
average mean_ae: 5.011080004708307
average median_ae: 4.037490339305328

with 3 splits and 10 repeats
average r2_score: 0.9555241154896782
average root_mean_sq_error: 5.794542669371337
average explained_variance: 0.9631871015751845
average mean_sq_error: 38.54550842020435
average mean_ae: 4.246830260782811
average median_ae: 3.168010696278745
"""

"""
Non-linear transformations added, started over

with 3 splits and 10 repeats
average r2_score: 0.9692269177991055
average root_mean_sq_error: 4.510349621318988
average explained_variance: 0.9732512433123808
average mean_sq_error: 23.24435271307241
average mean_ae: 3.4382322072605125
average median_ae: 2.712202668633315

with 3 splits and 10 repeats
average r2_score: 0.9700143727174664
average root_mean_sq_error: 4.430248948092821
average explained_variance: 0.974132617465656
average mean_sq_error: 22.62275261201872
average mean_ae: 3.3568956545556157
average median_ae: 2.5991535583753516

"""

# TODO After optimal is found, are there any groups with many features included? (highly correlated)
# TODO How do the results vary when using higher vs lower correlation groups? (95 vs 99)
# TODO What models improve the score? Can they be swapped out with SVM for selection?
# adaboost: .22 R2


RANDOM_STATE = 36851234

# AdaBoost
from sklearn.ensemble import BaggingRegressor

model = BaggingRegressor(LinearSVR(random_state=0), max_samples=.99, max_features=.99)

params = {
    "n_estimators": list(range(1, 20, 1)),
}

cv = ModelValidation().get_cv(x_train, y_train, pos_split=y_scaler.transform([[2.1]]))

grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, verbose=1, n_jobs=7,
                    scoring=make_scorer(mean_squared_error, greater_is_better=False))

grid.fit(x_train.loc[:, in_features], y_train)
grid.best_params_
grid.best_score_
from sklearn.svm import SVR

# TODO start here!!! Tune linear SVM


model = BaggingRegressor(LinearSVR(random_state=0), n_estimators=25, max_samples=.99, max_features=.99)

results = validation.score_regressor(x_train.loc[:, in_features], y_train, model, y_scaler,
                                     pos_split=y_scaler.transform([[2.1]]))
