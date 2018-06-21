from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from pandas import Series
import pandas as pd
import numpy as np
from src.model_validation import ModelValidation
from src.data.data_non_linear import DataNonLinear
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.data.util.interactions import InteractionChecker
from scipy.stats import skewtest

RANDOM_STATE = 36851234

# Load supporting data
data_class = DataNonLinear()
model_class = ModelCorrelationGrouper()
validation = ModelValidation()

# Load data
x_data, y_data = data_class.clean_data(data_class.data)
# Remove highly skewed data
x_data = x_data.loc[:, x_data.apply(
    lambda x: skewtest(x)[1] > 0.00001).values]
# Split train/test
x_train, x_test, y_train, y_scaler = data_class.test_train_split(x_data, y_data)
# Remove highly correlated features (1 out of 2 times)
features = model_class.select_features(x_train, corr_threshold=.91)
x_train = x_train.loc[:, features].copy()
# Find interactions
# ic = InteractionChecker(alpha=0.001)
# ic.fit(x_train, y_train)
# interactions = ic.transform(x_train)
# Create non-linear transformations
# non_linear_trans = data_class.engineer_features(x_train, y_train)
# Combine interactions with all
# x_train = pd.concat([interactions, non_linear_trans], axis=1)
# Remove highly correlated features (2 out of 2 times)
# features = model_class.select_features(x_train, corr_threshold=.97)
# x_train = x_train.loc[:, features].copy()
# Tune model with grid search
# Get cv splits
cv = ModelValidation().get_cv(x_train, y_train, pos_split=y_scaler.transform([[2.1]]))

# AdaBoost
base_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
ada_model = AdaBoostRegressor(base_estimator=base_model, random_state=RANDOM_STATE)

# Exhaustive
# params = {
#     "base_estimator__max_features": np.arange(.005, .02, 0.001),
#     "base_estimator__max_depth": list(range(1, 10, 1)),
#     "n_estimators": list(range(10, 150, 10))
# }

params = {
    "base_estimator__max_features": np.arange(.003, .008, 0.001),
    "base_estimator__max_depth": list(range(1, 10, 2)),
    "n_estimators": list(range(50, 80, 10)),
}

grid = GridSearchCV(estimator=ada_model, param_grid=params, cv=cv, verbose=1, n_jobs=7,
                    scoring=make_scorer(r2_score, greater_is_better=True))


grid.fit(x_train, y_train)
grid.best_params_
grid.best_score_
# Extract important features through cross validation using standard splits, average importance
# Use those features in other models

""""""

x_train, x_test, y_train, y_scaler = data_class.load_data()
features = model_class.select_features(x_train)

# Generate CV indices
RANDOM_STATE = 36851234
REPEATS = 10

# def get_cv(x_data, y_data, pos_split=10):
pos_split = y_scaler.transform([[2.1]])
# create y_class series for Stratified K-Fold split at pos_split
y_class = Series(data=[int(y < pos_split) for y in y_train])
# num_splits count number of positive examples
num_splits = sum(y_class.values)
# create splits using stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=REPEATS, random_state=RANDOM_STATE)
# loop through splits
cv = [(train, test) for train, test in rskf.split(x_train, y_class)]

# AdaBoost
base_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
ada_model = AdaBoostRegressor(base_estimator=base_model, random_state=RANDOM_STATE)

# Exhaustive
params = {
    "base_estimator__max_features": np.arange(.005, .02, 0.001),
    "base_estimator__max_depth": list(range(1, 10, 1)),
    "n_estimators": list(range(10, 150, 10))
}

# Fine tuning
# params = {
#     "base_estimator__max_features": [
#         # 0.01,
#         0.011,
#         # 0.012,
#     ],
#     "base_estimator__max_depth": [
#         # 1,
#         2,
#         # 3,
#     ],
#     "n_estimators": [
#         # 81,
#         82,
#         # 83,
#     ]
# }

grid = GridSearchCV(estimator=ada_model, param_grid=params, cv=cv, verbose=1, n_jobs=7,
                    scoring=make_scorer(r2_score, greater_is_better=True))

grid.fit(x_train.loc[:, features], y_train)
# Tune the base estimator depth and n_estimators
grid.best_score_

best_model = grid.best_estimator_

validation.score_regressor(x_train.loc[:, features],
                           y_train, best_model,
                           pos_split=y_scaler.transform([[2.1]]))

# Best adaboost performance: .059 R2


# Best features
best_features = [feat for feat, imp in zip(features, best_model.feature_importances_) if imp != 0]
len(best_features)

# Use best features for creating interactions

ic = InteractionChecker(alpha=0.01)
ic.fit(x_train.loc[:, best_features], y_train)
interactions = ic.transform(x_train.loc[:, best_features])

combined = pd.concat([x_train.loc[:, best_features], interactions], axis=1)


validation.score_regressor(interactions,
                           y_train, best_model,
                           pos_split=y_scaler.transform([[2.1]]))
