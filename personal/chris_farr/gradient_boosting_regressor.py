from src.data.data_step_interactions import DataStepInteractions
from src.models.model_mixed_stepwise import ModelMixedStepwise
from src.model_validation import ModelValidation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer
import numpy as np
import pandas as pd

# Load/clean/transform data used in the model mixed stepwise class
data_object = DataStepInteractions()
x_train, x_test, y_train, y_scaler = data_object.load_data()
# Select features
model_object = ModelMixedStepwise()
selected_features = model_object.select_features(None)
x_train = x_train.loc[:, selected_features]
x_test = x_test.loc[:, selected_features]
# Baseline score the model
model = GradientBoostingRegressor(random_state=0, max_depth=1, n_estimators=460)
ModelValidation().score_regressor(x_train, y_train, model, y_scaler, pos_split=y_scaler.transform([[2.1]]))

# Tune the gradient boosting regressor


params = {
    "n_estimators": list(range(10, 500, 50)),
}

cv = ModelValidation().get_cv(x_train, y_train, pos_split=y_scaler.transform([[2.1]]))

grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, verbose=1, n_jobs=3,
                    scoring=make_scorer(r2_score, greater_is_better=True))

grid.fit(x_train, y_train)
grid.best_params_
grid.best_score_

