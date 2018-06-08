import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.svm import LinearSVR
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear
# TODO start here!! Finish the get_data method, decide if it should be static or not
# TODO finish implementing ModelCorrelationGrouper and abstract, then move on to leaderboard.py


class ModelCorrelationGrouper(ModelAbstract):

    def __init__(self):
        self.data_object = DataNonLinear()
        self.selected_features = []

    def get_validation_support(self):
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        self.selected_features = self.select_features(x_data)
        x_data = x_data.loc[:, self.selected_features].copy()
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)
        model = self.choose_model(x_train, y_train)
        return x_train, y_train, x_test, model

    def get_test_prediction(self):
        x_train, y_train, x_test, model = self.get_validation_support()
        return self.predict_test(x_train, y_train, x_test, model)

    def get_data(self):
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        # Add non-linear transformations to x_data
        x_data = self.data_object.engineer_features(x_data)
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)
        return x_train, x_test, y_train

    @staticmethod
    def select_features(x_data) -> set:
        # Group features by correlation, select 1 from each group randomly
        corr_threshold = .95
        corr_matrix = x_data.corr()
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
                already_in.update(set([col]))
                corr_result.append([col])

        selected_feats = set([feats[0] for feats in corr_result])

        return selected_feats

    @staticmethod
    def choose_model(x_train, y_train):
        # model = AdaBoostRegressor(random_state=0)
        # params = {
        #     "n_estimators": list(range(4, 20, 3)),
        #     "learning_rate": list(np.arange(0.25, .55, .05)),
        #     "loss": ["linear", "square", "exponential"]
        # }
        # grid = GridSearchCV(model, param_grid=params, scoring=make_scorer(r2_score, greater_is_better=True), cv=10,
        #                     n_jobs=3)
        # grid.fit(x_train, y_train)
        # print("Model Params:")
        # print(grid.best_params_)
        model = LinearSVR(random_state=0, C=.05)
        return model

    @staticmethod
    def predict_test(x_train, y_train, x_test, model):
        # TODO move to abstract
        model.fit(x_train, y_train)
        return model.predict(x_test)
