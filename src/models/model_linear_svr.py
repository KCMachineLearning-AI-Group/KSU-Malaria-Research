import numpy as np
from sklearn.svm import LinearSVR
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear

class ModelLinearSVR(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataNonLinear()

    def get_validation_support(self):
        """
        Output is used for leaderboard scoring
        :return: x_train, x_test, y_train, model
        """
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        x_data = self.data_object.engineer_features(x_data)
        x_train, x_test, y_train, y_scaler = self.data_object.test_train_split(x_data, y_data)
        model = self.choose_model(x_train, y_train)
        return x_train, x_test, y_train, y_scaler, model

    @staticmethod
    def select_features(x_data):
        """
        Example workflow
        * Select columns
        * Return set or list of column names
        :param x_data: full dataset
        :return:
        """

        # TODO Implement custom feature selection algorithm, use additional methods if necessary
        selected_feats = set(x_data.columns)
        return selected_feats

    @staticmethod
    def choose_model(x_train, y_train):
        """
        Workflow
        * Add SelectKBest to pipeline select from the top X PCA components
        * Add SGDRegressor to pipeline as a model to fit the data
        """

        model = Pipeline(steps=[
            ('regress', LinearSVR(random_state=33642))
        ])

        model.set_params(regress__C=1.0, regress__loss='squared_epsilon_insensitive', regress__max_iter=1000)

        return model
