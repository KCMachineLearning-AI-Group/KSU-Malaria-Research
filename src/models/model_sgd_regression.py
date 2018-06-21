import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear
from src.data.data_dimension_reduction import DataDimensionReduction
from src.data.util.variance_score import VarianceScorer


class ModelSGDRegressor(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataDimensionReduction()

    def get_validation_support(self):
        """
        Output is used for leaderboard scoring
        :return: x_train, x_test, y_train, model
        """
        x_data, y_data = DataNonLinear.clean_data(self.data_object.data)
        x_data = DataNonLinear.engineer_features(x_data)
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
        :param x_data: full datasett
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
            ('select', SelectKBest(VarianceScorer.score, k=5)),
            ('regress', SGDRegressor(random_state=0))
        ])

        model.set_params(select__k=5, regress__loss='huber', regress__penalty='l1', regress__alpha=0.15, regress__l1_ratio=0.30, regress__max_iter=10)

        return model
