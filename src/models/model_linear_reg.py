from sklearn.linear_model import LinearRegression

from src.models.model_abstract import ModelAbstract
from src.data.data_interactions import DataInteractions

"""
Template for model classes in the KSU project.
TODO's
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class ModelLinearReg(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataInteractions()
        self.selected_features = []

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
        selected_features = set(x_data.columns)
        return selected_feats

    @staticmethod
    def choose_model(x_train, y_train):
        """
        Example workflow
          * Tune model parameters
        Return model (doesn't have to be trained)
        :param x_train:
        :param y_train:
        :return: sklearn model
        """

        # TODO Implement custom model selection, use additional methods if necessary
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model

    def get_validation_support(self):
        """
        Output is used for leaderboard scoring
        :return: x_train, x_test, y_train, model
        """
        x_train, x_test, y_train, y_scaler = self.data_object.load_data()
        model = self.choose_model(x_train, y_train)
        return x_train, x_test, y_train, y_scaler, model
