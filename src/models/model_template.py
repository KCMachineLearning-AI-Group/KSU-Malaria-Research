from sklearn.svm import LinearSVR

from src.models.model_abstract import ModelAbstract
from src.data.data_template import DataMyData

"""
Template for model classes in the KSU project.
TODO's
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class ModelMyModel(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataMyData()  # TODO use your own data class or steal
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
        selected_feats = set(x_data.columns)
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
        model = LinearSVR(random_state=0)
        return model
