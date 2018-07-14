from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
from src.models.model_abstract import ModelAbstract
from src.data.data_baseline import DataBaseline

"""
Template for model classes in the KSU project.
TODO's
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class ModelBestSVRBaseline(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataBaseline()  # TODO use your own data class or steal
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
        #feat_df = pd.read_csv("src/models/support/best_features.csv")
        #selected_feats = list(np.squeeze(feat_df.values))
        return x_data.columns.tolist()

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
