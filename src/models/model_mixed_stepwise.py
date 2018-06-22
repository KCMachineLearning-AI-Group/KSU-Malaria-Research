from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
from src.models.model_abstract import ModelAbstract
# from src.data.data_simple import DataSimple
# from src.data.data_non_linear import DataNonLinear
from src.data.data_step_interactions import DataStepInteractions

"""
Template for model classes in the KSU project.
TODO's
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class ModelMixedStepwise(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataStepInteractions()
        # self.data_object = DataSimple()
        # self.data_object = DataNonLinear()
        self.selected_features = []

    @staticmethod
    def select_features(x_data):
        """
        Reading features from stepwise feature selection performed outside of pipeline.
        :param x_data: full datasett
        :return:
        """
        # feat_df = pd.read_csv("src/models/support/mixed_stepwise_features.csv")
        # feat_df = pd.read_csv("src/models/support/mixed_stepwise_features_non_linear.csv")
        feat_df = pd.read_csv("src/models/support/mixed_stepwise_features_interactions.csv")
        selected_feats = list(np.squeeze(feat_df.values))
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
