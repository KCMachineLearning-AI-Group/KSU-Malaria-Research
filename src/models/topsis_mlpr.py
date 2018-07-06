from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
from src.models.model_abstract import ModelAbstract
# from src.data.data_simple import DataSimple
# from src.data.data_non_linear import DataNonLinear
from src.data.data_step_interactions import DataStepInteractions
from src.data.data_topsis import DataTOPSIS
import sys
from streamml.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

"""
Template for model classes in the KSU project.
TODO's
  * Copy into new file
  * Rename class
  * Implement TODO's below
"""


class ModelTOPSISMLPR(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataTOPSIS()
        self.selected_features = []

    @staticmethod
    def select_features( x_data):
        """
        Reading features from stepwise feature selection performed outside of pipeline.
        :param x_data: full datasett
        :return:
        """
        # feat_df = pd.read_csv("src/models/support/mixed_stepwise_features.csv")
        # feat_df = pd.read_csv("src/models/support/mixed_stepwise_features_non_linear.csv")
        
        
        return x_data.columns.tolist()

    @staticmethod
    def choose_model( x_train, y_train):
        """
        Example workflow
          * Tune model parameters
        Return model (doesn't have to be trained)
        :param x_train:
        :param y_train:
        :return: sklearn model
        """

        # TODO Implement custom model selection, use additional methods if necessary
        model = ModelSelectionStream(x_train,y_train).flow(["mlpr", "abr", "enet"], params={'mlpr__hidden_layer_sizes':[(int(x_train.shape[1]), int(x_train.shape[1]/2), int(x_train.shape[1]/4)),
                                                                                                        (100,10,2),
                                                                                                        (1000,100,10,1)],
                                                                                    'abr__n_estimators':[10,50,100],
                                                                                    'abr__learning_rate':[0.001,0.01,0.1,1],
                                                                                    'abr__loss':['linear','square','exponential'],
                                                                                    'enet__alpha':[0.25,0.5,0.75,1.0],
                                                                                    'enet__l1_ratio':[0.25,0.5,0.75,1.0]},
                                                                    regressors=True,
                                                                    cut=2)
        
        return model['abr']
