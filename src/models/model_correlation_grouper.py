import numpy as np
from sklearn.svm import LinearSVR
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear


class ModelCorrelationGrouper(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataNonLinear()
        self.selected_features = []

    @staticmethod
    def select_features(x_data):
        """
        Group features that are highly correlated based on a threshold correlation level.
        Select the (arbitrary) first feature from each group
        :param x_data:
        :return: set of selected feature names
        """
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
        """
        * Linear Support Vector Regressor
        * C=.05 was most frequently optimal in tuning with CV10 (see ______.ipynb)
        :param x_train:
        :param y_train:
        :return: sklearn LinearSVR model, training is not necessary as it is overriden
        """
        model = LinearSVR(random_state=0, C=.05)
        return model
