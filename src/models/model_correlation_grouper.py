import numpy as np
from sklearn.svm import LinearSVR
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear


class ModelCorrelationGrouper(ModelAbstract):

    def __init__(self):
        ModelAbstract.__init__(self)
        self.data_object = DataNonLinear()
        self.selected_features = []

    def get_validation_support(self) -> tuple:
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        self.selected_features = self.select_features(x_data)
        x_data = x_data.loc[:, self.selected_features].copy()
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)
        model = self.choose_model(x_train, y_train)
        return x_train, x_test, y_train, model

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
        model = LinearSVR(random_state=0, C=.05)
        return model
