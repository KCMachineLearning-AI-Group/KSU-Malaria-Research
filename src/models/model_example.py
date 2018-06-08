import numpy as np
from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear
# TODO start here!! Finish the get_data method, decide if it should be static or not
# TODO finish implementing ModelCorrelationGrouper and abstract, then move on to leaderboard.py


class ModelCorrelationGrouper(ModelAbstract):

    def __init__(self):
        self.data_object = DataNonLinear()

    # get data
    def get_data(self):
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        # Add non-linear transformations to x_data
        x_data = self.data_object.engineer_features(x_data)
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)

    def select_features(self, x_data) -> set:
        # Group features by correlation
        corr_threshold = .95
        corr_matrix = x_data.corr()
        corr_matrix.loc[:, :] = np.tril(corr_matrix, k=-1)  # borrowed from Karl D's answer

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


    # choose model
    # select features
    # predict test

    pass


