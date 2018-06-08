from src.models.model_abstract import ModelAbstract
from src.data.data_non_linear import DataNonLinear


class ModelCorrelClean(ModelAbstract):

    def __init__(self):
        self.data_object = DataNonLinear()

    # get data
    def get_data(self):
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        # Add non-linear transformations to x_data
        x_data = self.data_object.engineer_features(x_data)
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)
        # TODO start here!! Finish the get_data method, decide if it should be static or not
        # TODO finish implementing ModelCorrelClean and abstract, then move on to leaderboard.py

    # choose model
    # select features
    # predict test

    pass


