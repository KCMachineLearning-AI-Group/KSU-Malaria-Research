from abc import ABC, abstractmethod
# TODO finish adding documentation below, then fill out templates


class ModelAbstract(ABC):

    def __init__(self):
        self.data_object = None  # TODO Override this with your data class
        self.selected_features = []  # TODO set this dynamically with feature_selection to allow for analysis

    """ 
    Non-Abstract, Inherited Methods: These should be the same for everyone
    """

    def get_data(self):
        """
        Steps:
          * Clean data (data_object.clean_data)
          * Engineer features (data_object.engineer_features)
          * Train/test split (normalize)
        :return: x_train, x_test, y_train
        """
        x_data, y_data = self.data_object.clean_data(self.data_object.data)
        x_data = self.data_object.engineer_features(x_data)
        x_train, x_test, y_train = self.data_object.test_train_split(x_data, y_data)
        return x_train, x_test, y_train

    def get_test_prediction(self):
        x_train, x_test, y_train, model = self.get_validation_support()
        model.fit(x_train, y_train)
        return model.predict(x_test)

    """ Abstract Methods """
    @abstractmethod
    def get_validation_support(self):
        """
        Use output for leaderboard scoring
        :return: x_train, x_test, y_train, model
        """
        x_train, x_test, y_train, model = None, None, None, None
        return x_train, x_test, y_train, model

    @staticmethod
    @abstractmethod
    def select_features(x_data):
        """
        Select features
        :param x_data:
        :return: return list or set of features by name
        """
        return

    @staticmethod
    @abstractmethod
    def choose_model(x_train, y_train):
        """
        Tune model parameters
        Return model
        :param x_train:
        :param y_train:
        :return: sklearn model or wrapper
        """
        return
