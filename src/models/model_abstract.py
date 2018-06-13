from abc import ABC, abstractmethod

"""
Standardized ModelAbstract for KSU project.
Changes to this class can take place after group discussion to discover dependency
impacts, otherwise override methods in the model classes as needed.
"""


class ModelAbstract(ABC):

    def __init__(self):
        self.data_object = None  # Override this with your data class
        self.selected_features = []  # Set this dynamically with feature_selection to allow for analysis
        self.y_scaler = None  # Set in get_data function, used for inverse transform of test prediction
    """
    Non-Abstract, Inherited Methods: Override if necessary
    """

    def get_validation_support(self):
        """
        Output is used for leaderboard scoring
        :return: x_train, x_test, y_train, model
        """
        x_train, x_test, y_train, y_scaler = self.data_object.load_data()
        self.selected_features = self.select_features(x_train)
        x_train = x_train.loc[:, self.selected_features].copy()
        model = self.choose_model(x_train, y_train)
        return x_train, x_test, y_train, y_scaler, model

    def get_test_prediction(self):
        """
        Override in your class if necessary
        :return: prediction result
        """
        x_train, x_test, y_train, y_scaler, model = self.get_validation_support()
        self.selected_features = self.select_features(x_train)
        x_train = x_train.loc[:, self.selected_features].copy()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        return y_scaler.inverse_transform(prediction)

    """
    Abstract Methods, these must be overriden
    """

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
