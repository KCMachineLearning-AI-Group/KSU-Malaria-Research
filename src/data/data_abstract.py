from abc import ABC, abstractmethod

"""
Abstract class for all custom data classes (ex. data_non_linear.py)
"""


class DataAbstract(ABC):

    @staticmethod
    @abstractmethod
    def clean_data(data):
        # Clean data
        # Split x, y
        # Remove/impute missing data (may need separate method if complex)
        # one-hot-encoding if applicable
        # Check for missing values
        # return x_data, y_data
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def engineer_features(x_data):
        """Engineer features"""
        # Perform feature engineering on float columns
        # Check for unexpected values
        # return x_data
        raise NotImplementedError

    @abstractmethod
    def test_train_split(self, x_data, y_data):
        """Return train and test set ready for model"""
        # Scale/normalize
        # return x_train, x_test, y_train
        raise NotImplementedError
