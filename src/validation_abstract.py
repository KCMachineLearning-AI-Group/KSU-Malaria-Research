from abc import ABC, abstractmethod


class ValidationAbstract(ABC):

    @abstractmethod
    def score_regressor(self, x_data, y_data, model, verbose):
        # TODO Optional: Add or remove parameters as needed to match project specific implementation
        raise NotImplementedError

    @abstractmethod
    def score_classifier(self, x_data, y_data, model, verbose):
        # TODO Optional: Add or remove parameters as needed to match project specific implementation
        raise NotImplementedError
