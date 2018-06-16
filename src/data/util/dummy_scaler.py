from numpy import array

class DummyScaler:
    def __init__(self):
        return

    def fit(self, data):
        return self

    def transform(self, data):
        return array(data)

    def inverse_transform(self, data):
        return array(data)
