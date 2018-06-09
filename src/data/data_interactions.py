from src.data.lib.interactions import InteractionChecker
from src.data.data_non_linear import DataNonLinear

class DataInteractions(DataAbstract):

    @staticmethod
    def clean_data(data):
        """
        Seperate into x and y data
        """
        data = data.loc[:, (data.std() > 0).values]
        # Split x, y
        y_data = data.pop("IC50")
        # y_class = pd.Series(data=[int(y < 2.1) for y in y_data])
        x_data = data.copy()
        # Fill missing data with 0
        x_data = x_data.fillna(0)
        return x_data, y_data

    @staticmethod
    def engineer_features(x_data, y_data):
        """
        Example implementation steps:
          * Perform feature engineering (use additional methods as needed, or static file)
          * Check for unexpected values
        :param x_data:
        :return: return x_data with new features
        """
        ic = InteractionChecker(alpha=0.01)
        if __name__ == '__main__':
            interactions = ic.fit(x_data, y_data, "IC50")
        return pd.concat([x_data,ic.get_interactions()],axis=1)

    @staticmethod
    def test_train_split(x_data, y_data):
        """
        Example implementation steps:
          * Scale/normalize
          * Split train/test based on missing target variables
        :param x_data:
        :param y_data:
        :return: x_train, x_test, y_train
        """
        x_train, x_test, y_train, y_scaler = DataNonLinear.test_train_split(x_data, y_data)
        return x_train, x_test, y_train, y_scaler
