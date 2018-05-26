from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from numpy import sqrt, mean
from pandas import DataFrame, Series


class ModelValidation:

    RANDOM_STATE = 36851234
    REPEATS = 10

    def score_regressor(self, x_data, y_data, model, add_train_data=None, verbose=1, pos_split=10):
        """
        Model validation for producing comparable model evaluation. Uses Stratified K-Fold LOOCV adapted
        for regression with the positive equivalent <10 IC50, producing 5 folds.
        :param x_data: Pandas DataFrame object, Series 3 with 47 examples
        :param y_data: Pandas DataFrame or Series object, float datatype, target variables for Series 3
        :param model: must have fit and predict method, use sklearn or wrapper
        :param add_train_data: Additional data to be evenly spread across train splits
        :param verbose: If 0, return dictionary only, if 1 printed results
        :param pos_split: cutoff for positive class in StratifiedKFold (y<pos_split)
        :return: dictionary
        """
        assert isinstance(x_data, DataFrame), "x_data must be a pandas DataFrame"
        assert isinstance(y_data, DataFrame) or isinstance(y_data, Series), "y_data must be pandas DataFrame or Series"
        assert len(x_data) == 47, "Expected 47 training examples and received {}".format(len(x_data))
        assert y_data.dtypes == "float", "Expected y_data to be float dtype and received {}".format(y_data.dtypes)

        if add_train_data is not None:
            raise NotImplementedError

        # create logging dictionary to track scores
        scoring_dict = {"r2_score": [], "rmse": []}
        # create y_class series for Stratified K-Fold split at pos_split
        y_class = Series(data=[int(y < pos_split) for y in y_data])
        # num_splits count number of positive examples
        num_splits = sum(y_class.values)
        scoring_dict["num_splits"] = num_splits
        # create splits using stratified kfold
        rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=self.REPEATS, random_state=self.RANDOM_STATE)
        # loop through splits
        for train, test in rskf.split(x_data, y_class):
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test, :]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            assert sum(y_test) > 0, "no positive examples in split"
            # train model, test model with all scoring parameters
            model.fit(x_train, y_train)
            y_ = model.predict(x_test)
            # append scores to logging dictionary
            scoring_dict["r2_score"].append(r2_score(y_test, y_))
            scoring_dict["rmse"].append(sqrt(mean_squared_error(y_test, y_)))
        if verbose == 1:
            # Print contents of dictionary except confusion matrix
            print("with {} splits and {} repeats".format(num_splits, self.REPEATS))
            for metric in scoring_dict:
                if metric == "num_splits":
                    continue
                else:
                    print("average {}: {}".format(metric, mean(scoring_dict[metric])))
        return scoring_dict

    def score_classifier(self, x_data, y_data, model, add_train_data=None, verbose=1):
        """
        Model validation method for producing comparable model evaluation. Uses Stratified K-Fold LOOCV where
        K equals the number of positive class examples.
        :param x_data: Pandas DataFrame object, Series 3 with 47 examples
        :param y_data: Pandas DataFrame or Series object, int datatype, target variables for Series 3
        :param model: must have fit and predict method, use sklearn or wrapper
        :param add_train_data: Additional data to be evenly spread across train splits
        :param verbose: If 0, return dictionary only, if 1 printed results
        :return: dictionary
        """
        assert isinstance(x_data, DataFrame), "x_data must be a pandas DataFrame"
        assert isinstance(y_data, DataFrame) or isinstance(y_data, Series), "y_data must be pandas DataFrame or Series"
        assert len(x_data) == 47, "Expected 47 training examples and received {}".format(len(x_data))
        int_types = ["int", "int64", "int32"]
        assert y_data.dtypes in int_types, "Expected y_data to be int dtype and received {}".format(y_data.dtypes)

        if add_train_data is not None:
            raise NotImplementedError

        # create logging dictionary to track scores
        scoring_dict = {"log_loss": [], "roc_auc_score": [], "confusion_matrix": []}
        # num_splits count number of positive examples
        num_splits = sum(y_data.values)
        scoring_dict["num_splits"] = num_splits
        # create splits using stratified kfold
        rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=self.REPEATS, random_state=self.RANDOM_STATE)
        # loop through splits
        for train, test in rskf.split(x_data, y_data):
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test, :]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            assert sum(y_test) > 0, "no positive examples in split"
            # train model, test model with all scoring parameters
            model.fit(x_train, y_train)
            y_ = model.predict(x_test)
            # append scores to logging dictionary
            scoring_dict["log_loss"].append(log_loss(y_test, y_))
            scoring_dict["roc_auc_score"].append(roc_auc_score(y_test, y_))
            scoring_dict["confusion_matrix"].append(confusion_matrix(y_test, y_, labels=[1, 0]))
        if verbose == 1:
            # Print contents of dictionary except confusion matrix
            print("with {} splits and {} repeats".format(num_splits, self.REPEATS))
            for metric in scoring_dict:
                if metric == "num_splits":
                    continue
                elif metric == "confusion_matrix":
                    print("average confusion_matrix")
                    print(mean(scoring_dict["confusion_matrix"], axis=0))
                else:
                    print("average {}: {}".format(metric, mean(scoring_dict[metric])))
        return scoring_dict
