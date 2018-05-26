"""
Create an application that allows users to validate a model and input data with a universal validation process.
Accessible from a jupyter notebook or .py file.
Results are returned as a dictionary (verbose=0) or printed (verbose=1)
Optional benchmark comparison for best model? How is best determined with multiple metrics? Choose one?
* Best for each metric?
* Best overall?

Validation process:
* Leave-One-Out Stratified K-Fold

Warnings:
* add_train_data must not be derived from the x_data but must be from a separate Series

Assertions:
* x_data length must equal 47
* regression target must be float datatype
* classification target must be int datatype

Methods:
    * regression_validation
    Extend Stratified K-Fold to regression using binning at IC50<5
        Parameters:
            * x_data
            * y_data
            * model
            * add_train_data (to be included in training only, never testing)
            * verbose

    * classification_validation
        Parameters:
            * x_data
            * y_data
            * model
            * add_train_data (to be included in training only, never testing)
            * verbose

Metrics:
    ▪ classification
        • AUC
        • F-Beta: mixing 50/50
        • Log loss
        • Confusion matrix dictionary
    ▪ regression
        • RMSE
        • R2 score

Tests:
    * Assert performance of a regression and classification model/dataset

TODO Create an example notebook and set the initial benchmark. Place this in the readme along with instructions.
todo use ^^ as the unit test that anyone can use, store in my folder to show how to organize
Test in python 2 and 3

"""
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from numpy import sqrt, mean, unique
from pandas import DataFrame, Series
import warnings
warnings.filterwarnings('once')


class ModelValidation:
    RANDOM_STATE = 36851234
    REPEATS = 10

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
        assert y_data.dtypes in ["int", "int64", "int32"], "Expected y_data to be int dtype and received {}".format(y_data.dtypes)

        if add_train_data is not None:
            raise NotImplementedError

        # create logging dictionary to track scores
        scoring_dict = {"log_loss": [], "roc_auc_score": [], "confusion_matrix": []}
        # num_splits count number of positive examples, todo specify in output
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

    def score_regressor(self, x_data, y_data, model, add_train_data=None, verbose=1):
        """
        Model validation for producing comparable model evaluation. Uses Stratified K-Fold LOOCV adapted
        for regression with the positive equivalent <10 IC50, producing 5 folds.
        :param x_data: Pandas DataFrame object, Series 3 with 47 examples
        :param y_data: Pandas DataFrame or Series object, float datatype, target variables for Series 3
        :param model: must have fit and predict method, use sklearn or wrapper
        :param add_train_data: Additional data to be evenly spread across train splits
        :param verbose: If 0, return dictionary only, if 1 printed results
        :return: dictionary
        """
        raise NotImplementedError

# def test_classification_validation():
#     classification_validation()
# def test_regression_validation():
#     regression_validation()
