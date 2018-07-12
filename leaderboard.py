from src.models.model_template import ModelMyModel
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.models.model_linear_reg import ModelLinearReg
from src.models.model_sgd_regression import ModelSGDRegressor
from src.models.model_mixed_stepwise import ModelMixedStepwise
from src.models.model_linear_svr import ModelLinearSVR
from src.models.model_best_svr import ModelBestSVR
from src.model_validation import ModelValidation
from scipy import stats
import numpy as np
import math
import pandas as pd
np.set_printoptions(suppress=True)

"""
Standard leaderboard.py implementation
Used for comparing model selection, features engineering, feature selection and other techniques
with a standardized pipeline. Add implementations to the leaderboard_regressors list. Run from
command line with `python leaderboard.py` from the project root.
"""

# Initialize ModelValidation class
validation = ModelValidation()

leaderboard_regressors = [
    ModelMixedStepwise(),
    ModelBestSVR(),
    ModelMyModel(),
    ModelLinearReg(),
    ModelSGDRegressor(),
    ModelLinearSVR()
    # TODO add additional regression implementations
]
leaderboard_reg_scores = []


def score():
    compound_predictions = pd.DataFrame()
    for model_class in leaderboard_regressors:
        print("Running %s" % model_class.__class__.__name__ + "....\n")
        x_train, x_test, y_train, y_scaler, model = model_class.get_validation_support()
        print("\nScore Summary:")
        validation_result, error_model = validation.score_regressor(
            x_train, y_train, model, y_scaler,
            pos_split=y_scaler.transform([[2.1]])
        )
        leaderboard_reg_scores.append(validation_result)
        print("\nTest Prediction:")

        predictions = y_scaler.inverse_transform(model.fit(x_train, y_train).predict(x_test))

        for compound, pred in zip(x_test.index, predictions):
            print("{:<10}".format(compound), ": ", '{0:>5.2f}'.format(pred))
        print("\n")

        #calculating prediction intervals using the formula Zn-1 +- Tn-2 power 0.025* Sn-1 * sqroot ( n/(n-1)). This is calculated for every model.
        mean_value = predictions
        std_dev = y_scaler.inverse_transform(error_model.predict(x_test))
        n = len(x_train.index)
        t_stats_n_2 = stats.t.ppf(1-0.025, n-2)

        for compound, mu, sigma in zip(x_test.index, mean_value, std_dev):
            prediction_interval_low = mu - (sigma * t_stats_n_2 * math.sqrt(n/(n-1)))
            prediction_interval_high = mu + (sigma * t_stats_n_2 * math.sqrt(n/(n-1)))
            print("{:<10}".format(compound), ": ", '{0:>5.2f}'.format(prediction_interval_low), " - ", '{0:>5.2f}'.format(prediction_interval_high) )

        #calculating prediction interval using the predictions from all the trained models.

        if(compound_predictions.empty):
            column_name = 1
            d = {'compound' : x_test.index, column_name : predictions}
            compound_predictions = pd.DataFrame(d)
        else:
            compound_predictions[column_name] = predictions
        column_name += 1

    compound_predictions['prediction_interval_low'] = compound_predictions.min(axis=1)
    compound_predictions['prediction_interval_high'] = compound_predictions.max(axis=1)
    print(compound_predictions)


# Run from command line `python leaderboard.py` to view results
if __name__ == "__main__":
    print()
    score()
