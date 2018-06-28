from src.models.model_template import ModelMyModel
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.models.model_linear_reg import ModelLinearReg
from src.models.model_sgd_regression import ModelSGDRegressor
from src.models.model_mixed_stepwise import ModelMixedStepwise
from src.models.model_linear_svr import ModelLinearSVR
from src.models.model_best_svr import ModelBestSVR
from src.model_validation import ModelValidation
import numpy as np
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
    # ModelMyModel(),
    # ModelLinearReg(),
    # ModelSGDRegressor(),
    # ModelLinearSVR()
    # TODO add additional regression implementations
]
leaderboard_reg_scores = []


def score():
    for model_class in leaderboard_regressors:
        print("Running %s" % model_class.__class__.__name__ + "....\n")
        x_train, x_test, y_train, y_scaler, model = model_class.get_validation_support()
        print("\nScore Summary:")
        validation_result = validation.score_regressor(
            x_train, y_train, model, y_scaler,
            pos_split=y_scaler.transform([[2.1]])
        )
        leaderboard_reg_scores.append(validation_result)
        print("\nTest Prediction:")
        # print(y_scaler.inverse_transform(model.fit(x_train, y_train).predict(x_test)))
        predictions = y_scaler.inverse_transform(model.fit(x_train, y_train).predict(x_test))
        for compound, pred in zip(x_test.index, predictions):
            print("{:<10}".format(compound), ": ", '{0:>5.2f}'.format(pred))
        print("\n")


# Run from command line `python leaderboard.py` to view results
if __name__ == "__main__":
    print()
    score()
