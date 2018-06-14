from src.models.model_template import ModelMyModel
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.models.model_support_vector_regressor import ModelSupportVectorRegressor
from src.model_validation import ModelValidation

"""
Standard leaderboard.py implementation
Used for comparing model selection, features engineering, feature selection and other techniques
with a standardized pipeline. Add implementations to the leaderboard_regressors list. Run from
command line with `python leaderboard.py` from the project root.
"""

# Initialize ModelValidation class
validation = ModelValidation()

leaderboard_regressors = [
    #ModelMyModel(),
    #ModelCorrelationGrouper(),
    ModelSupportVectorRegressor()
    # TODO add additional regression implementations
]
leaderboard_reg_scores = []


def score():
    for model_class in leaderboard_regressors:
        print("Running %s" % model_class.__class__.__name__ + "....")
        x_train, x_test, y_train, y_scaler, model = model_class.get_validation_support()
        validation_result = validation.score_regressor(
            x_train, y_train, model,
            pos_split=y_scaler.transform([[2.1]])
        )
        leaderboard_reg_scores.append(validation_result)
        print()
        print("Test Prediction:")
        print(model_class.get_test_prediction())
        print("\n")


# Run from command line `python leaderboard.py` to view results
if __name__ == "__main__":
    print()
    score()
