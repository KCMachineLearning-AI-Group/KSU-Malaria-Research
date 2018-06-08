# Run models and print results....
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.model_validation import ModelValidation

# Initialize ModelValidation class
validation = ModelValidation()

# TODO Add regression implementations to this list similar to ModelCorrelationGrouper
leaderboard_regressors = [ModelCorrelationGrouper(), ]
leaderboard_reg_scores = []


def score():
    for model_class in leaderboard_regressors:
        print("Scoring %s" % model_class.__class__.__name__)
        x_train, x_test, y_train, model = model_class.get_validation_support()
        validation_result = validation.score_regressor(
            x_train, y_train, model,
            pos_split=model_class.data_object.y_scaler.transform([[2.1]])
        )
        leaderboard_reg_scores.append(validation_result)
        print("\n")


# TODO run from command line `python leaderboard.py` to view results
if __name__ == "__main__":
    score()
