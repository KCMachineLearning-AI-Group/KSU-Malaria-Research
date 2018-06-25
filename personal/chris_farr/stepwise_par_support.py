from src.model_validation import ModelValidation
import numpy as np


def par_addition(feature_list, in_features, x_train, y_train, model, y_scaler):
    add_dict = {}
    for feat in feature_list:
        # Evaluate addition and log results
        test_features = list(in_features) + [feat]
        results = ModelValidation().score_regressor(x_train.loc[:, test_features], y_train, model, y_scaler,
                                                    pos_split=y_scaler.transform([[2.1]]), verbose=0)
        new_score = np.mean(results["root_mean_sq_error"])
        add_dict[feat] = new_score

    return add_dict
