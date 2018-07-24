from src.models.model_template import ModelMyModel
from src.models.model_correlation_grouper import ModelCorrelationGrouper
from src.models.model_linear_reg import ModelLinearReg
from src.models.model_sgd_regression import ModelSGDRegressor
from src.models.model_mixed_stepwise import ModelMixedStepwise
from src.models.model_mixed_stepwise_baseline import ModelMixedStepwiseBaseline
from src.models.model_linear_svr import ModelLinearSVR
from src.models.model_best_svr import ModelBestSVR
from src.models.model_best_svr_baseline import ModelBestSVRBaseline
from src.model_validation import ModelValidation
from src.analysis.statistics import confidence_interval
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
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
    ModelMixedStepwiseBaseline(),
    ModelBestSVR(),
    ModelBestSVRBaseline()
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
        print("\nScore Summary:")
        
        x_train, x_test, y_train, y_scaler, model = model_class.get_validation_support()
        validation_result = validation.score_regressor(
            x_train, y_train, model, y_scaler,
            pos_split=y_scaler.transform([[2.1]]),
            random_state=True #same output if looped
        )
        leaderboard_reg_scores.append(validation_result)
            
        print("\nTrain Prediction:")
        for compound in y_train.index:
            print("{:<10}".format(compound),
                  ": ",
                  "{0:>5.2f}".format(validation_result["cv_predict"][compound]),
                  " actual: ",
                  "{0:>5.2f}".format(y_scaler.inverse_transform([[y_train.loc[compound]]])[0][0]))
        print("\n")
        print("\nTest Prediction:")
        # print(y_scaler.inverse_transform(model.fit(x_train, y_train).predict(x_test)))
        predictions = y_scaler.inverse_transform(model.fit(x_train, y_train).predict(x_test))
        for compound, pred in zip(x_test.index, predictions):
            print("{:<10}".format(compound),
                  ": ",
                  "{0:>5.2f}".format(pred))
        print("\n")


def get_confidence():
    
    
    ci_df = pd.read_csv("confidence.csv")
    print(ci_df)
    
    
    """
    ci_df = pd.DataFrame(columns=["model","metric","lower", "upper"])
    index=0
    ci=defaultdict(list)
    for model_class in leaderboard_regressors:
        print ("Obtaining Confidence Intervals for %s" %model_class.__class__.__name__)
        for i in range(10):
            x_train, x_test, y_train, y_scaler, model = model_class.get_validation_support()
            validation_result = validation.score_regressor(
                x_train, y_train, model, y_scaler,
                pos_split=y_scaler.transform([[2.1]]),
                random_state=False, #random output if looped
                verbose=0
            )
            for metric in ['r2_score',
                                     'root_mean_sq_error',
                                     'explained_variance',
                                     'mean_sq_error',
                                     'mean_absolute_error',
                                     'median_absolute_error']:
                ci[model_class.__class__.__name__+"_"+metric].append(np.mean(validation_result[metric]))
            leaderboard_reg_scores.append(validation_result)
        
    for k,v in ci.items():
        ci[k]=confidence_interval(ci[k], confidence=0.95)
        model = k.split("_")[0]
        metric = '_'.join(k.split("_")[1:])
        lower = ci[k][0]
        upper = ci[k][1]
        ci_df.loc[index]=[model, metric, lower, upper]
        index+=1    
    
    
    
    
    
    ci_df.to_csv("confidence.csv", sep=",")
    print(ci_df)
    """
    # Now get CI for models with all features
    # Determine if p0 < pA
    
# Run from command line `python leaderboard.py` to view results
if __name__ == "__main__":
    print()
    #score()
    get_confidence()
