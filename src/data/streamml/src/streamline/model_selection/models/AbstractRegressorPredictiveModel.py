import sys
import os
sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/"))
print(sys.path)
from AbstractPredictiveModel import AbstractPredictiveModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
class AbstractRegressorPredictiveModel(AbstractPredictiveModel):

    #constructor
    _options = ['explained_variance',
              'neg_mean_absolute_error',
              'neg_mean_squared_error',
              'neg_mean_squared_log_error',
              'neg_median_absolute_error',
              'r2']

    def __init__(self, modelType, X, y, params, nfolds, n_jobs, scoring, verbose):
        
        if self._verbose:
            print("Constructed AbstractRegressorPredictiveModel: "+self._code)
        assert modelType == "regressor", "You are creating a regressor, but have no specified it to be one."
        #assert any([isinstance(y.dtypes[0],float),isinstance(y.dtypes[0],float)]), "Your response variable y is not a float."
        self._modelType = modelType
        self._y=y
        self._scoring=scoring
        AbstractPredictiveModel.__init__(self, X, params, nfolds, n_jobs, verbose)
        
    #methods
    def validate(self, Xtest, ytest, verbose=False):
        
        scoring_dict = {"r2": [], "rmse": []}
        ypred = self._model.predict(Xtest)
        # append scores to logging dictionary
        scoring_dict["r2"].append(r2_score(ytest, ypred))
        scoring_dict["rmse"].append(np.sqrt(mean_squared_error(ytest, ypred)))
        self._validation_results=scoring_dict
        return self._validation_results
    
    def constructRegressor(self, model):
        self._pipe          = Pipeline([(self._code, model)])

        self._grid          = GridSearchCV(self._pipe,
                                           param_grid=self._params, 
                                           n_jobs=self._n_jobs, 
                                           cv=self._nfolds, 
                                           verbose=False)
        # fix with multiple scoring measures to train on
        #best_fit                 = self._grid.fit(self._X,self._y, scoring=self._scoring).best_estimator_.named_steps[self._code]
        best_fit                 = self._grid.fit(self._X,self._y).best_estimator_.named_steps[self._code]
        return best_fit    