
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(sys.path[0]+"/src/utils/validator/"))
from model_validation import ModelValidation

class AbstractPredictiveModel:
    #properties
    _model=None
    _grid=None
    _pipe=None
    _params=None
    _modelType=None
    _validator=None
    _validation_results=None
    _X=None
    _y=None
    _code=None
    _n_jobs=None
    _verbose=None
    
    #constructor
    def __init__(self, X, params, nfolds, n_jobs, verbose):
        if self._verbose:
            print ("Constructed AbstractPredictiveModel: "+self._code)
        assert isinstance(params, dict), "params must be dict"
        self._X = X
        self._params = params
        self._nfolds=nfolds
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._validator = ModelValidation()
    #methods
    def validate(self):
        pass
    
    def getCode(self):
        return self._code
    
    def getValidationResults(self):
        return self._validation_results
    
    def getBestEstimator(self):
        if self._verbose:
            print("Returning "+self._code+" best estiminator")
        return self._model