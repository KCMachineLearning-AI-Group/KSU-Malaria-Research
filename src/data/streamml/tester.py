
import pandas as pd
import numpy as np
from src.streamline.transformation.flow.TransformationStream import TransformationStream
from src.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))

Xnew = TransformationStream(X).flow(["boxcox","scale","normalize","pca"], params={"pca__percent_variance":0.75})


#options: lr, ridge, lasso, enet, svr, knnr, abr, rfr
#scoring option not working right, be okay with default scorers.
performances = ModelSelectionStream(Xnew,y).flow(["svr", "lr", "knnr","lasso","abr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                     'lr__fit_intercept':[False, True],
                                                     'knnr__n_neighbors':[5,10],
                                                     'lasso__alpha':[1.0,10.0,20.0],
                                                     'abr__n_estimators':[10,20,50],
                                                     'abr__learning_rate':[0.1,1,10]},
                                              verbose=True)