# -----------------------------------------------------------
# Model : LinearSVR ( Linear Support Vector Regression )
# Data : Train/Test - 0.55/0.45 + 5 CV
# Loss Function : ‘L2’ squared epsilon-insensitive loss.
# Scorer : R2
# -----------------------------------------------------------

# Library used
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Import Cleaned Data - from Repo
X = np.genfromtxt("Data/example_x.csv", delimiter=",", skip_header=1)
target = np.genfromtxt("Data/example_y.csv", delimiter=",")

# Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.45, random_state=0)

# Scale Test and Train Data - Standard Scale
sc_X = StandardScaler()
sc_Y = StandardScaler()
scT_X = StandardScaler()
scT_Y = StandardScaler()

# Fit the transformed Train and Test Data into Scaler
scale_x = sc_X.fit_transform(X_train)
scale_y = sc_Y.fit_transform(y_train)

scaleT_x = scT_X.fit_transform(X_test)
scaleT_y = scT_Y.fit_transform(y_test)

# R2 Score
rtwo_scorer = make_scorer(r2_score, greater_is_better=True)

# Linear SVR Model
LinearSVR_Model = LinearSVR(loss='squared_epsilon_insensitive')

# NP Array of possible 'C' values
possibleC = np.arange(0.001, 1.0, 0.01)

# GridSearchCV for the best parameters
grid = GridSearchCV(LinearSVR_Model, param_grid={'C': possibleC}, scoring=rtwo_scorer, cv=5)

# Fit GridSearchCV model
grid.fit(scale_x, scale_y)

# Fit LinearSVR Model
LinearSVR_Model.fit(scale_x, scale_y)

# Output parameters, scores
print(grid.best_params_)
print("R2 Score for LinearSVR : {}".format(grid.best_score_))
print("Grid Score for LinearSVR : {}".format(round(grid.score(scaleT_x, scaleT_y), 4)))
print("Model Score for LinearSVR : {}".format(LinearSVR_Model.score(scaleT_x, scaleT_y)))

# Model Score without tuning : 0.5039
# TODO : Create a loop to tune the LinearSVR Model - Remove 5% of the worst features in every iteration
# TODO : Test out other parameters and see if they can improve the model
# TODO : Other ways to improve the accuracy -  Combine multiple models
