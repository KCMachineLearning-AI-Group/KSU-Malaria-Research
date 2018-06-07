import pandas as pd
import numpy as np
from model_validation import ModelValidation
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

validate = ModelValidation()
# Load data
df = pd.read_csv("data/Series3_6.15.17_padel.csv", index_col=0)
# Eliminate features without variance
df = df.loc[:, (df.std() > 0).values]
# Seperate Series 3 test when IC50 is null
test_index = df.IC50.isnull()
test_df = df.loc[test_index]
df = df.loc[~test_index]
# Remove columns with missing data
df = df.dropna(axis=1)
# Transform discrete with one-hot-encoding
int_cols = df.columns[df.dtypes == 'int64']
float_cols = df.columns[df.dtypes == 'float64']
one_hot_df = pd.get_dummies(df[int_cols].astype('O'))
df = pd.merge(df[float_cols], one_hot_df, left_index=True, right_index=True)
# Split x, y
y_data = df.pop("IC50")
y_class = pd.Series([int(y<=2) for y in y_data], index=y_data.index)
x_data = df.copy()
# Perform feature engineering on float columns
for feat in x_data.columns[x_data.dtypes == 'float64']:
    feature_df = x_data.loc[:, feat]
    if feature_df.min() > 0:  # Avoid 0 or negative
        x_data.loc[:, feat + "_log"] = feature_df.apply(np.log)  # log
        x_data.loc[:, feat + "_log2"] = feature_df.apply(np.log2)  # log2
        x_data.loc[:, feat + "_log10"] = feature_df.apply(np.log10)  # log10
        x_data.loc[:, feat + "_cubert"] = feature_df.apply(
            lambda x: np.power(x, 1 / 3))  # cube root
        x_data.loc[:, feat + "_sqrt"] = feature_df.apply(np.sqrt)  # square root
    # Avoid extremely large values, keep around 1M max
    if feature_df.max() < 13:
        x_data.loc[:, feat + "_exp"] = feature_df.apply(np.exp)  # exp
    if feature_df.max() < 20:
        x_data.loc[:, feat + "_exp2"] = feature_df.apply(np.exp2)  # exp2
    if feature_df.max() < 100:
        x_data.loc[:, feat + "_cube"] = feature_df.apply(
            lambda x: np.power(x, 3))  # cube
    if feature_df.max() < 1000:
        x_data.loc[:, feat + "_sq"] = feature_df.apply(np.square)  # square
# Ensure no (+/-) inf or nan due to improper transformation
x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
assert not sum(x_data.isna().sum()), "Unexpected nulls found"

# For regression, gridsearch optimal params
# params =
# grid = GridSearchCV()

# Remove highly correlated features, group by correlation and select 1 from each group
corr_threshold = .99
for corr_threshold in np.arange(1, .9, -.01):
    corr_matrix = x_data.corr()
    corr_matrix.loc[:, :] = np.tril(corr_matrix, k=-1)  # borrowed from Karl D's answer

    already_in = set()
    corr_result = []
    for col in corr_matrix:
        correlated = corr_matrix[col][np.abs(corr_matrix[col]) > corr_threshold].index.tolist()
        if correlated and col not in already_in:
            already_in.update(set(correlated))
            correlated.append(col)
            corr_result.append(correlated)
        elif col not in already_in:
            already_in.update(set([col]))
            corr_result.append([col])

    print("There are %s correlated groups at > %s correlation threshold" % (len(corr_result), corr_threshold))

    non_correlated_feats = [corr_feats[0] for corr_feats in corr_result]






# Resample the dataset with combined approach
# http://contrib.scikit-learn.org/imbalanced-learn/stable/combine.html#combine
# Try many models on resulting pipeline
model = AdaBoostClassifier(n_estimators=5, learning_rate=0.075, random_state=0)
# Validate model and data
validate = ModelValidation()
score_dict = validate.score_classifier(x_data, y_class, model)

# benchmark: .529 roc_auc_score
