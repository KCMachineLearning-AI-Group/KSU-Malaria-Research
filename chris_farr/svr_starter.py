import pandas as pd
import numpy as np
from model_validation import ModelValidation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.cluster import KMeans

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
x_data = df.copy()

# Ensure no (+/-) inf or nan due to improper transformation
x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
assert not sum(x_data.isna().sum()), "Unexpected nulls found"

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

# Normalize
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_data.loc[:, :] = x_scaler.fit_transform(x_data)
y_data.loc[:] = np.squeeze(y_scaler.fit_transform(y_data.values.reshape(-1, 1)))

# Remove highly correlated features, group by correlation and select 1 from each group
# best_corr_thresh = .91
# best_r2_score = .18796

# best_cos_thresh = .91
# best_r2_score = .19379

# https://reader.elsevier.com/reader/sd/20C9D54036F9EB702CD7F163E0D03823B6FF3F3A4F2E0EA1BB6E3C8B6895A65F26CDFC723FAF98BC736BF5C7B429580C
# kmeans cluster features to determine similarity


kmeans = KMeans(n_clusters=int(x_data.shape[1] * .9), n_jobs=7)
result = kmeans.fit_transform(x_data.transpose())
from keras.utils import np_utils

np_utils.to_categorical(np.argmax(result, axis=1))
# TODO start here,!!!!!! reducing based on k-means
# perform kmeans, then count how many in a group
# rank removal with equal coef by number in similar group, then recalculate number in group


# Remove any features that have >.99 cosine similarity
# Backward stepwise, count similarity groups by feature, sort by number in group and then
# If only one feature in each group then lower cosine similarity threshold by .01
# Iteratively remove only one feature at a time, update the feature lists, then remove the next
cos_groups = dict([(feat, feat_list.copy()) for feat_list in cos_result for feat in feat_list])

print(cos_groups.keys())

len(cos_groups)

# Sort by number in group

# If removing, for every feature in group, remove that feature from the group
feat_for_removal = list(cos_groups.keys())[0]
# for feat in cos_groups[feat_for_removal]:

cos_groups[feat_for_removal].remove(feat_for_removal)

# Calculate length of list all features in group
for feat in cos_groups[feat_for_removal]:
    print(len(cos_groups[feat]))


cos_groups.pop(feat_for_removal)

# Benchmark for Lasso regression
model = LinearSVR(random_state=0)
params = {"C": np.arange(.1, 1.1, .1)}
# rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)  # Classification only
grid = GridSearchCV(model, param_grid=params, scoring=make_scorer(r2_score, greater_is_better=True), cv=10, n_jobs=7)

grid.fit(x_data.loc[:, selected_feats], y_data)

validate = ModelValidation()
results = validate.score_regressor(x_data.loc[:, selected_feats], y_data,
                                   grid.best_estimator_, pos_split=y_scaler.transform([[2.1]]))
round_r2_score = np.mean(results["r2_score"])


# Best R2 Score: .1879, corr_threshold = .91

# Using backward stepwise selection, remove a max of 5% each iteration (tune)
# Favor the features that are part of larger correlated group when tie

# Rank features by coef_
[feat for coef, feat in sorted(zip(abs(grid.best_estimator_.coef_), selected_feats))]
# TODO Start here!!!
# TODO determine best way to select features for removal. If the starting features are less correlated then
# todo it'll be easier to keep the model stable when removing features. Try keeping it simple and tune the threshold
# So just remove the first 1-5% or so...


# Starting with groups of perfect correlations, select one from each group to represent that dimension
# Create loop that selects features for removal by number of correlated features in a group with a decreasing
# threshold and also by coefficient (primary sort)

# Build a set for removal




