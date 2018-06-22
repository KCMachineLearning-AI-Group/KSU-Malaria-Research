

add_dict = {}
for feat, corr_group in test_feats_for_addition.items():
    # Evaluate addition and log results
    test_features = list(in_features) + [feat]
    results = validation.score_regressor(x_train.loc[:, test_features], y_train, model, y_scaler,
                                         pos_split=y_scaler.transform([[2.1]]), verbose=0)
    new_score = np.mean(results["root_mean_sq_error"])
    add_dict[feat] = new_score
