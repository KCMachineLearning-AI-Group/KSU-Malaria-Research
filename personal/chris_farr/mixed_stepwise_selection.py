import numpy as np
from sklearn.svm import LinearSVR
import pandas as pd
import random
from src.data.data_simple import DataSimple
from src.model_validation import ModelValidation
import time
import math
from personal.chris_farr.stepwise_par_support import par_addition, par_removal
from joblib import Parallel, delayed
from multiprocessing import cpu_count


class MixedStepSelect:

    def __init__(self, random_start=True, multiplier=50, starting_batch_size=100, data_class=DataSimple(),
                 corr_threshold=.99, n_start_feats=100):
        self.multiplier = multiplier  # TODO Rename this or make more intuitive
        self.n_jobs = cpu_count() - 1 - 3
        self.starting_batch_size = starting_batch_size
        self.no_improvement_count = 0
        self.last_benchmark = np.inf
        self.benchmark = np.inf
        self.validation = ModelValidation()
        self.data_class = data_class
        self.corr_threshold = corr_threshold
        self.corr_dict = None
        self.in_features = None
        self.feature_file_path = "src/models/support/best_features.csv"
        self.model = LinearSVR(random_state=0)
        self.random_start = random_start
        self.n_start_feats = n_start_feats

    def run(self, iterations: int):
        """
        Every other loop add/remove
        Select for add by correlation group, one from random selection of them
        Select for removal a random sample up to the size of in_features
        :param iterations:
        :return:
        """
        x_train, x_test, y_train, y_scaler = self.data_class.load_data()

        self.create_corr_groups(x_train)

        self.init_features(self.random_start, self.n_start_feats)

        self.in_features = dict([(feat, i) for i, group in self.corr_dict.items() for feat in group["in"]])

        self.set_benchmark(x_train, y_train, y_scaler)

        for i in range(iterations):

            np.random.seed(int(time.time()))

            batch_size = self.starting_batch_size

            # Extract selected features from corr_dict, create new dict with feat as key and group as value
            self.in_features = dict([(feat, i) for i, group in self.corr_dict.items() for feat in group["in"]])

            if self.benchmark >= self.last_benchmark:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            print("\nNew Benchmark RMSE:", '{0:.2f}'.format(self.benchmark), " iteration: ", i, " no improve: ",
                  self.no_improvement_count,
                  " feats: ", len(self.in_features), end="", flush=True)
            self.last_benchmark = self.benchmark

            batch_size += self.multiplier * self.no_improvement_count

            if self.no_improvement_count > 15:
                print("\nEarly stopping....")
                break

            if i % 2 == 0:  # Adding a feature
                self.add_feature(x_train, y_train, y_scaler, batch_size)
            else:
                last_removal_n = ((self.no_improvement_count - 2) * self.multiplier) + self.starting_batch_size
                if self.no_improvement_count > 1 and last_removal_n > len(self.in_features):
                    print(" ....skipping removal", end="", flush=True)
                    continue
                self.remove_feature(x_train, y_train, y_scaler, batch_size)
            self.set_benchmark(x_train, y_train, y_scaler)
        # Set final benchark after iterations complete or early stopping
        self.set_benchmark(x_train, y_train, y_scaler)
        return

    def init_features(self, random_start=True, n_start_feats=None):

        if not random_start:
            # Starting Point A: Read a csv file
            # Upload a starting point from a csv dataframe with no index
            feature_df = pd.read_csv(self.feature_file_path)
            feature_list = list(np.squeeze(feature_df.values))
            # Find the dict key for each feature and add to the list
            for feat in feature_list:
                for group in self.corr_dict.keys():
                    if feat in self.corr_dict[group]["out"]:
                        self.corr_dict[group]["in"].add(feat)
                        self.corr_dict[group]["out"].remove(feat)
                        break
        else:
            # Starting Point B: randomly select features from half of the correlation
            # groups (or arbitrary number of them)
            # Start with 100 features
            np.random.seed(int(time.time()))
            choices = np.random.choice(range(len(self.corr_dict)), size=n_start_feats, replace=False)
            for c in choices:
                # if not len(corr_dict[c]["out"]):  # Ensure there are more to add from group
                self.corr_dict[c]["in"].add(self.corr_dict[c]["out"].pop())
        return

    def set_benchmark(self, x_train, y_train, y_scaler):
        # Measure model benchmark
        benchmark = self.validation.score_regressor(x_train.loc[:, self.in_features], y_train, self.model, y_scaler,
                                                    pos_split=y_scaler.transform([[2.1]]), verbose=0)
        self.benchmark = np.mean(benchmark["root_mean_sq_error"])
        return

    def add_feature(self, x_train, y_train, y_scaler, batch_size):

        test_feats_for_addition = dict()
        # Pick random group, set replace=True if lower correlation threshold used for groupings
        choices = np.random.choice(range(len(self.corr_dict)), size=min(len(self.corr_dict), batch_size * 10),
                                   replace=False)
        k = 0
        # Test if any are out and pick one randomly
        for c in choices:
            if len(self.corr_dict[c]["out"]) > 0:
                feat = random.sample(self.corr_dict[c]["out"], 1)[0]  # Pull random feature
                test_feats_for_addition[feat] = c  # Store the corr group
                k += 1
            if k == batch_size:
                break

        list_size = int(math.ceil(len(test_feats_for_addition) / self.n_jobs))
        # Create list of even lists for parallel
        par_list = [list(test_feats_for_addition)[i:i + list_size]
                    for i in range(0, len(test_feats_for_addition), list_size)]
        par_results = Parallel(n_jobs=self.n_jobs)(
            delayed(par_addition)(feature_list, self.in_features, x_train, y_train, self.model, y_scaler)
            for feature_list in par_list)
        add_dict = dict([(key_, value_) for sub_dict in par_results for key_, value_ in sub_dict.items()])

        final_addition = sorted(add_dict, key=add_dict.get, reverse=True)[-1]
        # Add and update corr dict if improves score
        if add_dict[final_addition] < self.benchmark:
            self.corr_dict[test_feats_for_addition[final_addition]]["in"].add(final_addition)
            self.corr_dict[test_feats_for_addition[final_addition]]["out"].remove(final_addition)
        return

    def remove_feature(self, x_train, y_train, y_scaler, batch_size):
        # * Test the individual removal of a number of features, each from a different correlation group.
        # Max this out at the number of features or close to for batch_size min(n_feats, batch_size)
        test_feats_for_removal = dict()
        # Choose testing features randomly
        choices = np.random.choice(range(len(self.in_features)), size=min(len(self.in_features), batch_size),
                                   replace=False)
        for i_, feat in enumerate(self.in_features.keys()):
            if i_ in choices:
                test_feats_for_removal[feat] = self.in_features[feat]

        list_size = int(math.ceil(len(test_feats_for_removal) / self.n_jobs))
        # Create list of even lists for parallel
        par_list = [list(test_feats_for_removal)[i:i + list_size]
                    for i in range(0, len(test_feats_for_removal), list_size)]
        par_results = Parallel(n_jobs=self.n_jobs)(
            delayed(par_removal)(feature_list, self.in_features, x_train, y_train, self.model, y_scaler)
            for feature_list in par_list)
        remove_dict = dict([(key_, value_) for sub_dict in par_results for key_, value_ in sub_dict.items()])

        # Find the best of those tested
        final_removal = sorted(remove_dict, key=remove_dict.get, reverse=True)[-1]
        # Remove and update corr dict if improves score
        if remove_dict[final_removal] <= self.benchmark:
            self.corr_dict[test_feats_for_removal[final_removal]]["out"].add(final_removal)
            self.corr_dict[test_feats_for_removal[final_removal]]["in"].remove(final_removal)
        return

    def create_corr_groups(self, x_train):
        corr_matrix = x_train.corr()
        corr_matrix.loc[:, :] = np.tril(corr_matrix, k=-1)
        already_in = set()
        corr_result = []
        for col in corr_matrix:
            correlated = corr_matrix[col][np.abs(corr_matrix[col]) > self.corr_threshold].index.tolist()
            if correlated and col not in already_in:
                already_in.update(set(correlated))
                correlated.append(col)
                corr_result.append(correlated)
            elif col not in already_in:
                already_in.update(set(col))
                corr_result.append([col])
        # Create a feature selection dictionary: Contains all features, grouped by correlation
        # Within each corr group there's an "in" and "out" portion for tracking selection
        self.corr_dict = dict(
            [(i, {"out": set(feats), "in": set([])}) for i, feats in zip(range(len(corr_result)), corr_result)])
        return
