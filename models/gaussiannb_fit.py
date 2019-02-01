# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import json
import copy
from hierarchical_model import HierarchicalModel
from naive_gaussian import NaiveGaussianModel
from sklearn import metrics, model_selection
import pandas as pd
from pdb import set_trace


def transform_features(data):
    # feature 0 : cost
    # feature 1 : price
    # feature 2 : price1
    # feature 3 : price2
    # feature 4 : price3
    # feature 5 : avg_historical_price

    # No need to set nan -- already done earlier. Significant speedup without this.
    # data[data < 0] = np.nan
    # No need to convert to float -- already done earlier. Optimize speed.
    # data = data.astype(float)
    data[:, 1] = np.log((data[:, 1] + 10.) / (data[:, 0] + 10.))
    data[:, 2] = np.log((data[:, 2] + 10.) / (data[:, 0] + 10.))
    data[:, 3] = np.log(np.maximum((data[:, 3] + 10.)
                                   / (data[:, 0] + 10.) - 0.5, 0.00001))
    data[:, 4] = np.log((data[:, 4] + 10.) / (data[:, 0] + 10.))
    data[:, 5] = np.log((data[:, 5] + 10.) / (data[:, 0] + 10.))
    return data[:, 1:]


class GaussianNBFit():
    def __init__(self, data_dframe, hierarchy_level, model_min_std):
        if ((data_dframe is None) or (hierarchy_level is None) or (model_min_std is None)):
            print("Please provide all required data: data_dframe, hierarchy_level, model_min_std.")
            raise ValueError("Please provide all required data: data_dframe, hierarchy_level, model_min_std.")

        self.hierarchy_columns = ['sub_category_id', 'category_id', 'department_id', 'super_department_id', 'division_id']
        data_dframe.loc[(data_dframe[self.hierarchy_columns] < 0).any(axis=1), self.hierarchy_columns] = np.nan
        data_dframe = data_dframe.loc[data_dframe[self.hierarchy_columns].notnull().all(axis=1)]
        # convert all hierarchy columns to int
        for col in self.hierarchy_columns:
            data_dframe[col] = data_dframe[col].map(int)
        self.data_dframe = data_dframe
        self.feature_hierarchy = [self.data_dframe[col].values for col in self.hierarchy_columns]
        self.model_columns = ['cost', 'price', 'price1', 'price2', 'price3', 'avg_historical_price']
        n_features = len(self.model_columns) - 1
        self.feature_levels = np.ones(n_features) * hierarchy_level
        model_params = dict()
        model_params['min_std'] = np.array([model_min_std] * n_features)
        hierarchy_mappings = self.data_dframe[self.hierarchy_columns].drop_duplicates()
        hierarchy_mappings = [hierarchy_mappings[col].values for col in self.hierarchy_columns]
        self.model = HierarchicalModel(NaiveGaussianModel,
                                       hierarchy_mappings=hierarchy_mappings,
                                       min_samples=9,
                                       model_params=model_params,
                                       model=dict())
        self.threshold = None

    def fit(self, beta=0.1, threshold=None):
        flag_not_anomaly = (self.data_dframe['is_anomaly'] == 0)
        X_normal = self.data_dframe[flag_not_anomaly]
        y_normal = self.data_dframe['is_anomaly'][flag_not_anomaly]
        X_cv_anomaly = self.data_dframe[~flag_not_anomaly]
        y_cv_anomaly = self.data_dframe['is_anomaly'][~flag_not_anomaly]

        if not threshold:
            # use cross-validation to select best threshold
            test_size = len(y_cv_anomaly) * 1. / len(y_normal)
            X_train, X_cv_split = model_selection.train_test_split(X_normal,
                                                                   test_size=test_size)
            X_cv = pd.concat([X_cv_anomaly, X_cv_split])
            feature_hierarchy_cv = [X_cv[col].values for col in self.hierarchy_columns]
            feature_hierarchy_train = [X_train[col].values for col in self.hierarchy_columns]

            # convert to numpy array
            X_cv_raw = X_cv[self.model_columns].values
            X_cv_raw[X_cv_raw < 0] = np.nan
            X_cv = transform_features(X_cv_raw)
            y_cv = np.append(np.ones(shape=(len(X_cv_anomaly), 1)),
                             np.zeros(shape=(len(X_cv_split), 1)),
                             axis=0)
            X_train_raw = X_train[self.model_columns].values
            X_train[X_train < 0] = np.nan
            X_train = transform_features(X_train_raw)

            self.model.fit(X_train, feature_hierarchy_train, self.feature_levels)
            (y_score_cv, max_score_idx, y_full_scores, y_above_mean) = self.model.score(X_cv,
                                                                                        feature_hierarchy_cv[0])

            (precision_cv, recall_cv, thresholds_cv) = metrics.precision_recall_curve(y_cv, y_score_cv,
                                                                                      pos_label=1)

            # use f_beta score
            # beta = 1.0 (weights precision equal to recall
            beta_sq = beta ** 2
            fbeta_score = ((1 + beta_sq) * (precision_cv * recall_cv)
                            / ((beta_sq * precision_cv) + recall_cv + 0.001))
            # maximize f_beta score
            max_idx = np.argmax(fbeta_score)
            self.threshold = thresholds_cv[max_idx]

            print("Threshold: " + str(self.threshold))
            print("Beta: " + str(beta))
            print("Fbeta score: " + str(fbeta_score[max_idx]))
            print("Precision: " + str(precision_cv[max_idx]))
            print("Recall: " + str(recall_cv[max_idx]))
        else:
            self.threshold = threshold

        # refit on entire available dataset
        X_raw = self.data_dframe[flag_not_anomaly][self.model_columns].values
        X_raw[X_raw < 0] = np.nan
        X = transform_features(X_raw)

        feature_hierarchy_model = [arr[flag_not_anomaly] for arr in self.feature_hierarchy]
        self.model.fit(X, feature_hierarchy_model, self.feature_levels)

        return True

    def write_to_fileserver(self, filename):
        parameter_types = self.model.model[-1].model.keys()
        data = dict()
        for i in self.model.model.keys():
            data[i] = dict()
            for j in parameter_types:
                data[i][j] = copy.copy(self.model.model[i].model[j].tolist())
        output = dict()
        output['model'] = data
        output['threshold'] = self.threshold
        with open(filename, 'w') as f:
            json.dump(output, f)
