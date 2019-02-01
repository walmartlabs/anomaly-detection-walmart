# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pdb import set_trace


class HierarchicalModel():
    def __init__(self, sub_model, hierarchy_mappings=None,
                 min_samples=9, model=dict(), model_params=None):
        # parameters of model(s)
        # hierarchy_mappings is a list of numpy arrays corresponding
        # to full list of hierachies, e.g.,
        # sub_category_id, category_id, department_id, ...
        # numpy arrays would be of length of number of sub_category_ids
        self.hierarchy_mappings = hierarchy_mappings
        self.sub_model = sub_model
        self.min_samples = min_samples
        self.model = model
        self.model_params = model_params

    def fit(self, X, sample_hierarchy_labels, feature_hierarchy_level):
        # X is a numpy array with shape (n_samples, n_features)
        # sample_hierarchy_labels is a list of numpy arrays corresponding
        # to the id associated with each hierarchy. It is ordered
        # by increasing broader hierachy, e.g., in this order:
        # sub_category_id, category_id, department_id,
        # super_department_id, division_id. It's a list of numpy
        # arrays each with length n_samples.
        # feature_hierarchy_level is a numpy array with length n_features
        # which corresponds to index of the hierachy level
        # we want to fit each of the features.
        n_samples, n_features = X.shape
        n_levels = len(sample_hierarchy_labels)

        hierarchy_id = np.unique(self.hierarchy_mappings[0])
        if len(feature_hierarchy_level) != n_features:
            raise ValueError('Length of input feature_hierarchy_level '
                             'needs to equal number of features (cols) '
                             ' in X.')
        if len(self.hierarchy_mappings) != n_levels:
            raise ValueError('Length of input sample_hierarchy_labels '
                             'needs to equal number of hierarchy levels '
                             'in hierarchy_mappings.')

        # calculate params for entire dataset, use -1 as key for full fit
        if n_samples > 2:
            m = self.sub_model(model_params=self.model_params)
            m.fit(X)
            model_info = m.model.keys()
            self.model[-1] = m.model.copy()
        else:
            raise ValueError('Not enough data points for fitting.')

        # initialize model in advance
        for j in hierarchy_id:
            tmp_dict = dict()
            for key in model_info:
                tmp_dict[key] = np.empty(n_features) * np.nan
            self.model[j] = tmp_dict

        # cache results
        cache = dict()

        for i in xrange(n_features):
            feature_level = int(feature_hierarchy_level[i])
            for j in hierarchy_id:
                flag = self.hierarchy_mappings[0] == j
                model_out = None
                for k in xrange(feature_level, n_levels):
                    level_list = sample_hierarchy_labels[k]
                    level_id = np.unique(self.hierarchy_mappings[k][flag])
                    if (i, k, level_id[0]) in cache.keys():
                        model_out = cache[(i, k, level_id[0])]
                        break
                    else:
                        level_flag = level_list == level_id[0]
                        X_t = X[level_flag][:, i].reshape(-1, 1)
                        n_samples = (~np.isnan(X_t)).sum()
                        if n_samples > self.min_samples:
                            m = self.sub_model()
                            m.fit(X_t)
                            model_out = m.model
                            cache[(i, k, level_id[0])] = model_out.copy()
                            break
                if model_out:
                    for key in model_info:
                        self.model[j][key][i] = model_out[key]
                else:
                    for key in self.model[-1].keys():
                        self.model[j][key][i] = self.model[-1][key][i].copy()

        # create an instance of sub_model for every lowest_hierarchy_label
        self.model = {k: self.sub_model(model=v,
                                        model_params=self.model_params)
                      for k, v in self.model.items()}

    def score(self, X, lowest_hierarchy_label):
        # anomaly score for samples in X
        # hierachy_id is the id of the lowest hierachy level, e.g.,
        # sub_category_id for each of the samples

        if not self.model:
            raise Exception('Need to first run fit() before predict.')

        n_samples, n_features = X.shape
        if len(lowest_hierarchy_label) != n_samples:
            raise ValueError('Length of input lowest_hierarchy_label '
                             'needs to equal number of samples (rows) '
                             ' in X.')

        score = np.empty(n_samples) * np.nan
        max_score_idx = np.empty(n_samples) * np.nan
        full_scores = np.empty((n_samples, n_features)) * np.nan
        above_mean = np.empty((n_samples, n_features)) * np.nan
        hierarchy_id_unique = set(lowest_hierarchy_label)
        for i in hierarchy_id_unique:
            flag = lowest_hierarchy_label == i
            X_t = X[flag]
            if i in self.model:
                (tmp_score, tmp_max_score_idx,
                 tmp_full_scores, tmp_above_mean) = self.model[i].score(X_t)
            else:
                (tmp_score, tmp_max_score_idx,
                 tmp_full_scores, tmp_above_mean) = self.model[-1].score(X_t)
            max_score_idx[flag] = tmp_max_score_idx
            score[flag] = tmp_score
            full_scores[flag, :] = tmp_full_scores
            above_mean[flag, :] = tmp_above_mean

        return score, max_score_idx, full_scores, above_mean
