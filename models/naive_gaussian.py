# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pdb import set_trace


class NaiveGaussianModel():
    def __init__(self, model=dict(), model_params=None):
        # parameters of model(s)
        self.model = model
        self.model_params = model_params
        if (self.model and self.model_params
                and ('min_std' in self.model_params.keys())):
            if len(self.model_params['min_std'].shape) != 1:
                raise ValueError('min_std field in model_params should '
                                 'have dimension 1.')
            self.model['std'] = np.maximum(self.model['std'],
                                           self.model_params['min_std'])

    def fit(self, X):
        # X is a numpy array with shape (n_samples, n_features)
        if len(X.shape) != 2:
            raise ValueError('Input X must have a shape of 2.')
        self.model['mean'] = np.nanmean(X, axis=0)
        if self.model_params and ('min_std' in self.model_params.keys()):
            self.model['std'] = np.maximum(np.nanstd(X, axis=0, ddof=1),
                                           self.model_params['min_std'])
        else:
            self.model['std'] = np.nanstd(X, axis=0, ddof=1)

    def score(self, X):
        # anomaly score for samples in X
        if not self.model:
            raise Exception('Need to first run fit() before predict.')

        if len(X.shape) != 2:
            raise ValueError('Input X must have a shape of 2.')

        # fill nans with means
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.model['mean'], inds[1])

        tmp_diff = X - self.model['mean']
        full_scores = (np.square(tmp_diff) / np.square(self.model['std']))
        max_score_idx = np.argmax(full_scores, axis=1)
        score = full_scores.sum(axis=1)
        above_mean = tmp_diff > 0

        return score, max_score_idx, full_scores, above_mean
