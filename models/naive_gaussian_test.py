# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from naive_gaussian import NaiveGaussianModel
import numpy as np
from pdb import set_trace


def test_naive_gaussian():
    n_samples = 2
    n_features = 2
    m = NaiveGaussianModel()

    X = np.array([[1, 2], [5, 6]])
    m.fit(X)

    assert np.allclose(m.model['mean'], np.array([3, 4]))
    assert np.allclose(m.model['std'], np.ones(2) * np.sqrt(8))

    X = np.array([[3, 5]])
    score, max_score_idx, full_scores, above_mean = m.score(X)
    assert np.allclose(score, np.array([0.125]))
    assert np.allclose(max_score_idx, np.array([1]))
    assert np.allclose(full_scores, np.array([[0, 0.125]]))
    assert np.allclose(above_mean, np.array([[False, True]]))

    # example with nans
    X = np.array([[1, np.nan], [2, np.nan]])
    m.fit(X)

    assert np.allclose(m.model['mean'][0], 1.5)
    assert np.isnan(m.model['mean'][1])
    assert np.allclose(m.model['std'][0], 1.0/np.sqrt(2))
    assert np.isnan(m.model['std'][1])

    X = np.array([[2, 5]])
    score, max_score_idx, full_scores, above_mean = m.score(X)
    assert np.allclose(score, np.array([np.nan]), equal_nan=True)
    # Don't check because we no longer use np.nanargmax.
    #assert np.allclose(max_score_idx, np.array([0]))
    assert np.allclose(full_scores, np.array([[0.5, np.nan]]), equal_nan=True)
    assert np.allclose(above_mean, np.array([[True, False]]))
