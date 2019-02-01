# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from naive_gaussian import NaiveGaussianModel
from hierarchical_model import HierarchicalModel
import numpy as np
from pdb import set_trace
import copy


def test_hierarchical_model():
    # example 1
    # output should be
    # subcat 1:
    # mean = [4, 4], std = [sqrt(3), 2]
    # subcat 2:
    # mean = [4, 5], std = [sqrt(2), sqrt(20/3)]
    X = np.array([[2, 2], [5, 6], [5, 4], [4, 8]])
    sub_category_id = np.array([1, 1, 1, 2])
    category_id = np.array([3, 3, 3, 3])
    department_id = np.array([4, 4, 4, 4])
    super_department_id = np.array([5, 5, 5, 5])
    division_id = np.array([6, 6, 6, 6])
    sample_hierarchy_labels = [sub_category_id, category_id, department_id,
                               super_department_id, division_id]
    hierarchy_mappings = [np.array([1, 2]), np.array([3, 3]),
                          np.array([4, 4]), np.array([5, 5]),
                          np.array([6, 6])]
    # feature hiearchy = fit both features on sub_category level
    feature_hierarchy_level = np.array([0, 0])
    m = HierarchicalModel(NaiveGaussianModel,
                          hierarchy_mappings=hierarchy_mappings,
                          min_samples=2)
    m.fit(X, sample_hierarchy_labels, feature_hierarchy_level)

    assert np.allclose(m.model[1].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[1].model['std'], np.array([np.sqrt(3), 2]))
    assert np.allclose(m.model[2].model['mean'], np.array([4, 5]))
    assert np.allclose(m.model[2].model['std'], np.array([np.sqrt(2),
                                                         np.sqrt(20.0/3)]))
    assert np.allclose(m.model[-1].model['mean'], np.array([4, 5]))
    assert np.allclose(m.model[-1].model['std'], np.array([np.sqrt(2),
                                                          np.sqrt(20.0/3)]))

    # score = [7.0/12, 2.25, 18 + 3.0/5]
    X = np.array([[3, 5], [4, 7], [10, 3]])
    sub_category_id = np.array([1, 1, 2])
    (score, max_score_idx,
     full_scores, above_mean) = m.score(X, sub_category_id)
    assert np.allclose(score, np.array([7.0/12, 2.25, 18 + 3.0/5]))
    assert np.allclose(max_score_idx, np.array([0, 1, 0]))
    assert np.allclose(score, full_scores.sum(axis=1))
    assert np.allclose(above_mean, np.array([[0, 1], [0, 1], [1, 0]]))

    # score = [2.15]
    X = np.array([[2, 4]])
    sub_category_id = np.array([2])
    (score, max_score_idx,
     full_scores, above_mean) = m.score(X, sub_category_id)
    assert np.allclose(score, np.array([2.15]))
    assert np.allclose(max_score_idx, np.array([0]))
    assert np.allclose(score, full_scores.sum(axis=1))
    assert np.allclose(above_mean, np.array([[0, 0]]))

    # example 2: more heirarchical tests
    #
    X = np.array([[1, 2], [5, 6], [3, 4], [7, 4]])
    sub_category_id = np.array([1, 2, 3, 4])
    category_id = np.array([5, 5, 6, 7])
    department_id = np.array([8, 8, 9, 9])
    super_department_id = np.array([10, 10, 11, 11])
    division_id = np.array([12, 12, 12, 12])
    sample_hierarchy_labels = [sub_category_id, category_id, department_id,
                               super_department_id, division_id]
    hierarchy_mappings = copy.deepcopy(sample_hierarchy_labels)
    feature_hierarchy_level = np.array([0, 0])
    m = HierarchicalModel(NaiveGaussianModel,
                          hierarchy_mappings=hierarchy_mappings,
                          min_samples=2)
    m.fit(X, sample_hierarchy_labels, feature_hierarchy_level)

    assert np.allclose(m.model[1].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[1].model['std'], np.array([np.sqrt(20.0/3),
                                                          np.sqrt(8.0/3)]))
    assert np.allclose(m.model[2].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[2].model['std'], np.array([np.sqrt(20.0/3),
                                                          np.sqrt(8.0/3)]))
    assert np.allclose(m.model[3].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[3].model['std'], np.array([np.sqrt(20.0/3),
                                                          np.sqrt(8.0/3)]))
    assert np.allclose(m.model[4].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[4].model['std'], np.array([np.sqrt(20.0/3),
                                                          np.sqrt(8.0/3)]))
    assert np.allclose(m.model[-1].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[-1].model['std'], np.array([np.sqrt(20.0/3),
                                                           np.sqrt(8.0/3)]))

    X = np.array([[2, 4], [3, 8]])
    sub_category_id = np.array([2, 4])
    (score, max_score_idx,
     full_scores, above_mean) = m.score(X, sub_category_id)
    assert np.allclose(score, np.array([0.6, 6.15]))
    assert np.allclose(max_score_idx, np.array([0, 1]))
    assert np.allclose(score, full_scores.sum(axis=1))
    assert np.allclose(above_mean, np.array([[0, 0], [0, 1]]))

    # example 3: single feature
    #
    X = np.array([[1], [5], [3], [7]])
    sub_category_id = np.array([1, 2, 3, 4])
    category_id = np.array([5, 5, 5, 6])
    department_id = np.array([7, 7, 7, 7])
    super_department_id = np.array([8, 8, 8, 8])
    division_id = np.array([9, 9, 9, 9])
    sample_hierarchy_labels = [sub_category_id, category_id, department_id,
                               super_department_id, division_id]
    hierarchy_mappings = copy.deepcopy(sample_hierarchy_labels)
    # fit at the department level
    feature_hierarchy_level = np.array([2])
    m = HierarchicalModel(NaiveGaussianModel,
                          hierarchy_mappings=hierarchy_mappings,
                          min_samples=2)
    m.fit(X, sample_hierarchy_labels, feature_hierarchy_level)

    assert np.allclose(m.model[1].model['mean'], np.array([4]))
    assert np.allclose(m.model[1].model['std'], np.array([np.sqrt(20.0/3)]))
    assert np.allclose(m.model[2].model['mean'], np.array([4]))
    assert np.allclose(m.model[2].model['std'], np.array([np.sqrt(20.0/3)]))
    assert np.allclose(m.model[3].model['mean'], np.array([4]))
    assert np.allclose(m.model[3].model['std'], np.array([np.sqrt(20.0/3)]))
    assert np.allclose(m.model[4].model['mean'], np.array([4]))
    assert np.allclose(m.model[4].model['std'], np.array([np.sqrt(20.0/3)]))
    assert np.allclose(m.model[-1].model['mean'], np.array([4]))
    assert np.allclose(m.model[-1].model['std'], np.array([np.sqrt(20.0/3)]))

    X = np.array([[2], [3]])
    sub_category_id = np.array([2, 3])
    (score, max_score_idx,
     full_scores, above_mean) = m.score(X, sub_category_id)
    assert np.allclose(score, np.array([0.6, 0.15]))
    assert np.allclose(max_score_idx, np.array([0, 0]))
    assert np.allclose(score, full_scores.sum(axis=1))
    assert np.allclose(above_mean, np.array([[0], [0]]))

    # example 4: features built on different hierarchies
    #
    X = np.array([[1, 4], [5, 2], [3, 7], [7, 3]])
    sub_category_id = np.array([1, 2, 3, 4])
    category_id = np.array([5, 5, 5, 6])
    department_id = np.array([7, 8, 8, 8])
    super_department_id = np.array([9, 9, 9, 9])
    division_id = np.array([10, 10, 10, 10])
    sample_hierarchy_labels = [sub_category_id, category_id, department_id,
                               super_department_id, division_id]
    hierarchy_mappings = copy.deepcopy(sample_hierarchy_labels)
    # fit at category and super department levels respectively
    feature_hierarchy_level = np.array([1, 3])
    m = HierarchicalModel(NaiveGaussianModel,
                          hierarchy_mappings=hierarchy_mappings,
                          min_samples=2)
    m.fit(X, sample_hierarchy_labels, feature_hierarchy_level)

    assert np.allclose(m.model[1].model['mean'], np.array([3, 4]))
    assert np.allclose(m.model[1].model['std'], np.array([2, np.sqrt(14.0/3)]))
    assert np.allclose(m.model[2].model['mean'], np.array([3, 4]))
    assert np.allclose(m.model[2].model['std'], np.array([2, np.sqrt(14.0/3)]))
    assert np.allclose(m.model[3].model['mean'], np.array([3, 4]))
    assert np.allclose(m.model[3].model['std'], np.array([2, np.sqrt(14.0/3)]))
    assert np.allclose(m.model[4].model['mean'], np.array([5, 4]))
    assert np.allclose(m.model[4].model['std'], np.array([2, np.sqrt(14.0/3)]))
    assert np.allclose(m.model[-1].model['mean'], np.array([4, 4]))
    assert np.allclose(m.model[-1].model['std'], np.array([np.sqrt(20.0/3),
                                                           np.sqrt(14.0/3)]))

    X = np.array([[2, 5], [2, 5], [5, 2]])
    sub_category_id = np.array([2, 4, 20])
    (score, max_score_idx,
     full_scores, above_mean) = m.score(X, sub_category_id)
    assert np.allclose(score, np.array([0.25 + 3.0/14, 9.0/4 + 3.0/14,
                                        3.0/20 + 12.0/14]))
    assert np.allclose(max_score_idx, np.array([0, 0, 1]))
    assert np.allclose(score, full_scores.sum(axis=1))
    assert np.allclose(above_mean, np.array([[0, 1], [0, 1], [1, 0]]))
