# Copyright (c) Walmart Inc.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from argparse import ArgumentParser
from pdb import set_trace


def handle_missing_values(train, test):
    from sklearn.preprocessing import Imputer
    imp_cols = train.columns
    # put 0 in columns with all nan values
    default = 0
    train.loc[:, np.isnan(train).all(axis=0)] = default
    # replace nan values with mean of the column
    imp_model = Imputer(missing_values=np.nan, strategy='mean')
    train_np = imp_model.fit_transform(train)
    train = pd.DataFrame(train_np, columns=imp_cols)

    # apply to val and test set
    test_np = imp_model.transform(test)
    test = pd.DataFrame(test_np, columns=imp_cols)

    return train, test


def apply_cv(model, model_type, test_split, random_state):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X = test_split[model_features].values
    y = test_split['is_anomaly'].values
    if model_type == 'gaussiannb':
        sub_cat_id = test_split['sub_category_id'].values
    precision_cv_val = []
    recall_cv_val = []
    f1_cv_val = []
    precision_cv_full = []
    recall_cv_full = []
    f1_cv_full = []
    thresholds_cv_full = []
    auc_cv_val = []
    precision_cv_test = []
    recall_cv_test = []
    f1_cv_test = []
    val_predict_time_cv = []
    test_predict_time_cv = []
    thresholds_cv_val = []
    test_scores = []
    val_scores = []
    test_true = []
    val_true = []
    val_indexes = []
    test_indexes = []
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_val, X_test = X[train_index], X[test_index]
        y_val, y_test = y[train_index], y[test_index]
        if model_type == 'gaussiannb':
            sub_cat_val, sub_cat_test = (sub_cat_id[train_index],
                                         sub_cat_id[test_index])

        # select threshold using validation set
        start = time.time()
        if model_type == 'gaussiannb':
            X_transformed = transform_features(X_val)
            (anomaly_score, max_score_idx,
             y_full_scores,
             y_above_mean) = model.model.score(X_transformed, sub_cat_val)
            val_score = np.squeeze(np.around(anomaly_score, decimals=1))
        elif model_type == 'iforest':
            val_score = -1. * model.score_samples(X_val)
        elif model_type == 'autoencoder':
            val_score = model.decision_function(X_val)
        elif model_type == 'xgboost':
            val_score = model.predict(X_val)
        elif model_type == 'rf':
            val_score = model.predict_proba(X_val)[:, 1]
        end = time.time()
        val_predict_time = end - start
        val_predict_time_cv.append(val_predict_time)

        # Calculate best threshold
        (precision_cv, recall_cv,
         thresholds_cv) = precision_recall_curve(y_val, val_score, pos_label=1)
        beta = 1.
        beta_sq = beta ** 2
        fbeta_score = ((1 + beta_sq) * (precision_cv * recall_cv)
                       / ((beta_sq * precision_cv) + recall_cv))
        precision_cv_full.append(precision_cv)
        recall_cv_full.append(recall_cv)
        thresholds_cv_full.append(thresholds_cv)
        f1_cv_full.append(fbeta_score)
        # maximize f_beta score
        max_idx = np.argmax(fbeta_score)
        threshold = thresholds_cv[max_idx]
        thresholds_cv_val.append(threshold)
        y_pred = (val_score >= threshold).astype(int)
        precision_val = precision_score(y_val, y_pred)
        recall_val = recall_score(y_val, y_pred)
        f1_val = f1_score(y_val, y_pred)
        precision_cv_val.append(precision_val)
        recall_cv_val.append(recall_val)
        f1_cv_val.append(f1_val)
        # AUC of precision recall in val test
        auc_precision_recall = auc(recall_cv, precision_cv)
        auc_cv_val.append(auc_precision_recall)

        # Test set
        start = time.time()
        if model_type == 'gaussiannb':
            X_transformed = transform_features(X_test)
            (anomaly_score, max_score_idx,
             y_full_scores,
             y_above_mean) = model.model.score(X_transformed, sub_cat_test)
            test_score = np.squeeze(np.around(anomaly_score, decimals=1))
        elif model_type == 'iforest':
            test_score = -1. * model.score_samples(X_test)
        elif model_type == 'autoencoder':
            test_score = model.decision_function(X_test)
        elif model_type == 'xgboost':
            test_score = model.predict(X_test)
        elif model_type == 'rf':
            test_score = model.predict_proba(X_test)[:, 1]

        y_pred = (test_score >= threshold).astype(int)
        end = time.time()
        test_predict_time = end - start
        test_predict_time_cv.append(test_predict_time)

        precision_test = precision_score(y_test, y_pred)
        recall_test = recall_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)
        print("Test set precision = " + str(precision_test))
        print("Test set recall = " + str(recall_test))
        print("Test set F1 score = " + str(f1_test))
        precision_cv_test.append(precision_test)
        recall_cv_test.append(recall_test)
        f1_cv_test.append(f1_test)
        test_scores.append(test_score)
        val_scores.append(val_score)
        test_true.append(y_test)
        val_true.append(y_val)
        val_indexes.append(train_index)
        test_indexes.append(test_index)

    return (precision_cv_val, recall_cv_val, f1_cv_val, auc_cv_val,
            precision_cv_test, recall_cv_test, f1_cv_test,
            test_predict_time_cv, val_predict_time_cv, thresholds_cv_val,
            precision_cv_full, recall_cv_full, f1_cv_full, thresholds_cv_full,
            test_scores, val_scores, test_true, val_true, val_indexes,
            test_indexes)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        choices=['gaussiannb', 'iforest', 'autoencoder',
                                 'xgboost', 'rf'],
                        default='gaussiannb',
                        help='Choose model to train and evaluate.')
    parser.add_argument("--input-filename",
                        default="data.pkl",
                        type=str,
                        help="File name to load data from.")
    parser.add_argument("--output-filename",
                        default="output.pkl",
                        type=str,
                        help="File name to write results.")
    parsed_args = parser.parse_args()

    # Load data
    data = pickle.load(open(parsed_args.input_filename, 'rb'))
    train = data['train']
    test = data['test']
    col_types = data['col_types']

    if parsed_args.model == 'gaussiannb':
        from models.gaussiannb_fit import GaussianNBFit, transform_features
        train.rename(index=str,
                     columns={"price1_plus_shipping": "price1",
                              "price3_plus_shipping": "price3"},
                     inplace=True)
        test.rename(index=str,
                    columns={"price1_plus_shipping": "price1",
                             "price3_plus_shipping": "price3"},
                    inplace=True)
        model_features = ['cost', 'price', 'price1',
                          'price2', 'price3',
                          'avg_historical_price']
        num_features = len(model_features)

        model = GaussianNBFit(train, 2, 0.01)
        start = time.time()
        model.fit(beta=1.)
        end = time.time()
        fit_time = end - start
    elif parsed_args.model == 'iforest':
        #impute features with mean
        train, test = handle_missing_values(train, test)
        from sklearn.ensemble import IsolationForest
        model_features = list(train.columns)
        model_features.remove('is_anomaly')
        num_features = len(model_features)

        model = IsolationForest(behaviour='new', n_estimators=100,
                                contamination=0.1, n_jobs=-1, max_features=0.1,
                                max_samples=0.05)
        start = time.time()
        X_train = train[train['is_anomaly'] == 0][model_features].values
        model.fit(X_train)
        end = time.time()
        fit_time = end - start
    elif parsed_args.model == 'autoencoder':
        from pyod.models.auto_encoder import AutoEncoder
        # one hot encoded features
        le = LabelEncoder()
        le.fit(train['super_department_id'].values)
        enc = OneHotEncoder()
        one_hot_train = enc.fit_transform(le.transform(train['super_department_id'].values).reshape(-1, 1))
        one_hot_test = enc.transform(le.transform(test['super_department_id'].values).reshape(-1, 1))

        features_to_remove = ['hierarchy', 'price_transformed']
        train_columns = list(train.columns)
        new_col_types = []
        for i in range(len(train_columns)):
            if (col_types[i] in features_to_remove):
                del train[train_columns[i]]
                del test[train_columns[i]]
            else:
                new_col_types.append(col_types[i])

        # gather price features
        train_columns = list(train.columns)
        print(len(train_columns))
        price_features = []
        for i in range(len(new_col_types)):
            if new_col_types[i] == 'price':
                price_features.append(train_columns[i])

        # add log based features
        # first cost
        for col in (set(price_features) - set(['cost'])):
            train[str(col) + '_cost_log'] = np.log((train[col] + 10.)/(train['cost'] + 10.))
            test[str(col) + '_cost_log'] = np.log((test[col] + 10.)/(test['cost'] + 10.))
            new_col_types.append('log_new_cost_transformed')

        # now price
        for col in (set(price_features) - set(['cost', 'price'])):
            train[str(col) + '_price_log'] = np.log((train[col] + 10.)/(train['price'] + 10.))
            test[str(col) + '_price_log'] = np.log((test[col] + 10.)/(test['price'] + 10.))
            new_col_types.append('log_new_price_transformed')

        model_features = list(train.columns)
        model_features.remove('is_anomaly')
        num_features = len(model_features)

        train, test = handle_missing_values(train, test)

        model = AutoEncoder(output_activation='tanh', batch_size=512,
                            contamination=0.000001, epochs=100)
        start = time.time()
        X_train = train[train['is_anomaly'] == 0][model_features].values
        model.fit(X_train)
        end = time.time()
        fit_time = end - start
    elif parsed_args.model == 'xgboost':
        #impute features with mean
        train, test = handle_missing_values(train, test)
        from xgboost.sklearn import XGBModel
        model_features = list(train.columns)
        model_features.remove('is_anomaly')
        num_features = len(model_features)

        model = XGBModel(n_estimators=400, max_depth=5, n_jobs=-1,
                         objective='binary:logistic')
        start = time.time()
        model.fit(train[model_features].values, train['is_anomaly'].values)
        end = time.time()
        fit_time = end - start
    elif parsed_args.model == 'rf':
        #impute features with mean
        train, test = handle_missing_values(train, test)
        from sklearn.ensemble import RandomForestClassifier
        model_features = list(train.columns)
        model_features.remove('is_anomaly')
        num_features = len(model_features)

        model = RandomForestClassifier(n_estimators=400, max_depth=80,
                                       class_weight={1: 3, 0: 1}, n_jobs=-1)
        start = time.time()
        model.fit(train[model_features], train['is_anomaly'])
        end = time.time()
        fit_time = end - start

    print("Training time: " + str(fit_time) + " seconds.")
    print("Number of features = " + str(num_features))

    percent_anomalies = np.logspace(np.log10(0.001), np.log10(0.25), 10)
    seed_split = 1
    seed_cv = 100
    num_n = len(test[test['is_anomaly'] == 0])
    num_p = len(test[test['is_anomaly'] == 1])
    precision_val_list = []
    recall_val_list = []
    f1_val_list = []
    auc_precision_recall_list = []
    precision_test_list = []
    recall_test_list = []
    f1_test_list = []
    val_predict_time_list = []
    test_predict_time_list = []
    threshold_list = []
    num_anomalies = num_p
    num_normal = []
    test_scores_list = []
    val_scores_list = []
    test_true_list = []
    val_true_list = []
    val_index_list = []
    test_index_list = []
    for p in percent_anomalies:
        test_size = (num_p / p - num_p) / num_n
        print(test_size)
        if not (test_size >= 1.0):
            _, test_n = train_test_split(test[test['is_anomaly'] == 0],
                                         test_size=test_size,
                                         random_state=seed_split)
            seed_split += 1
            test_split = pd.concat([test[test['is_anomaly'] == 1], test_n])
            num_normal.append(len(test_n))
        else:
            test_split = test
            num_normal.append(num_n)

        (precision_cv_val, recall_cv_val, f1_cv_val, auc_cv_val,
         precision_cv_test, recall_cv_test, f1_cv_test,
         test_predict_time_cv, val_predict_time_cv, thresholds_cv_val,
         precision_cv_full, recall_cv_full, f1_cv_full, thresholds_cv_full,
         test_scores, val_scores, test_true, val_true, val_indexes,
         test_indexes) = apply_cv(model, parsed_args.model,
                                  test_split, seed_cv)

        seed_cv += 1

        precision_val_list.append(precision_cv_val)
        recall_val_list.append(recall_cv_val)
        f1_val_list.append(f1_cv_val)
        auc_precision_recall_list.append(auc_cv_val)
        precision_test_list.append(precision_cv_test)
        recall_test_list.append(recall_cv_test)
        f1_test_list.append(f1_cv_test)
        val_predict_time_list.append(val_predict_time_cv)
        test_predict_time_list.append(test_predict_time_cv)
        threshold_list.append(thresholds_cv_val)
        test_scores_list.append(test_scores)
        val_scores_list.append(val_scores)
        test_true_list.append(test_true)
        val_true_list.append(val_true)
        val_index_list.append(val_indexes)
        test_index_list.append(test_indexes)

    # calculate batch and streaming predict times
    test_size = 750. / len(test[test['is_anomaly'] == 0])
    _, test_time_n = train_test_split(test[test['is_anomaly'] == 0],
                                      test_size=test_size, random_state=0)
    test_size = 250. / len(test[test['is_anomaly'] == 1])
    _, test_time_p = train_test_split(test[test['is_anomaly'] == 1],
                                      test_size=test_size, random_state=0)
    test_time = pd.concat([test_time_p, test_time_n])

    # batch_predict_time
    X = test_time[model_features].values
    y = test_time['is_anomaly'].values
    if parsed_args.model == 'gaussiannb':
        sub_cat_id = test_time['sub_category_id'].values
    start = time.time()
    if parsed_args.model == 'gaussiannb':
        X_transformed = transform_features(X)
        (score, max_score_idx,
         y_full_scores,
         y_above_mean) = model.model.score(X_transformed, sub_cat_id)
    elif parsed_args.model == 'iforest':
        score = -1. * model.score_samples(X)
    elif parsed_args.model == 'autoencoder':
        score = model.decision_function(X)
    elif parsed_args.model == 'xgboost':
        score = model.predict(X)
    elif parsed_args.model == 'rf':
        score = model.predict_proba(X)[:, 1]
    end = time.time()
    batch_predict_time = end - start
    print("Batch prediction time: " + str(batch_predict_time))

    # streaming_predict_time
    X = test_time[model_features].values
    y = test_time['is_anomaly'].values
    start = time.time()
    for i in range(X.shape[0]):
        if parsed_args.model == 'gaussiannb':
            X_transformed = transform_features(X[i, :].reshape(1, -1))
            (score, max_score_idx,
             y_full_scores, y_above_mean
             ) = model.model.score(X_transformed,
                                   np.array(sub_cat_id[i]).reshape(-1))
        elif parsed_args.model == 'iforest':
            score = -1. * model.score_samples(X[i, :].reshape(1, -1))
        elif parsed_args.model == 'autoencoder':
            score = model.decision_function(X[i, :].reshape(1, -1))
        elif parsed_args.model == 'xgboost':
            score = model.predict(X[i, :].reshape(1, -1))
        elif parsed_args.model == 'rf':
            score = model.predict_proba(X[i, :].reshape(1, -1))[:, 1]
    end = time.time()
    streaming_predict_time = end - start
    print("Streaming prediction time: " + str(streaming_predict_time))

    # save results
    results = dict()
    results['precision_val_list'] = precision_val_list
    results['recall_val_list'] = recall_val_list
    results['f1_val_list'] = f1_val_list
    results['precision_test_list'] = precision_test_list
    results['recall_test_list'] = recall_test_list
    results['f1_test_list'] = f1_test_list
    results['fit_time'] = fit_time
    results['val_predict_time_list'] = val_predict_time_list
    results['test_predict_time_list'] = test_predict_time_list
    results['threshold_list'] = threshold_list
    results['auc_list'] = auc_precision_recall_list
    results['num_normal'] = num_normal
    results['num_anomalies'] = num_anomalies
    results['num_features'] = num_features
    results['percent_anomalies'] = percent_anomalies
    results['test_scores_list'] = test_scores_list
    results['val_scores_list'] = val_scores_list
    results['test_true_list'] = test_true_list
    results['val_true_list'] = val_true_list
    results['val_index_list'] = val_index_list
    results['test_index_list'] = test_index_list
    results['batch_predict_time'] = batch_predict_time
    results['streaming_predict_time'] = streaming_predict_time
    pickle.dump(results, open(parsed_args.output_filename, "wb"))
