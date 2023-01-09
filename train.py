import pandas as pd
import numpy as np
import sklearn
import pickle
import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'
MODEL_SAVE_PATH = 'model.pk'
TRAIN_SIZE = 0.8

folds_dict = {f'Fold 1': [['C', 'D', 'E'], ['A', 'B']],
              'Fold 2': [['B', 'D', 'E'], ['A', 'C']],
              'Fold 3': [['B', 'C', 'E'], ['A', 'D']],
              'Fold 4': [['B', 'C', 'D'], ['A', 'E']],
              'Fold 5': [['A', 'D', 'E'], ['B', 'C']],
              'Fold 6': [['A', 'C', 'E'], ['B', 'D']],
              'Fold 7': [['A', 'C', 'D'], ['B', 'E']],
              'Fold 8': [['A', 'B', 'E'], ['C', 'D']],
              'Fold 9': [['A', 'B', 'D'], ['C', 'E']],
              'Fold 10': [['A', 'B', 'C'], ['D', 'E']]}


def preprocess(features, labels):
    features = utils.one_hot_encode(features)
    print(features.head())
    features = utils.fill_NaN_with_mean(features)
    labels = utils.fill_NaN_with_mean(labels)

    test_features = utils.load_data(TEST_DATA_PATH)[0]
    test_features = utils.one_hot_encode(test_features)
    test_features = utils.fill_NaN_with_mean(test_features)

    columns = utils.get_common_columns(features, test_features)
    features = features[columns]

    return features, labels


def train(X, Y):

    kf = GroupKFold(n_splits=5)
    C_list = np.logspace(-10, 0, num=100, base=10)
    solver_list = ['newton-cholesky']
    best_C = 0
    best_solver = 'liblinear'
    best_score = 0
    for C in C_list:
        for solver in solver_list:
            score_mean = 0
            model = LogisticRegression(max_iter=500, C=C, dual=False, tol=1e-4, penalty="l2", solver=solver)
            for (key, fold) in folds_dict.items():
                X_fold_train = X[X['product_code'].isin(fold[0])]
                Y_fold_train = Y[X['product_code'].isin(fold[0])]
                X_fold_val = X[X['product_code'].isin(fold[1])]
                Y_fold_val = Y[X['product_code'].isin(fold[1])]

                X_fold_train = X_fold_train.drop(
                    labels=['product_code'], axis='columns')
                X_fold_val = X_fold_val.drop(
                    labels=['product_code'], axis='columns')

                model.fit(X_fold_train.values, Y_fold_train.values)
                pred = model.predict_proba(X_fold_val.values)[:, 1].reshape(-1, 1)
                score = roc_auc_score(Y_fold_val.values, pred)
                score_mean += score / len(folds_dict)
            print('C = %.9f, solver = %18s, score_mean = %.6f' % (C, solver, score_mean))
            if score_mean > best_score:
                best_C = C
                best_solver = solver
                best_score = score_mean

    print('Best C =', best_C)
    print('Best Solver=', best_solver)
    model = LogisticRegression(max_iter=500, C=best_C, dual=False, tol=1e-4, penalty="l2", solver=best_solver)
    model.fit(X.drop(labels=['product_code'], axis='columns').values, Y.values)

    return model


if __name__ == '__main__':
    features_raw, labels_raw = utils.load_data(TRAIN_DATA_PATH, label='failure')

    print('Preprocessing...')
    features, labels = preprocess(features_raw, labels_raw)
    print(features.columns.values)

    print('Splitting data...')
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, labels, random_state=109550133, train_size=TRAIN_SIZE)
    print('training...')

    model = train(X_train, Y_train)

    X_val = X_val.drop(labels=['product_code'], axis='columns')

    pred = model.predict_proba(X_val.values)[:, 1].reshape(-1, 1)
    score = roc_auc_score(Y_val.values, pred)
    print('score = ', score)

    features_train = features.drop(labels=['product_code'], axis='columns')

    model.fit(features_train.values, labels.values)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        print('OK. Model has been trained with:')
        print('    Number of data:', len(features.index))
        print('Number of features:', len(features.columns))
        print('Save at', MODEL_SAVE_PATH)