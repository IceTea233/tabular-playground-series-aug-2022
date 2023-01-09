import pandas as pd
import numpy as np
import sklearn
import pickle
import utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'
MODEL_SAVE_PATH = 'model.pk'
TRAIN_SIZE = 0.8

def preprocess(features, labels):
    features = utils.one_hot_encode(features)
    features = utils.fill_NaN_with_mean(features)
    labels = utils.fill_NaN_with_mean(labels)

    test_features = utils.load_data(TEST_DATA_PATH)[0]
    test_features = utils.one_hot_encode(test_features)
    test_features = utils.fill_NaN_with_mean(test_features)
    
    columns = utils.get_common_columns(features, test_features)
    features = features[columns]

    return features, labels

def train(X, Y):
    model = LogisticRegression(max_iter=500,C=0.1, dual=False, penalty="l2",solver='newton-cg')
    model.fit(X, Y)
    return model

if __name__ == '__main__':
    features, labels = utils.load_data(TRAIN_DATA_PATH, label='failure')
    
    # indexes = features.index[ features['product_code'] == 'A'].tolist()
    # features = features.iloc[indexes]
    # labels = labels.iloc[indexes]
    print('Preprocessing...')
    features, labels = preprocess(features, labels)
    print(features.columns.values)
    
    print('Splitting data...')
    X_train, X_val, Y_train, Y_val = train_test_split(features.values, labels.values, train_size=TRAIN_SIZE)
    
    print('training...')
    model = train(X_train, Y_train)
    pred = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    score = roc_auc_score(Y_val, pred)
    print('score = ', score)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        print('OK. Model has been trained with:')
        print('    Number of data:', len(features.index))
        print('Number of features:', len(features.columns))