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
from sklearn.metrics import accuracy_score

TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'
MODEL_SAVE_PATH = 'model.pk'
TRAIN_SIZE = 0.8

def preprocess():
    features, labels = utils.load_data(TRAIN_DATA_PATH, label='failure')
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
    adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10, max_depth=4),n_estimators=10,learning_rate=0.6)
    adb.fit(X, Y)
    return adb

if __name__ == '__main__':
    print('Preprocessing...')
    features, labels = preprocess()
    
    print('Splitting data...')
    X_train, X_val, Y_train, Y_val = train_test_split(features.values, labels.values, train_size=TRAIN_SIZE)
    
    print('training...')
    model = train(X_train, Y_train)
    acc = model.score(X_val, Y_val)
    print('acc = ', acc)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        print('OK. Model has been trained with:')
        print('    Number of data:', len(features.index))
        print('Number of features:', len(features.columns))