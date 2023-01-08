import pandas as pd
import numpy as np
import sklearn
import utils
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = 'train.csv'
TRAIN_SIZE = 0.8

def preprocess():
    features, labels = utils.load_data(TRAIN_DATA_PATH)
    features = utils.one_hot_encode(features)
    features = utils.fill_NaN_with_mean(features)
    labels = utils.fill_NaN_with_mean(labels)

    return features, labels

if __name__ == '__main__':
    features, labels = preprocess()
    
    X_train, X_val, Y_train, Y_val = train_test_split(features, labels, train_size=TRAIN_SIZE)
    print(X_val)