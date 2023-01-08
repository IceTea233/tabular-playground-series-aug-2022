import pandas as pd
import numpy as np
import utils

TRAIN_DATA_PATH = 'train.csv'

def preprocess():
    labels, features = utils.load_data(TRAIN_DATA_PATH)
    features = utils.one_hot_encode(features)
    features = utils.fill_NaN_with_mean(features)
    labels = utils.fill_NaN_with_mean(labels)

    return features, labels

if __name__ == '__main__':
    features, label = preprocess()