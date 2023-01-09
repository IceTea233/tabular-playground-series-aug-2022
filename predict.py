import pandas as pd
import numpy as np
import sklearn
import pickle
import utils

TRAIN_DATA_PATH = 'train.csv'
SAMPLE_PATH = 'sample_submission.csv'
MODEL_SAVE_PATH = 'model.pk'
OUTPUT_PATH = 'submission.csv'
TEST_DATA_PATH = 'test.csv'

def load_model(path):
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def inference(model):
    submission_df = pd.read_csv(SAMPLE_PATH)

    features = utils.load_data(TRAIN_DATA_PATH, label='failure')[0]
    features = utils.one_hot_encode(features)
    features = utils.fill_NaN_with_mean(features)

    test_features = utils.load_data(TEST_DATA_PATH, drop_id=False)[0]
    test_features = utils.one_hot_encode(test_features)
    test_features = utils.fill_NaN_with_mean(test_features)

    columns = ['id']
    columns.extend(utils.get_common_columns(features, test_features))
    test_features = test_features[columns]

    print(test_features.columns.values)
    total_row = len(submission_df.index)
    print('total test cases = ', total_row)
    progress = 0
    for index, row in submission_df.iterrows():
        x = test_features.loc[test_features['id'] == row['id']]
        x = x.drop(labels=['id', 'product_code'], axis='columns').values

        pred = model.predict_proba(x)[:, 1].reshape(-1, 1)
        
        submission_df.at[index, 'failure'] = pred
        if progress <= index / total_row:
            progress += 0.1
            print('%.0f%%...' % (progress * 100))
    submission_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    model = load_model(MODEL_SAVE_PATH)
    
    inference(model)