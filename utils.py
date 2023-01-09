import numpy as np
import pandas as pd
from pandas import DataFrame as df

def one_hot_encode(df):
    product_code = df['product_code']
    data_dum = pd.get_dummies(df)
    output_df = pd.concat([product_code, data_dum], axis='columns')
    return output_df

def fill_NaN_with_mean(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def get_common_columns(df1, df2):
    columns = set(df1.columns.values) & set(df2.columns.values)
    return sorted(columns)

def load_data(file_path: str, label=None, drop_id=True):
    data = pd.read_csv(file_path)
    labels = data[label] if label else None

    features = data
    if (drop_id):
        features = features.drop(labels=['id'], axis='columns')
    if (label):
        features = features.drop(labels=[label], axis='columns')
    return features, labels
