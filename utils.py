import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df):
    data_dum = pd.get_dummies(df)
    return data_dum

def fill_NaN_with_mean(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def load_data(file_path: str):
    data = pd.read_csv(file_path)
    labels = data['failure']
    features = data.drop(labels=['failure', 'id'], axis="columns")
    return features, labels
