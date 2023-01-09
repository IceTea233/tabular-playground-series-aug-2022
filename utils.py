import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
    print(columns)
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

def plot_matrix(acc_grid, gamma_list, c_list):
    fig, ax = plt.subplots()

    ax.set_xlabel('Gamma parameter')
    ax.set_ylabel('C parameter')
    ax.set_xticks(ticks=np.arange(gamma_list.size), labels=gamma_list, rotation=45)
    ax.set_yticks(ticks=np.flip(np.arange(c_list.size)), labels=np.flip(c_list))

    for i in range(c_list.size):
        for j in range(gamma_list.size):
            ax.text(i, j, round(acc_grid[j][i], 3),
                color='white' if acc_grid[j][i] < 0.5 else 'black', 
                ha='center', va='center')

    im = ax.imshow(acc_grid, origin='lower', cmap='cividis', interpolation='none')

    fig.colorbar(im, orientation='vertical')
    plt.show()
