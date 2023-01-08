import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
