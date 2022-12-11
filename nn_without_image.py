import statistics

import math
import ast
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics


def train_model(size):
    data = pd.read_csv("Graph.csv")

    columns_to_normalize = ["sx","sy","mx","my","ex","ey"]
    data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    newData = pd.DataFrame(index=range(len(data)), columns=range(len(data.columns) + 2 * size * (size - 1)))
    newData.rename(columns={0: "adj_mat",
                         1: "sx",
                         2: "sy",
                         3: "mx",
                         4: "my",
                         5: "ex",
                         6: "ey",
                         7: "on_path"}, errors='raise', inplace=True)
    for i in range(len(data)):
        adjmat = data.iloc[i]['adj_mat']
        adjmat = ast.literal_eval(adjmat)
        adjmat = pd.DataFrame(adjmat).T
        cur_row = data.iloc[[i]]
        new_row = pd.concat([cur_row.reset_index(drop=True), adjmat.reset_index(drop=True)], axis=1)
        newData.iloc[[i]] = new_row
    data = newData
    data = data.drop(["adj_mat"], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    X = train_data.drop(['on_path'], axis=1)
    y = train_data[['on_path']]
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X_test = test_data.drop(['on_path'], axis=1)
    y_test = test_data[['on_path']]
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    model = keras.Sequential()
    model.add(layers.Dense(256 * 1, activation='relu', input_dim=6 + 2 * size * (size - 1)))
    model.add(layers.Dense(256 * 1, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=50, validation_data=(X_test, y_test))

