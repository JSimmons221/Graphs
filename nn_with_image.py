import statistics

import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def train_model(adjmat):
    data = pd.read_csv("Graph.csv")

    adjmat = adjmat.flatten()
    adjmatmat = np.array([[0] * len(adjmat)] * len(data.index))
    for i in range(len(data.index)):
        adjmatmat[i] = adjmat

    data = adjmatmat = pd.concat([data, pd.DataFrame(adjmatmat)], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    X = train_data.drop(['shortest-path'], axis=1)
    y = train_data[['shortest-path']]

    X_test = test_data.drop(['shortest-path'], axis=1)
    y_test = test_data[["shortest-path"]]
    error_rate = []
    for i in range(1,50):
        print(i)
        knn = KNeighborsClassifier(n_neighbors=i*4)
        knn.fit(X, y)
        y_pred = knn.predict(X_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        error_rate.append(np.mean(y_pred != y_test))
    print(error_rate)