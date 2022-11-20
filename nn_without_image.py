import statistics

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
    adjmatmat = [np.empty(len(data.index))]
    for i in range(len(data.index)):
        adjmatmat[i] = adjmat

    data['adjmat'] = adjmatmat
    train_data, test_data = train_test_split(data, test_size=0.5, shuffle=True)

    X = train_data[['adjmat','start-x', 'start-y', 'end-x', 'end-y', 'query-x', 'query-y']]
    y = train_data[['shortest-path']]

    X_test = test_data[['adjmat','start-x', 'start-y', 'end-x', 'end-y', 'query-x', 'query-y']]
    y_test = test_data[["shortest-path"]]



    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    p_pred = knn.predict(X_test)
    y_pred = np.where(p_pred > 0.5, 1, 0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                display_labels=['Below median crime', 'Above median crime'])

    cm_display.plot()
    plt.show()
