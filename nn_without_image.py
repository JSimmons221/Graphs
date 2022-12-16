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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, svm

# takes in the size n of an nxn graph and trains four machine learning models.
def train_model(size):
    # Read in CSV
    data = pd.read_csv("Graph.csv")

    # Normalizes the position values of the three points.
    columns_to_normalize = ["sx","sy","mx","my","ex","ey"]
    data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Creates a new dataframe this is to change our 1x40 edge array into 40 distince features
    newData = pd.DataFrame(index=range(len(data)), columns=range(len(data.columns) + 2 * size * (size - 1)))
    newData.rename(columns={0: "adj_mat",
                         1: "sx",
                         2: "sy",
                         3: "mx",
                         4: "my",
                         5: "ex",
                         6: "ey",
                         7: "on_path"}, errors='raise', inplace=True)
    # Goes row by row and turns the 1x40 edge array into 40 distince features
    for i in range(len(data)):
        adjmat = data.iloc[i]['adj_mat']
        adjmat = ast.literal_eval(adjmat)
        adjmat = pd.DataFrame(adjmat).T
        cur_row = data.iloc[[i]]
        new_row = pd.concat([cur_row.reset_index(drop=True), adjmat.reset_index(drop=True)], axis=1)
        newData.iloc[[i]] = new_row

    # Drops the array of edge values
    data = newData
    data = data.drop(["adj_mat"], axis=1)

    # Splits the dataset into train and test datasets with a split of .8/.2 respectively
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    # Drops the response variable from the feature data
    X = train_data.drop(['on_path'], axis=1)
    X_test = test_data.drop(['on_path'], axis=1)
    X = np.asarray(X).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)

    # Drops the feature variables from the response data
    y = train_data[['on_path']]
    y = np.asarray(y).astype(np.float32)
    y_test = test_data[['on_path']]
    y_test = np.asarray(y_test).astype(np.float32)


    # Creates a sequential keras model with two dense layers
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=6 + 2 * size * (size - 1)))
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=100, validation_data=(X_test, y_test))

    # Using knn to classify training data
    knn = KNeighborsClassifier(n_neighbors=40)
    knn.fit(X, y)

    # Testing the knn model against the testing data
    print("Score for knn model with k =40: " + str(knn.score(X_test, y_test)))

    # Using decision tree classifier to classify training data
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X,y)

    # Testing the decisino tree classifier model against the training data
    print("Score for decision tree model: " + str(clf.score(X_test, y_test)))

    # Using support vectorm machine to classify training data
    clf = svm.SVC(kernel='poly')
    clf.fit(X, y)
    print("Score for svm model with poly kernal: " + str(clf.score(X_test, y_test)))



