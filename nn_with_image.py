import statistics

import cv2 as cv
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics


def train_model(adjmat):
    data = pd.read_csv("Graph.csv")

    adjmat = adjmat.flatten()
    adjmatmat = np.array([[0] * len(adjmat)] * len(data.index))
    for i in range(len(data.index)):
        adjmatmat[i] = adjmat

    data = adjmatmat = pd.concat([data, pd.DataFrame(adjmatmat)], axis=1)
    columns_to_normalize = ["start-x", "start-y", "end-x", "end-y", "query-x", "query-y"]
    data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


    image_arrays = []
    i = 0
    for row in data:
        i += 1
        image = cv.imread(r'Images/graph%i.jpg' % i)
        image_arrays.append(image)
    image_arrays = np.array(image_arrays)

    train_data, test_data,train_images, test_images = train_test_split(data, image_arrays, test_size=0.2, shuffle=True)

    X = train_data.drop(['shortest-path'], axis=1)
    y = train_data[['shortest-path']]

    X_test = test_data.drop(['shortest-path'], axis=1)
    y_test = test_data[["shortest-path"]]

    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_target, epochs=50, validation_data=(test_data, test_target))

    model = keras.Sequential()
    model.add(layers.Dense(10006*1, activation='relu', input_dim=10006))
    model.add(layers.Dense(10006*1, activation='relu'))
    model.add(layers.Dense(10006*1, activation='relu'))
    model.add(layers.Dense(10006 * 1, activation='relu'))
    model.add(layers.Dense(10006 * 1, activation='relu'))
    model.add(layers.Dense(10006*1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=50, validation_data=(X_test, y_test))