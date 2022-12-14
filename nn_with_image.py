import statistics

import cv2 as cv
import math
import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

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

    image_arrays = []
    i = 0
    for row in data.iterrows():
        i += 1
        image = cv.imread(r'Resources/images/graph%i.jpg' % i)
        image_arrays.append(image)
    image_arrays = np.array(image_arrays)
    image_arrays = image_arrays / 255.0

    train_data, test_data, train_images, test_images = train_test_split(data, image_arrays, test_size=0.2, shuffle=True)

    X = train_data.drop(['on_path'], axis=1)
    y = train_data[['on_path']]
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X_test = test_data.drop(['on_path'], axis=1)
    y_test = test_data[['on_path']]
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    image_input = layers.Input(shape=(50, 50, 3), name='image_input')
    non_image_input = layers.Input(shape=(6 + 2 * size * (size - 1)), name='non_image_input')

    pretrained_model = ResNet50(include_top=False,
                                input_tensor=image_input,
                                pooling='avg',
                                weights='imagenet')

    non_image_layer = layers.Dense(32, activation='relu')(non_image_input)

    merge_layer = layers.concatenate([pretrained_model.output, non_image_layer])
    merge_layer = layers.Dense(2, activation='softmax')(merge_layer)
    merged_model = Model(inputs=[image_input, non_image_input], outputs=merge_layer)

    merged_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    merged_model.fit([train_images, X], y, epochs=50, validation_data=([test_images, X_test], y_test))

