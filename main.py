import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nn_without_image as nnwi
import nn_with_image as nni
import graphs as gp

adj = []
for i in range(0,100):
    adj.append([0] * 100)

for i in range(0, 100):
    for j in range(i, 100):
        if j == i + 10:
            adj[i][j] = 1
            adj[j][i] = 1
        if j == i + 1 and int(i/10) == int(j/10):
            adj[i][j] = 1
            adj[j][i] = 1

adj_mat = np.array(adj)

positions = {0: [0, 0]}
for i in range(1, 100):
    positions[i] = [i % 10, int((i - i % 10)/10)]

nni.train_model(adj_mat)
nnwi.train_model(adj_mat)

