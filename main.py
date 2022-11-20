import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nn_without_image as nnwi

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

edges = np.array(adj)
positions = {0: [0, 0]}
for i in range(1, 100):
    positions[i] = [i % 10, int((i - i % 10)/10)]
print(positions)

plt.figure(3,figsize=(1,1), dpi = 70)
G = nx.from_numpy_array(edges)
nx.draw(G, with_labels=False, font_weight='bold', node_size=5, pos=positions)

plt.show()

nnwi.train_model(edges)