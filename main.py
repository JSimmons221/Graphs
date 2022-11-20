import networkx as nx
import matplotlib.pyplot as plt

adj = []
for i in range(0,100):
    adj.append([0] * 100)

for i in range(0, 100):
    for j in range(i, 100):
        if j == i + 1 or j == i + 10:
            adj[i][j] = 1
            adj[j][i] = 1

print(adj[0][1])
for i in adj:
    print(i)
