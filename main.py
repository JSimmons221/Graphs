import networkx as nx
import matplotlib.pyplot as plt

row = [0] * 100
adj = [row] * 100

for i in range(0, 100):
    for j in range(i, 100):
        if j == i + 1 or j == i + 10:
            adj[i][j] = 1
            adj[j][i] = 1
        else:
            adj[i][j] = 0

print(adj[0][1])
for i in adj:
    print(i)
