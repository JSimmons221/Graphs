import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand

rand.seed()

def graph(adj_mat, positions, nodes):
    plt.figure(3,figsize=(1,1), dpi=70)
    G = nx.from_numpy_array(adj_mat)
    colors = color_list(nodes)
    nx.draw(G, with_labels=False, font_weight='bold', node_size=5, pos=positions, node_color=colors)
    plt.show()


def rand_nodes():
    ret = []
    r1 = rand.randint(0, 99)
    ret.append(r1)

    r2 = rand.randint(0, 99)
    while(r2 == r1):
        r2 = rand.randint(0, 99)
    ret.append(r2)

    r3 = rand.randint(0, 99)
    while(r3 == r1 or r3 == r1):
        r3 = rand.randint(0, 99)
    ret.append(r3)

    return ret


def color_list(nodes):
    colors = []
    for i in range(0, 100):
        if i == nodes[0] or i == nodes[1]:
            colors.append('green')
        elif i == nodes[2]:
            colors.append('red')
        else:
            colors.append('blue')

    print(len(colors))

    return colors
