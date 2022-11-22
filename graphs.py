import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand


def coord(n):
    return int((n - n % 10)/10), n % 10


def node(ni, nj):
    return ni * 10 + nj


def dist(n1, n2):
    n1i, n1j = coord(n1)
    n2i, n2j = coord(n2)
    return abs(n2i - n1i) + abs(n2j - n2j)


def graph(adj_mat, positions, nodes, path):
    plt.figure(3,figsize=(1,1), dpi=70)
    G = nx.from_numpy_array(adj_mat)
    colors = color_list(nodes)
    nx.draw(G, with_labels=False, font_weight='bold', node_size=5, pos=positions, node_color=colors)



def third_node(node_1, node_2, r):
    node_3 = 0
    node_1i, node_1j = coord(node_1)
    node_2i, node_2j = coord(node_2)
    i_min = min(node_1i, node_2i)
    i_max = max(node_1i, node_2i)
    j_min = min(node_1j, node_2j)
    j_max = max(node_1j, node_2j)

    nodes = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            nodes.append(node(i, j))

    if r:
        nodes.remove(node_1)
        nodes.remove(node_2)
        node_3 = rand.choice(nodes)
    else:
        not_nodes = []
        for i in range(0, 100):
            if i not in nodes:
                not_nodes.append(i)
        node_3 = rand.choice(not_nodes)

    return node_3


def rand_nodes():
    node_1 = rand.randint(1, 98)
    while node_1 == 9 or node_1 == 90:
        node_1 = rand.randint(1, 98)


    node_2 = rand.randint(1, 98)
    while node_2 == node_1 or dist(node_1, node_2) < 2 or node_2 == 9 or node_2 == 90:
        node_2 = rand.randint(1, 98)

    r = rand.randint(0, 1)

    node_3 = third_node(node_1, node_2, r)

    n1i, n1j = coord(node_1)
    n2i, n2j = coord(node_2)
    n3i, n3j = coord(node_3)
    return [n1i, n1j, n2i, n2j, n3i, n3j, r]


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


def make_data(n, path):
    # sets the seed for rand
    rand.seed()
    # open the csv
    with open(path, 'w+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the labels for the columns
        writer.writerow(['Polygon 1', 'Polygon 2', 'Congruent'])
        # make data and add to csv
        c = 0
        for i in range(n):
            print(c)
            writer.writerow(rand_nodes())
            c += 1


# def make_images(path, adj_mat, positions, nodes):
