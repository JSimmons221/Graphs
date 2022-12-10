import ast
import csv
import glob
import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random as rand


def remove_edge(g):
    edges = list(g.edges)
    edge = rand.choice(edges)
    g.remove_edge(edge[0], edge[1])
    while not nx.is_connected(g):
        g.add_edge(edge[0], edge[1])
        edge = rand.choice(edges)
        g.remove_edge(edge[0], edge[1])

    return g


def remove_edges(g):
    for i in range(0, 15):
        g = remove_edge(g)

    return g


def make_graph(size):
    g = nx.Graph(get_adj(size))
    start_list = list(g.edges)
    remove_edges(g)
    edge_list = list(g.edges)

    edge_array = []
    for edge in start_list:
        if edge in edge_list:
            edge_array.append(1)
        else:
            edge_array.append(0)

    return g, edge_array


def get_adj(size):
    adj_mat = []
    for i in range(0, size ** 2):
        adj_mat.append([0] * size ** 2)

    for i in range(0, size ** 2):
        for j in range(i, size ** 2):
            if j == i + size:
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
            if j == i + 1 and int(i / size) == int(j / size):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

    return np.array(adj_mat)


def view_adj(size):
    adj = get_adj(size)
    print(size)
    for i in adj:
        print(i)
    print()


def get_positions(size):
    pos = {0: [0, 0]}
    for i in range(1, size ** 2):
        node_i = i % size
        node_j = int((i - i % size)/size)
        pos[i] = [node_i, node_j]

    return pos


def base_colors(size):
    colors = []
    for i in range(0, size ** 2):
        colors.append('blue')

    return colors


def shortest_paths(g, source, end):
    pred = nx.predecessor(g, source)
    paths_list = list(nx.algorithms.shortest_paths.generic._build_paths_from_predecessors([source], end, pred))
    return paths_list


def shortest_path_nodes(g, source, end):
    nodes = []
    paths = shortest_paths(g, source, end)
    for i in paths:
        nodes.extend(i[1:len(i)-1])

    return list(set(nodes))


# on_path is for testing purposes
def node_selection(g, source, end, on_path=-1):
    if on_path == -1:
        on_path = rand.randint(0, 1)
    sp_nodes = shortest_path_nodes(g, source, end)
    if on_path:
        node = rand.choice(sp_nodes)
    else:
        nodes = list(g.nodes)
        not_sp_nodes = list(set(nodes) - set(sp_nodes))
        node = rand.choice(not_sp_nodes)

    return node, on_path


def rand_nodes(g, size):
    start = rand.randint(0, size ** 2 - 1)

    end = rand.randint(0, size ** 2 - 1)
    while end == start or nx.shortest_path_length(g, start, end) == 1:
        end = rand.randint(0, size ** 2 - 1)

    mid, on_path = node_selection(g, start, end)

    return start, mid, end, on_path


def draw_graph_no_save(g, size, colors):
    positions = get_positions(size)
    plt.figure(3, figsize=(1, 1), dpi=70)
    nx.draw(g, with_labels=True, node_size=10, pos=positions, node_color=colors)
    plt.show()
    plt.clf()


def node_to_coords(node, size):
    node_x = node % size
    node_y = int((node - node % size)/size)
    return node_x, node_y


def coords_to_node(node_x, node_y, size):
    return node_y * size + node_x


def make_data(n, path, size):
    rand.seed()
    with open(path, 'w+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the labels for the columns
        writer.writerow(['adj_mat', 'sx', 'sy', 'mx', 'my', 'ex', 'ey', 'on_path'])
        # make data and add to csv
        for i in range(n):
            g, e_array = make_graph(size)
            start, mid, end, on_path = rand_nodes(g, size)
            sx, sy = node_to_coords(start, size)
            mx, my = node_to_coords(mid, size)
            ex, ey = node_to_coords(end, size)

            row = [e_array, sx, sy, mx, my, ex, ey, on_path]
            writer.writerow(row)


def clear_images():
    path = r'Resources/Images/*.jpg'
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def list_nodes_to_graph(edge_list, all_edges):
    ret_edges = []
    for i in range(0, len(edge_list)):
        if edge_list[i]:
            ret_edges.append(all_edges[i])

    G = nx.Graph(ret_edges)

    return G

def change_colors(colors, nodes, size):
    colors[coords_to_node(nodes[0], nodes[1], size)] = 'green'
    colors[coords_to_node(nodes[2], nodes[3], size)] = 'red'
    colors[coords_to_node(nodes[4], nodes[5], size)] = 'green'
    return colors


def undo_colors(colors, nodes, size):
    colors[nodes[0] + nodes[1] * size] = 'blue'
    colors[nodes[2] + nodes[3] * size] = 'blue'
    colors[nodes[4] + nodes[5] * size] = 'blue'
    return colors


def graph(G, positions, colors, path):
    nx.draw(G, with_labels=False, font_weight='bold', node_size=10, pos=positions, node_color=colors)
    plt.savefig(path)
    plt.clf()


def csv_to_images(path, size):
    colors = base_colors(size)
    g = (nx.Graph(get_adj(size)))
    positions = get_positions(5)
    all_edges = list(g.edges)
    plt.figure(3, figsize=(1, 1), dpi=50)

    with open(path) as file:
        clear_images()
        reader = csv.reader(file, delimiter=',')
        c = 0
        for row in reader:
            if c != 0:
                if c % 100 == 0:
                    print(c)

                edge_list = list(ast.literal_eval(row[0]))
                g = list_nodes_to_graph(edge_list, all_edges)
                H = nx.Graph()
                H.add_nodes_from(sorted(g.nodes(data=True)))
                H.add_edges_from(g.edges(data=True))

                nodes = []
                for i in range(1, 7):
                    nodes.append(int(row[i]))
                colors = change_colors(colors, nodes, size)

                graph(H, positions, colors, r'Resources/images/graph%i.jpg' % c)
                colors = undo_colors(colors, nodes, size)
            c += 1


make_data(10000, r'Graph.csv', 5)
csv_to_images(r'Graph.csv', 5)
