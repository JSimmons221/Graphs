import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nn_without_image as nnwi
import nn_with_image as nni
import graphs as gp
import more_graphs as mg

n = 5
mg.make_data(50000, r'Graph.csv', n)
# nni.train_model(n)
nnwi.train_model(n)

