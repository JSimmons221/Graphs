import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nn_without_image as nnwi
import nn_with_image as nni
import graphs as gp


n = 5
nnwi.train_model(n)
nni.train_model(n)

