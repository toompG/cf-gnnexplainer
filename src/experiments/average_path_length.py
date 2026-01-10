
import torch
import torch.nn.functional as F

import networkx as nx
from torch_geometric.utils import to_networkx


from itertools import combinations
from torch_geometric.data import Data

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset

def average_path_length(data):
    data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    G = to_networkx(data, to_undirected=True)

    return nx.average_shortest_path_length(G)


def main():
    datasets = {
        'BA Shapes': load_dataset('../../data/gnn_explainer/syn1.pickle', device='cpu'),
        'Tree Cycles': load_dataset('../../data/gnn_explainer/syn4.pickle', device='cpu'),
        'Tree Grid': load_dataset('../../data/gnn_explainer/syn5.pickle', device='cpu')
    }

    for i, j in datasets.items():
        print(f'{i}: average path length={average_path_length(j):.2f}')


if __name__ == '__main__':
    main()