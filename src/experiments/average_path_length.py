"""
average_path_length.py

Script used to populate table about the datasets used in experiments.
"""

import networkx as nx
from torch_geometric.utils import to_networkx

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset


def average_path_length(data):
    data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    G = to_networkx(data, to_undirected=True)

    # print(nx.degree(G))
    return nx.average_shortest_path_length(G)


def dataset_stats(data):
    graph_data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    G = to_networkx(graph_data, to_undirected=True)

    # print(nx.degree(G))
    print(f'Classes: {data.num_classes}')
    print(f'Nodes in total: {data.x.shape[0]}')
    print(f'Edges in total: {data.edge_index.shape[1]}')

    subgraph_nodes = []
    subgraph_edges = []
    degrees = []
    for i in range(data.x.shape[0]):
        nodes, sub_edge, _, _ = k_hop_subgraph(int(i), 4, data.edge_index)
        # print(nodes.shape)

        subgraph_nodes.append(int(nodes.shape[0]))
        subgraph_edges.append(int(sub_edge.shape[1]))
        degrees.append(int(k_hop_subgraph(int(i), 1, data.edge_index)[0].shape[0]))

    print(f'Avg node degree: {sum(i[1] for i in nx.degree(G)) / len(G)}')
    print(f'Avg n nodes in A_v: {sum(subgraph_nodes) / len(subgraph_nodes)}')
    print(f'Avg edges in A_v: {sum(subgraph_edges) / len(subgraph_edges) / 2}')
    print('')

def main():
    datasets = {
        'BA Shapes': load_dataset('../../data/gnn_explainer/syn1.pickle', device='cpu'),
        'BA Community': load_dataset('../../data/gnn_explainer/syn2.pickle', device='cpu'),
        'Tree Cycles': load_dataset('../../data/gnn_explainer/syn4.pickle', device='cpu'),
        'Tree Grid': load_dataset('../../data/gnn_explainer/syn5.pickle', device='cpu')
    }

    for i, j in datasets.items():
        print(f'{i}: average path length={average_path_length(j):.2f}')
        dataset_stats(j)


if __name__ == '__main__':
    main()