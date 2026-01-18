import os
import sys
import torch
import pandas as pd
import pickle
from tqdm import tqdm

import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from .utils import normalize_adj, k_hop_subgraph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cf_explanation.cf_explainer import CFExplainer, CFExplainerNew


columns = ['node', 'label', 'prediction', 'cf_prediction',
           'distance', 'cf_mask']


def load_dataset(path, device):
    with open(path, 'rb') as f:
        graphdata = pickle.load(f)

    adj = torch.Tensor(graphdata["adj"]).squeeze()
    features = torch.Tensor(graphdata["feat"]).squeeze()

    labels = torch.tensor(graphdata["labels"]).squeeze()
    idx_train = torch.tensor(graphdata["train_idx"])
    idx_test = torch.tensor(graphdata["test_idx"])
    edge_index, edge_attr = dense_to_sparse(adj)

    norm_adj = normalize_adj(adj)

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        num_features=10,
        num_classes=len(labels.unique()),
        train_set = idx_train,
        test_set = idx_test,
        norm_adj = norm_adj,
        adj = adj
    )

    data.to(device)
    return data


def load_sparse_dense_weights(model, path):
    ''' Load GCN or GCNConvSynthetic with weights from
     original classifier models. '''

    old_state_dict = torch.load(path)
    print(*old_state_dict.keys())
    new_state_dict = {
        'conv1.lin.weight': old_state_dict['gc1.weight'].T,
        'conv2.lin.weight': old_state_dict['gc2.weight'].T,
        'conv3.lin.weight': old_state_dict['gc3.weight'].T,
        'bias1': old_state_dict['gc1.bias'],
        'bias2': old_state_dict['gc2.bias'],
        'bias3': old_state_dict['gc3.bias'],
        'lin.weight': old_state_dict['lin.weight'],
        'lin.bias': old_state_dict['lin.bias']
    }
    model.load_state_dict(new_state_dict)


def explain_original(model, data, lr=.1, n_momentum=0.0, epochs=500,
                     device='cpu', target=None):
    predictions = torch.argmax(model(data.x, data.norm_adj), dim=1)

    nodes = data.test_set if target == None else target

    test_cf_examples = []
    for i in tqdm(nodes):
        sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            int(i),
            4,
            data.edge_index,
            relabel_nodes=True
        )
        sub_index = mapping[0]
        sub_x = data.x[sub_nodes]

        sub_adj = torch.zeros(sub_nodes.shape[0], sub_nodes.shape[0])
        sub_adj[sub_edge_index[0], sub_edge_index[1]] = 1

        explainer = CFExplainer(model, sub_adj, sub_x, 20, 0.0, None,
                                predictions[i], data.num_classes,
                                beta=.5, device=device)

        cf_example = explainer.explain('SGD', i, sub_index, lr=lr,
                                       n_momentum=n_momentum, num_epochs=epochs)

        if not np.isnan(cf_example[1]):
            # convert n x n cf_adj to 1d vector
            cf_mask = (cf_example[-1][sub_edge_index[0], sub_edge_index[1]]).bool()

            subgraph_edge_positions = torch.where(edge_mask)[0]
            full_edge_mask = torch.ones(data.edge_index.shape[1], dtype=bool)

            for sub_idx, full_idx in enumerate(subgraph_edge_positions):
                full_edge_mask[full_idx] = bool(cf_mask[sub_idx])
            cf_example[-1] = full_edge_mask
        else:
            cf_example[-1] = torch.zeros(data.edge_index.shape[1], dtype=bool)
        test_cf_examples.append([i.item(), data.y[i].item(), predictions[i].item()] + cf_example)

    return pd.DataFrame(test_cf_examples, columns=columns)


def explain_new(model, x, edge_index, y, target, cf_model=CFExplainerNew, n_hops=4,
               device='cpu', epochs=500, lr=0.1, n_momentum=0.0, eps=0.0):
    '''
    Explain nodes in target using a cf_model

    Calculates node subgraph
    Converts explanation in subgraph to explanation in full graph
    Adds other required data for analysis of performance

    Returns dataframe that might be used in analysis using evaluate.py
    '''
    predictions = torch.argmax(model(x, edge_index), dim=1)
    explainer = cf_model(model, device, epochs, lr, n_momentum, eps=eps)

    counterfactuals = []
    for i in tqdm(target):
        # build subgraph
        sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            int(i),
            n_hops,
            edge_index,
            relabel_nodes=True
        )
        sub_index = mapping[0]
        sub_x = x[sub_nodes]

        cf_example = explainer(sub_index, sub_x, sub_edge_index)

        if cf_example.sum() > 0:
            # convert mask for subgraph edges to a full graph mask
            cf_distance = float((cf_example.shape[0] - sum(cf_example))) / 2
            cf_out = model(sub_x, sub_edge_index[:, cf_example])[sub_index]
            cf_prediction = int(torch.argmax(cf_out))

            subgraph_edge_positions = torch.where(edge_mask)[0]
            full_edge_mask = torch.ones(edge_index.shape[1], dtype=bool)

            for sub_idx, full_idx in enumerate(subgraph_edge_positions):
                full_edge_mask[full_idx] = bool(cf_example[sub_idx])
        else:
            # set values for missing counterfactual
            cf_distance = float('nan')
            cf_prediction = int(predictions[i])
            full_edge_mask = torch.zeros(edge_index.shape[1], dtype=bool)

        counterfactuals.append([
            int(i),
            int(y[i]),
            int(predictions[i]),
            cf_prediction,
            cf_distance,
            full_edge_mask
        ])

    return pd.DataFrame(counterfactuals, columns=columns)
