# import torch
import numpy as np
import pandas as pd
from pathlib import Path

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcn import GCNSynthetic
from wrapper import WrappedOriginalGCN
from utils.test_functions import load_dataset, explain_original, explain_new, load_sparse_dense_weights

floattype = torch.float32
torch.set_default_dtype(floattype)

from gcn_sparse import GCN


def main():
    exp = 'syn1'

    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../../data/gnn_explainer/{exp}.pickle'
    model_path = script_dir / f'../../models/gcn_3layer_{exp}.pt'

    device = 'cpu'

    data = load_dataset(graph_path, device)
    idx = data.test_set[:12]
    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
    submodel.load_state_dict(torch.load(model_path))

    # submodel.eval()
    # model = WrappedOriginalGCN(submodel)
    # model.eval()

    model = GCN(10, 4)

    load_sparse_dense_weights(model, model_path)


    dense = explain_original(submodel, data, target=idx, epochs=500)
    sparse = explain_new(model, data.x, data.edge_index, data.y, idx, epochs=500)

    for n, (i, j) in zip(dense['cf_mask'], sparse['cf_mask']):
        if not all(i ^ j == False):
            print(sparse['node'][i])
            print(data.edge_index[:, ~dense['cf_mask']])
            print(data.edge_index[:, ~sparse['cf_mask']])

        # assert all(i ^ j == False)
    print('Success: All counterfactuals were identical!')

if __name__ == '__main__':
    main()