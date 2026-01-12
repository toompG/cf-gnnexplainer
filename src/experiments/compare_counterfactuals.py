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
from utils.test_functions import load_dataset, explain_original, explain_new

floattype = torch.float32
torch.set_default_dtype(floattype)


def main():
    exp = 'syn4'

    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../../data/gnn_explainer/{exp}.pickle'
    model_path = script_dir / f'../../models/gcn_3layer_{exp}.pt'

    device = 'cpu'

    data = load_dataset(graph_path, device)
    idx = data.test_set
    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
    submodel.load_state_dict(torch.load(model_path))

    submodel.eval()
    model = WrappedOriginalGCN(submodel)
    model.eval()

    dense = explain_original(submodel, data, target=idx, epochs=500)
    sparse = explain_new(model, data.x, data.edge_index, idx, data.y, epochs=500)

    for i, j in zip(dense['cf_mask'], sparse['cf_mask']):
        assert all(i ^ j == False)
    print('Success: All counterfactuals were identical!')

if __name__ == '__main__':
    main()