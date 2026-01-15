'''
compare_model_formats.py

Compare the forward pass, gradient descent and counterfactuals between
classification models from original paper as implemented and different
recreations that allow us to find counterfactuals using the new framework.

GCNConvGCNSynthetic and GCN use (modified) weights from the original models
directly but use PyG's GCNConv for GNN layers rather than torch.mm.
GCN lets GCNConv handle normalisation, and represents the expected use most
closely. GCNConvGCNSynthetic normalises adjacency itself using matrix multiplication.
WrappedOriginalGCN simply acts as an interface that converts between formats
for use in GCNSynthetic.

Results
                     | Forward pass       | Backward pass       | Explanations
---------------------|--------------------|---------------------|--------------
GCNConGCNSyntethetic | within float error | withing float error |  identical
GCN                  | within float error | different           |  different
WrappedOriginalGCN   | identical          | identical           |  identical

'''

import sys
import os

import torch
from compare_gradients import compare_dense_vs_sparse_gradients

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, load_sparse_dense_weights

from gcn import GCNSynthetic
from gcn_sparse import GCN

from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight
from wrapper import GCNConvGCNSynthetic, WrappedOriginalGCN
from utils.test_functions import explain_new, explain_original

to_test = GCNConvGCNSynthetic
# to_test = GCN
# to_test = WrappedOriginalGCN

def main():
    torch.set_default_dtype(torch.float64)

    dense_dense = GCNSynthetic(10, 20, 20, 2, 0.0)
    path_dense = '../../models/gcn_3layer_syn4.pt'
    dense_dense.load_state_dict(torch.load(path_dense))
    dense_dense.eval()

    if to_test == WrappedOriginalGCN:
        sparse_dense = WrappedOriginalGCN(dense_dense)
    else:
        sparse_dense = GCNConvGCNSynthetic(10, 2)
        load_sparse_dense_weights(sparse_dense, path_dense)


    sparse_dense.eval()

    data = load_dataset('../../data/gnn_explainer/syn4.pickle', device='cpu')
    data.norm_adj = data.norm_adj.double()
    data.x = data.x.double()

    index = torch.tensor([400, 500, 600])

    out_dd = dense_dense(data.x, data.norm_adj)
    out_sd = sparse_dense(data.x, data.edge_index)

    # Calculate error from forward call
    print('Original weights')
    print(f'Total err: {(out_dd - out_sd).sum():.2e}')
    print(f'Max err:   {(out_dd - out_sd).max():.2e}')
    print(f'Mean err:  {(out_dd - out_sd).mean():.2e}')

    for i in index:
        cf_dd = GCNSyntheticPerturb(10, 20, 20, 2, data.adj, 0.0, 0.5)
        cf_dd.load_state_dict(dense_dense.state_dict(), strict=False)

        cf_sd = GCNSyntheticPerturbEdgeWeight(sparse_dense, i, data.x, data.edge_index)

        for j in [cf_dd, cf_sd]:
            for name, param in j.named_parameters():
                if 'P_vec' not in name:
                    param.requires_grad_(False)

        print(f'Node {i}')
        compare_dense_vs_sparse_gradients(cf_dd, cf_sd, data.x, data.adj,
                                          data.edge_index, epochs=5, correct_grads=False, verbose=False)
        print('\n')

    index = data.test_set

    df_original = explain_original(dense_dense, data, .1, 0, 500, target=index)
    df_original_gcnconv = explain_new(sparse_dense, data.x, data.edge_index, data.y, target=index, lr=.1)

    df_original_gcnconv.to_pickle(f"../../results/syn4dense_classifier_gcnconv.pkl")

    mask1 = df_original['cf_mask']
    mask2 = df_original_gcnconv['cf_mask']

    all_identical = True
    for i, j in zip(mask1, mask2):
        if any(i ^ j == 1):
            all_identical = False
            print(data.edge_index.T[torch.where(~i)])
            print(data.edge_index.T[torch.where(~j)])

    if all_identical:
        print('\033[92m' + 'SUCCESS: All counterfactuals are identical!')


if __name__ == '__main__':
    main()
