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

Results (double precision)

                     | Forward pass       | Backward pass       | Explanations
---------------------|--------------------|---------------------|--------------
GCNConGCNSyntethetic | within float error | withing float error |  identical
GCN                  | within float error | different           |  different
WrappedOriginalGCN   | identical          | identical           |  identical

'''

import sys
import os

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, load_sparse_dense_weights

from gcn import GCNSynthetic
from gcn_sparse import GCN

from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight
from wrapper import GCNConvGCNSynthetic, WrappedOriginalGCN
from utils.test_functions import explain_new, explain_original
from utils.utils import create_symm_matrix_from_vec

to_test = GCNConvGCNSynthetic
# to_test = GCN
# to_test = WrappedOriginalGCN


def extract_grads_for_sparse_edges(dense_grad_vec, edge_index, num_nodes):
    dense_grad_matrix = create_symm_matrix_from_vec(dense_grad_vec, num_nodes)
    sparse_edge_grads = dense_grad_matrix[edge_index[0], edge_index[1]]

    return sparse_edge_grads


def compare_dense_vs_sparse_gradients(dense_model, sparse_model, x, adj_dense,
                                      edge_index, epochs=100, correct_grads=False, verbose=True):
    """
    Compare gradients between dense and sparse implementations
    """
    # get list of unique edges, used for matching individual gradients
    P_edge = edge_index[:, sparse_model.matched_edges[0]]
    y_pred_orig = sparse_model.original_class
    idx = sparse_model.index


    # Run wrapped model in sparse explainer and original as implemented side by side
    for i in range(epochs):
        out_dense = dense_model.forward(x, adj_dense)
        out_sparse = sparse_model.forward()

        y_dense, _ = dense_model.forward_prediction(x)
        y_sparse = sparse_model.forward_hard()

        y_dense = torch.argmax(y_dense, dim=1)[idx]
        y_sparse = torch.argmax(y_sparse)

        loss_dense, _, _, _ = dense_model.loss(out_dense[idx], y_pred_orig, y_dense)
        loss_sparse, _, _, _ = sparse_model.loss(out_sparse, y_sparse)

        print(f"difference loss: {abs(loss_dense.item() - loss_sparse.item()):.2e}")

        # Backward
        dense_model.zero_grad()
        sparse_model.zero_grad()

        loss_dense.backward()
        loss_sparse.backward()

        # Compare P_vec gradients (dense) vs edge_weight_params gradients (sparse)
        if dense_model.P_vec.grad is None or sparse_model.P_vec.grad is None:
            print("missing gradients")
            return

        dense_grads = extract_grads_for_sparse_edges(dense_model.P_vec.grad,
                                                     P_edge,
                                                     dense_model.num_nodes)
        sparse_grads = sparse_model.P_vec.grad

        if verbose:
            for E, i, j in zip(P_edge.T, dense_grads, sparse_grads):
                if i-j != 0:
                    print(E, i, j, i-j)

        print(f"mean difference grad: {sum(abs(dense_grads - sparse_grads)) / dense_grads.shape[0]}")
        print(f"max difference grad:  {(dense_grads - sparse_grads).max()}")

        # print(*sparse_model.named_parameters())
        # update params for next epoch
        with torch.no_grad():
            lr = 0.1
            if dense_model.P_vec.grad is not None:
                dense_model.P_vec -= lr * dense_model.P_vec.grad
            if sparse_model.P_vec.grad is not None:
                sparse_model.P_vec -= lr * (dense_grads if correct_grads else sparse_model.P_vec.grad)


def jaccard_similarity(mask1, mask2):
    removed1 = set(torch.where(~mask1)[0].tolist())
    removed2 = set(torch.where(~mask2)[0].tolist())

    intersection = len(removed1 & removed2)
    union = len(removed1 | removed2)

    return intersection / union if union > 0 else 1.0


def main():
    torch.set_default_dtype(torch.float64)

    dense_dense = GCNSynthetic(10, 20, 20, 2, 0.0)
    path_dense = '../../models/gcn_3layer_syn4.pt'
    dense_dense.load_state_dict(torch.load(path_dense))
    dense_dense.eval()

    if to_test == WrappedOriginalGCN:
        sparse_dense = WrappedOriginalGCN(dense_dense)
    else:
        sparse_dense = to_test(10, 2)
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

    scores = []
    for i, j in zip(mask1, mask2):
        if any(i ^ j == 1):
            print(data.edge_index.T[torch.where(~i)])
            print(data.edge_index.T[torch.where(~j)])

        scores.append(jaccard_similarity(i, j))

    similarity = sum(scores) / len(scores)
    if similarity == 1:
        print('\033[92m' + 'SUCCESS: All counterfactuals are identical!')
    else:
        print(f'Jaccard similarity: {similarity:.3f}')


if __name__ == '__main__':
    main()
