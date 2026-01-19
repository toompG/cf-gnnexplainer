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
import argparse

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, load_sparse_dense_weights

from gcn import GCNSynthetic
from gcn_sparse import GCN

from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight
from wrapper import GCNConvGCNSynthetic, WrappedOriginalGCN
from utils.test_functions import explain_new, explain_original
from utils.utils import create_symm_matrix_from_vec


def extract_grads_for_sparse_edges(dense_grad_vec, edge_index, num_nodes):
    dense_grad_matrix = create_symm_matrix_from_vec(dense_grad_vec, num_nodes)
    sparse_edge_grads = dense_grad_matrix[edge_index[0], edge_index[1]]

    return sparse_edge_grads



def compare_dense_vs_sparse_gradients(dense_model, sparse_model, x, adj_dense,
                                      edge_index, epochs=100, correct_grads=False, verbose=True):
    """
    Compare gradients between dense and sparse implementations

    Returns:
        tuple: (mean_error, max_error) after final epoch
    """
    # get list of unique edges, used for matching individual gradients
    P_edge = edge_index[:, sparse_model.matched_edges[0]]
    y_pred_orig = sparse_model.original_class
    idx = sparse_model.index

    mean_error = 0.0
    max_error = 0.0

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

        if verbose:
            print(f"difference loss: {abs(loss_dense.item() - loss_sparse.item()):.2e}")

        # Backward
        dense_model.zero_grad()
        sparse_model.zero_grad()

        loss_dense.backward()
        loss_sparse.backward()

        # Compare P_vec gradients (dense) vs edge_weight_params gradients (sparse)
        if dense_model.P_vec.grad is None or sparse_model.P_vec.grad is None:
            print("missing gradients")
            return None, None

        dense_grads = extract_grads_for_sparse_edges(dense_model.P_vec.grad,
                                                     P_edge,
                                                     dense_model.num_nodes)
        sparse_grads = sparse_model.P_vec.grad

        if verbose:
            for E, i, j in zip(P_edge.T, dense_grads, sparse_grads):
                if i-j != 0:
                    print(E, i, j, i-j)

        mean_error = (sum(abs(dense_grads - sparse_grads)) / dense_grads.shape[0]).item()
        max_error = (dense_grads - sparse_grads).max().item()

        if verbose:
            print(f"mean difference grad: {mean_error}")
            print(f"max difference grad:  {max_error}")

        # print(*sparse_model.named_parameters())
        # update params for next epoch
        with torch.no_grad():
            lr = 0.1
            if dense_model.P_vec.grad is not None:
                dense_model.P_vec -= lr * dense_model.P_vec.grad
            if sparse_model.P_vec.grad is not None:
                sparse_model.P_vec -= lr * (dense_grads if correct_grads else sparse_model.P_vec.grad)

    return mean_error, max_error


def jaccard_similarity(mask1, mask2):
    removed1 = set(torch.where(~mask1)[0].tolist())
    removed2 = set(torch.where(~mask2)[0].tolist())

    intersection = len(removed1 & removed2)
    union = len(removed1 | removed2)

    return intersection / union if union > 0 else 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='Compare GCN model formats')

    parser.add_argument('--exp', type=str, required=True,
                        choices=['syn1', 'syn2', 'syn4', 'syn5'],
                        help='Dataset to use (syn1, syn2, syn4, or syn5)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['wrapped', 'gcn', 'gcnmm'],
                        help='Interfacing method to use')
    parser.add_argument('--verbose', type=bool, default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    method_map = {
        'gcnmm': GCNConvGCNSynthetic,
        'gcn': GCN,
        'wrapped': WrappedOriginalGCN
    }

    to_test = method_map[args.method]

    # Dataset configuration
    data_path = f'../../data/gnn_explainer/{args.exp}.pickle'
    model_path = f'../../models/gcn_3layer_{args.exp}.pt'

    print(f"Using: {args.exp}, {args.method}")

    torch.set_default_dtype(torch.float64)

    data = load_dataset(data_path, device='cpu')
    data.norm_adj = data.norm_adj.double()
    data.x = data.x.double()

    dense_dense = GCNSynthetic(10, 20, 20, data.num_classes, 0.0)
    dense_dense.load_state_dict(torch.load(model_path))
    dense_dense.eval()

    if to_test == WrappedOriginalGCN:
        sparse_dense = WrappedOriginalGCN(dense_dense)
    else:
        sparse_dense = to_test(10, data.num_classes)
        load_sparse_dense_weights(sparse_dense, model_path)

    sparse_dense.eval()

    # index = torch.tensor([400, 500, 600])
    index = data.test_set

    out_dd = dense_dense(data.x, data.norm_adj)
    out_sd = sparse_dense(data.x, data.edge_index)

    # Calculate error from forward call
    print('Original weights')
    print(f'Total err: {(out_dd - out_sd).sum():.2e}')
    print(f'Forward max err:   {(out_dd - out_sd).max():.2e}')
    print(f'Forward mean err:  {(out_dd - out_sd).mean():.2e}')

    mean_errors = []
    max_errors = []

    for i in tqdm(index):
        cf_dd = GCNSyntheticPerturb(10, 20, 20, data.num_classes, data.adj, 0.0, 0.5)
        cf_dd.load_state_dict(dense_dense.state_dict(), strict=False)

        cf_sd = GCNSyntheticPerturbEdgeWeight(sparse_dense, i, data.x, data.edge_index)

        for j in [cf_dd, cf_sd]:
            for name, param in j.named_parameters():
                if 'P_vec' not in name:
                    param.requires_grad_(False)

        mean_err, max_err = compare_dense_vs_sparse_gradients(cf_dd, cf_sd, data.x, data.adj,
                                          data.edge_index, epochs=1, correct_grads=False, verbose=args.verbose)

        if mean_err is not None and max_err is not None:
            mean_errors.append(mean_err)
            max_errors.append(max_err)

    print(f'Gradient mean err: {sum(mean_errors) / len(mean_errors):.2e}')
    print(f'Gradient max err: {sum(max_errors) / len(max_errors):.2e}')

    index = data.test_set

    momentum = .9 if args.exp == 'syn1' else 0.0
    df_original = explain_original(dense_dense, data, .1, 0, 500, target=index, n_momentum=momentum)
    df_original_gcnconv = explain_new(sparse_dense, data.x, data.edge_index, data.y, target=index, lr=.1)

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