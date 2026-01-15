# import torch
import numpy as np
from pathlib import Path

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcn import GCNSynthetic
from wrapper import WrappedOriginalGCN
from utils.utils import create_symm_matrix_from_vec
from utils.test_functions import load_dataset

from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight


floattype = torch.float32
torch.set_default_dtype(floattype)

def remove_bidirectional_edges(edge_index):
    edges = edge_index.numpy().T

    sorted_edges = np.sort(edges, axis=1)
    _, indices = np.unique(sorted_edges, axis=0, return_index=True)

    result = edges[indices]

    return torch.tensor(result).t().contiguous()


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

        dense_grads = extract_grads_for_sparse_edges(dense_model.P_vec.grad, P_edge, dense_model.num_nodes)
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


def main():
    exp = 'syn4'
    idx = [49, 600]

    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../../data/gnn_explainer/{exp}.pickle'
    model_path = script_dir / f'../../models/gcn_3layer_{exp}.pt'

    device = 'cpu'

    data = load_dataset(graph_path, device)
    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
    submodel.load_state_dict(torch.load(model_path))

    if floattype == torch.float64:
        data.x = data.x.double()
        data.adj = data.adj.double()
        submodel.double()

    submodel.eval()
    model = WrappedOriginalGCN(submodel)
    model.eval()

    edge_index = data.edge_index

    x = data.x
    adj_dense = data.adj

    for index in idx:

        y_orig = torch.argmax(model(x, edge_index), dim=1)[index]
        dense_cf_model = GCNSyntheticPerturb(
            10, 20, 20, 2, data.adj, 0, .5
        )
        dense_cf_model.load_state_dict(submodel.state_dict(), strict=False)
        sparse_cf_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index)

        # 1. First, check if forward passes match
        print("Step 1: Checking forward pass equivalence...")
        fwd_dense = dense_cf_model.forward(x, data.adj)[index]
        bwd_sparse = sparse_cf_model.forward()
        assert (fwd_dense - bwd_sparse).sum() == 0.0

        # 2. Check if loss computation matches
        print("\nStep 2: Checking loss computation...")
        out_dense = dense_cf_model.forward(x, adj_dense)
        _ = dense_cf_model.forward_prediction(x)
        out_sparse = sparse_cf_model.forward()
        _ = sparse_cf_model.forward_hard()

        loss_dense = dense_cf_model.loss(out_dense[index], y_orig, y_orig)[0]
        loss_sparse = sparse_cf_model.loss(out_sparse, y_orig)[0]

        assert (loss_dense.item() - loss_sparse.item()) == 0.0
        compare_dense_vs_sparse_gradients(dense_cf_model, sparse_cf_model, x, adj_dense, edge_index)

if __name__ == '__main__':
    main()