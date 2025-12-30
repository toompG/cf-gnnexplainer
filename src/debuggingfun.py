# import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from functools import partial

from example import *
from cmp_original import WrappedOriginalGCN
from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_edge_perturb import GCNSyntheticPerturbEdgeWeight

from pathlib import Path
from gcn import GCNSynthetic
from utils.utils import *
from torch_geometric.utils import to_dense_adj
from functools import lru_cache

from cf_explanation.cf_explainer import CFExplainer, CFExplainerOriginal
from cf_explanation.cf_greed import GreedyCFExplainer
from cf_explanation.cf_bruteforce import BFCFExplainer

from torch_geometric.nn.conv.gcn_conv import gcn_norm

import argparse


# Debugging utilities
def compare_models(original_model, edge_weight_model, x, edge_index, adj_dense):
    """
    Compare outputs and gradients between the two model types
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Forward pass comparison
    with torch.no_grad():
        # Original model
        norm_adj_orig = original_model.forward(x, adj_dense)

        # Edge weight model
        edge_weights = torch.sigmoid(edge_weight_model.edge_weight_params)
        norm_adj_edge = edge_weight_model.edge_index2norm_adj(edge_index, edge_weights)

        print("\nNormalized adjacency matrix difference:")
        print(f"Max abs diff: {(norm_adj_orig - norm_adj_edge).abs().max().item():.2e}")
        print(f"Mean abs diff: {(norm_adj_orig - norm_adj_edge).abs().mean().item():.2e}")

    # Gradient comparison
    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON")
    print("=" * 80)

    # Get outputs
    out_orig = original_model.forward(x, adj_dense)
    out_edge = edge_weight_model.forward()

    # Compute simple loss for gradient check
    loss_orig = out_orig[edge_weight_model.index].sum()
    loss_edge = out_edge.sum()

    # Backward
    loss_orig.backward()
    loss_edge.backward()

    print(f"\nLoss values:")
    print(f"Original: {loss_orig.item():.6f}")
    print(f"Edge weight: {loss_edge.item():.6f}")
    print(f"Difference: {abs(loss_orig.item() - loss_edge.item()):.2e}")

    # Check parameter gradients
    print(f"\nEdge weight parameter gradients:")
    print(f"Min: {edge_weight_model.edge_weight_params.grad.min().item():.6f}")
    print(f"Max: {edge_weight_model.edge_weight_params.grad.max().item():.6f}")
    print(f"Mean: {edge_weight_model.edge_weight_params.grad.mean().item():.6f}")
    print(f"Std: {edge_weight_model.edge_weight_params.grad.std().item():.6f}")


import torch
import torch.nn.functional as F

def debug_gradient_flow(model, forward_fn, x, edge_index, steps=3):
    """
    Track gradient flow through multiple backward passes to identify divergence
    """
    print("=" * 80)
    print("GRADIENT FLOW DEBUGGING")
    print("=" * 80)

    for step in range(steps):
        print(f"\n{'=' * 80}")
        print(f"STEP {step + 1}")
        print(f"{'=' * 80}")

        # Forward pass
        output = forward_fn()

        if len(output) > 10:
            output = output[600]

        # Compute loss
        y_pred_new_actual = torch.argmax(output)

        loss_total, loss_pred, loss_graph_dist, cf_adj = model.loss(
            output, y_pred_new_actual, y_pred_new_actual
        )

        print(f"\nLoss components:")
        print(f"  Total: {loss_total.item():.6f}")
        print(f"  Pred: {loss_pred:.6f}")
        print(f"  Graph dist: {loss_graph_dist:.6f}")

        # Check if gradients are enabled
        print(f"\nGradient tracking:")
        print(f"  edge_weight_params.requires_grad: {model.edge_weight_params.requires_grad}")
        print(f"  loss_total.requires_grad: {loss_total.requires_grad}")

        # Backward pass
        model.zero_grad()
        loss_total.backward()

        # Analyze gradients
        if model.edge_weight_params.grad is not None:
            grad = model.edge_weight_params.grad
            print(f"\nEdge weight gradients:")
            print(f"  Shape: {grad.shape}")
            print(f"  Min: {grad.min().item():.6e}")
            print(f"  Max: {grad.max().item():.6e}")
            print(f"  Mean: {grad.mean().item():.6e}")
            print(f"  Std: {grad.std().item():.6e}")
            print(f"  Num zeros: {(grad == 0).sum().item()}")
            print(f"  Num non-zeros: {(grad != 0).sum().item()}")

            # Check for NaN or Inf
            if torch.isnan(grad).any():
                print(f"  WARNING: NaN detected in gradients!")
            if torch.isinf(grad).any():
                print(f"  WARNING: Inf detected in gradients!")
        else:
            print(f"\n  WARNING: No gradients computed!")

        # Update parameters (simple SGD step)
        with torch.no_grad():
            lr = 0.01
            if model.edge_weight_params.grad is not None:
                model.edge_weight_params -= lr * model.edge_weight_params.grad
                print(f"\nParameter update:")
                print(f"  Updated {(model.edge_weight_params.grad != 0).sum().item()} parameters")



def extract_grads_for_sparse_edges(dense_grad_vec, edge_index, num_nodes):
    dense_grad_matrix = create_symm_matrix_from_vec(dense_grad_vec, num_nodes)
    sparse_edge_grads = dense_grad_matrix[edge_index[0], edge_index[1]]

    return sparse_edge_grads / 2


def compare_dense_vs_sparse_gradients(dense_model, sparse_model, x, adj_dense, edge_index):
    """
    Compare gradients between dense and sparse implementations
    """
    print("=" * 80)
    print("DENSE VS SPARSE GRADIENT COMPARISON")
    print("=" * 80)

    for i in range(50):
    # Forward pass for both
        out_dense = dense_model.forward(x, adj_dense)
        out_sparse = sparse_model.forward()

        _ = dense_model.forward_prediction(x)
        _ = sparse_model.forward_hard()

        # Get predictions
        idx = sparse_model.index
        # print(out_sparse)
        y_pred_orig_dense = torch.argmax(out_dense[idx])
        y_pred_orig_sparse = torch.argmax(out_sparse)

        # print(f"\nPredictions:")
        # print(f"  Dense: {y_pred_orig_dense.item()}")
        # print(f"  Sparse: {y_pred_orig_sparse.item()}")

        # Compute losses
        loss_dense, _, _, _ = dense_model.loss(out_dense[idx], y_pred_orig_dense, y_pred_orig_dense)
        loss_sparse, _, _, _ = sparse_model.loss(out_sparse, y_pred_orig_sparse)

        # print(f"\nLosses:")
        # print(f"  Dense: {loss_dense.item():.6f}")
        # print(f"  Sparse: {loss_sparse.item():.6f}")
        # print(f"  Difference: {abs(loss_dense.item() - loss_sparse.item()):.2e}")

        # Backward
        dense_model.zero_grad()
        sparse_model.zero_grad()

        loss_dense.backward()
        loss_sparse.backward()

        # Compare P_vec gradients (dense) vs edge_weight_params gradients (sparse)
        if dense_model.P_vec.grad is None or sparse_model.edge_weight_params.grad is None:
            print("missing gradients")
            return
        bruh = extract_grads_for_sparse_edges(dense_model.P_vec.grad, sparse_model.edge_index, dense_model.num_nodes)
        moment = sparse_model.edge_weight_params.grad
        for E, i, j in zip(sparse_model.edge_index.T, bruh, moment):
            if i-j != 0:
                print(E, i, j, i-j)

        # print(f"\nGradient statistics:")
        # print(f"  Dense P_vec grad mean: {abs(bruh).mean().item():.6e}")
        # print(f"  Sparse edge_weight grad mean: {abs(moment).mean().item():.6e}")
        # print(f"  Dense P_vec grad std: {bruh.std().item():.6e}")
        # print(f"  Sparse edge_weight grad std: {moment.std().item():.6e}")
        print(f"difference: {sum(abs(bruh - moment))}")


        with torch.no_grad():
            lr = 0.1
            if dense_model.P_vec.grad is not None:
                dense_model.P_vec -= lr * dense_model.P_vec.grad
                # print(f"\nParameter update:")
                # print(f"  Updated {(dense_model.P_vec.grad != 0).sum().item()} parameters")
            if sparse_model.edge_weight_params.grad is not None:
                sparse_model.edge_weight_params -= lr * sparse_model.edge_weight_params.grad * 2
                # print(f"\nParameter update:")
                # print(f"  Updated {(sparse_model.edge_weight_params.grad != 0).sum().item()} parameters")


def trace_backward_chain(tensor, name="tensor", depth=0, max_depth=5):
    """
    Recursively trace the backward computation graph
    """
    indent = "  " * depth
    if tensor.grad_fn is None:
        print(f"{indent}{name}: Leaf tensor (no grad_fn)")
        return

    if depth >= max_depth:
        print(f"{indent}{name}: ... (max depth reached)")
        return

    print(f"{indent}{name}: {tensor.grad_fn}")

    if hasattr(tensor.grad_fn, 'next_functions'):
        for i, (fn, _) in enumerate(tensor.grad_fn.next_functions):
            if fn is not None:
                print(f"{indent}  -> input {i}: {fn}")


def check_numerical_stability(model, x, edge_index, epsilon=1e-5):
    """
    Check numerical stability of gradients using finite differences
    """
    print("=" * 80)
    print("NUMERICAL GRADIENT CHECK")
    print("=" * 80)

    # Compute analytical gradient
    output = model.forward()
    y_pred = model.original_class
    loss, _, _, _ = model.loss(output, y_pred, torch.argmax(output))

    model.zero_grad()
    loss.backward()
    analytical_grad = model.edge_weight_params.grad.clone()

    # Compute numerical gradient for first few parameters
    numerical_grad = torch.zeros_like(model.edge_weight_params)

    num_params_to_check = min(10, len(model.edge_weight_params))
    print(f"\nChecking first {num_params_to_check} parameters...")

    for i in range(num_params_to_check):
        # Perturb parameter up
        model.edge_weight_params.data[i] += epsilon
        output_plus = model.forward()
        loss_plus, _, _, _ = model.loss(output_plus, y_pred, torch.argmax(output_plus))

        # Perturb parameter down
        model.edge_weight_params.data[i] -= 2 * epsilon
        output_minus = model.forward()
        loss_minus, _, _, _ = model.loss(output_minus, y_pred, torch.argmax(output_minus))

        # Restore parameter
        model.edge_weight_params.data[i] += epsilon

        # Compute numerical gradient
        numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # Compare
        rel_error = abs(analytical_grad[i] - numerical_grad[i]) / (abs(analytical_grad[i]) + abs(numerical_grad[i]) + 1e-8)
        print(f"  Param {i}: analytical={analytical_grad[i].item():.6e}, "
              f"numerical={numerical_grad[i].item():.6e}, "
              f"rel_error={rel_error.item():.6e}")



script_dir = Path(__file__).parent
graph_path = script_dir / f'../data/gnn_explainer/syn4.pickle'
model_path = script_dir / f'../models/gcn_3layer_syn4.pt'


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

data = load_dataset(graph_path, device)

submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                        nclass=len(data.y.unique()), dropout=0)

submodel.load_state_dict(torch.load(model_path))
submodel.eval()

model = WrappedOriginalGCN(submodel).eval()

def remove_bidirectional_edges(edge_index):
    edges = edge_index.numpy().T

    sorted_edges = np.sort(edges, axis=1)
    unique_edges, indices = np.unique(sorted_edges, axis=0, return_index=True)

    result = edges[indices]

    return torch.tensor(result).t().contiguous()

# edge_index = remove_bidirectional_edges(data.edge_index)
edge_index = data.edge_index


x = data.x
index = 600
adj_dense = data.adj
y_orig = torch.argmax(model(x, edge_index), dim=1)[index]

dense_model = GCNSyntheticPerturb(
    10, 20, 20, 2, data.adj, 0, .5
)
dense_model.load_state_dict(submodel.state_dict(), strict=False)
sparse_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index)

# 1. First, check if forward passes match
print("Step 1: Checking forward pass equivalence...")
# edge_weights = torch.sigmoid(torch.ones(edge_index.shape[1]))
# norm_adj_dense = dense_model.forward(x, data.adj)[index]
# norm_adj_sparse = sparse_model.forward()
# print(norm_adj_dense, norm_adj_sparse)

# print(f"Norm adj difference: {(norm_adj_dense - norm_adj_sparse).abs().max()}")
# assert (norm_adj_dense - norm_adj_sparse).abs().max() < .01

# 2. Check if loss computation matches
print("\nStep 2: Checking loss computation...")
# out_dense = dense_model.forward(x, adj_dense)
# _ = dense_model.forward_prediction(x)
# out_sparse = sparse_model.forward()
# _ = sparse_model.forward_hard()

# loss_dense = dense_model.loss(out_dense[index], y_orig, y_orig)[0]
# loss_sparse = sparse_model.loss(out_sparse, y_orig)[0]

# print(loss_dense - loss_sparse)
# Compare losses...

# 3. Run gradient debugging
print("\nStep 3: Debugging gradients...")
# debug_gradient_flow(sparse_model, sparse_model.forward, x, edge_index, steps=3)
# debug_gradient_flow(dense_model, partial(dense_model.forward, x, data.adj), x, edge_index, steps=3)


# 4. Compare dense vs sparse gradients
print("\nStep 4: Comparing implementations...")
compare_dense_vs_sparse_gradients(dense_model, sparse_model, x, adj_dense, edge_index)

# 5. Numerical gradient check
print("\nStep 5: Numerical stability check...")
# check_numerical_stability(sparse_model, x, edge_index)