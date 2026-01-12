# import torch
from example import *
from cmp_original import WrappedOriginalGCN
from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from cf_explanation.gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight

from pathlib import Path
from gcn import GCNSynthetic
from utils.utils import *

from example import GCN

import time


floattype = torch.float32

torch.set_default_dtype(floattype)


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

    return sparse_edge_grads


def compare_dense_vs_sparse_gradients(dense_model, sparse_model, x, adj_dense, edge_index):
    """
    Compare gradients between dense and sparse implementations
    """
    print("=" * 80)
    print("DENSE VS SPARSE GRADIENT COMPARISON")
    print("=" * 80)

    P_edge = remove_bidirectional_edges(edge_index)
    y_pred_orig = torch.argmax(sparse_model.forward())
    print(y_pred_orig, type(y_pred_orig))

    for i in range(50):
    # Forward pass for both
        out_dense = dense_model.forward(x, adj_dense)
        out_sparse = sparse_model.forward()

        y_dense, _ = dense_model.forward_prediction(x)
        y_sparse = sparse_model.forward_hard()

        y_dense = torch.argmax(y_dense, dim=1)[index]
        y_sparse = torch.argmax(y_sparse)

        # Get predictions
        idx = sparse_model.index
        # print(out_sparse)
        # print(f"\nPredictions:")
        # print(f"  Dense: {y_pred_orig_dense.item()}")
        # print(f"  Sparse: {y_pred_orig_sparse.item()}")

        # Compute losses
        loss_dense, _, dist, _ = dense_model.loss(out_dense[idx], y_pred_orig, y_dense)
        loss_sparse, _, dist2, _ = sparse_model.loss(out_sparse, y_sparse)

        print(dist, dist2)
        print(f"Difference loss: {loss_dense.item()} {loss_sparse.item()}")
        print(f"Difference loss: {abs(loss_dense.item() - loss_sparse.item()):.2e}")
        # print(f'loss:{loss_dense - loss_sparse}')


        # Backward
        dense_model.zero_grad()
        sparse_model.zero_grad()

        loss_dense.backward()
        loss_sparse.backward()

        # Compare P_vec gradients (dense) vs edge_weight_params gradients (sparse)
        if dense_model.P_vec.grad is None or sparse_model.P_vec.grad is None:
            print("missing gradients")
            return
        bruh = extract_grads_for_sparse_edges(dense_model.P_vec.grad, P_edge, dense_model.num_nodes)
        moment = sparse_model.P_vec.grad
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
            if sparse_model.P_vec.grad is not None:
                sparse_model.P_vec -= lr * sparse_model.P_vec.grad
                # print(f"\nParameter update:")
                # print(f"  Updated {(sparse_model.edge_weight_params.grad != 0).sum().item()} parameters")




def compare_performance(dense_model, sparse_model, x, adj_dense,
                        index, epochs, lr=0.1):
    """
    Compare performance on a number of epochs
    """
    print("=" * 80)
    print(f"DENSE VS SPARSE TIME TO RUN {epochs} EPOCHS")
    print("=" * 80)

    start = time.time()
    index = sparse_model.index

    for _ in range(epochs):
        out_dense = dense_model.forward(x, adj_dense)
        _ = dense_model.forward_prediction(x)

        y_pred_orig_dense = torch.argmax(out_dense[index])

        # Compute losses
        loss_dense, _, _, _ = dense_model.loss(out_dense[index], y_pred_orig_dense, y_pred_orig_dense)

        dense_model.zero_grad()
        loss_dense.backward()

        # Compare P_vec gradients (dense) vs edge_weight_params gradients (sparse)
        with torch.no_grad():
            lr = 0.1
            if dense_model.P_vec.grad is not None:
                dense_model.P_vec -= lr * dense_model.P_vec.grad
    dense_time = time.time() - start
    print(f"DENSE: total {dense_time}s, average {dense_time / epochs}")

    start = time.time()
    for _ in range(epochs):
    # Forward pass for both
        out_sparse = sparse_model.forward()
        _ = sparse_model.forward_hard()

        y_pred_orig_sparse = torch.argmax(out_sparse)
        loss_sparse, _, _, _ = sparse_model.loss(out_sparse, y_pred_orig_sparse)

        # Backward
        sparse_model.zero_grad()
        loss_sparse.backward()

        with torch.no_grad():
            if sparse_model.P_vec.grad is not None:
                sparse_model.P_vec -= lr * sparse_model.P_vec.grad
    sparse_time = time.time() - start
    print(f"SPARSE: total {sparse_time:.3f}s, average {sparse_time / epochs:.3f}s")
    print(f"SPEEDUP: {dense_time / sparse_time}x")



script_dir = Path(__file__).parent
graph_path = script_dir / f'../data/gnn_explainer/syn4.pickle'
model_path = script_dir / f'../models/gcn_3layer_syn4.pt'
sparse_model_path = script_dir / f'../models/sparse_gcn_3layer_syn4.pt'


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

data = load_dataset(graph_path, device)
submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                        nclass=len(data.y.unique()), dropout=0)
submodel.load_state_dict(torch.load(model_path))

if floattype == torch.float64:
    data.x = data.x.double()
    # data.edge_index = data.edge_index.double()
    data.adj = data.adj.double()
    submodel.double()

submodel.eval()

model = WrappedOriginalGCN(submodel).eval()
# model = GCN(10, 2)
# model.load_state_dict(torch.load(sparse_model_path))


def remove_bidirectional_edges(edge_index):
    edges = edge_index.numpy().T

    sorted_edges = np.sort(edges, axis=1)
    unique_edges, indices = np.unique(sorted_edges, axis=0, return_index=True)

    result = edges[indices]

    return torch.tensor(result).t().contiguous()

edge_index = data.edge_index

index = 49
x = data.x
adj_dense = data.adj
y_orig = torch.argmax(model(x, edge_index), dim=1)[index]

dense_model = GCNSyntheticPerturb(
    10, 20, 20, 2, data.adj, 0, .5
)
dense_model.load_state_dict(submodel.state_dict(), strict=False)
sparse_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index)

# 1. First, check if forward passes match
print("Step 1: Checking forward pass equivalence...")
edge_weights = torch.sigmoid(torch.ones(edge_index.shape[1]))
norm_adj_dense = dense_model.forward(x, data.adj)[index]
norm_adj_sparse = sparse_model.forward()
print(norm_adj_dense, norm_adj_sparse)

print(f"Norm adj difference: {(norm_adj_dense - norm_adj_sparse).abs().max()}")
# assert (norm_adj_dense - norm_adj_sparse).abs().max() < .01

# 2. Check if loss computation matches
print("\nStep 2: Checking loss computation...")
out_dense = dense_model.forward(x, adj_dense)
_ = dense_model.forward_prediction(x)
out_sparse = sparse_model.forward()
_ = sparse_model.forward_hard()

loss_dense = dense_model.loss(out_dense[index], y_orig, y_orig)[0]
loss_sparse = sparse_model.loss(out_sparse, y_orig)[0]

print(loss_dense.item() - loss_sparse.item())
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
# quick_diagnostic(dense_model, sparse_model, x, adj_dense, edge_index)
# check_numerical_stability(sparse_model, x, edge_index)

dense_model.reset_parameters()
sparse_model.reset_parameters()

print('comparing 100 epochs')
compare_performance(dense_model, sparse_model, x, adj_dense, edge_index, epochs=100)