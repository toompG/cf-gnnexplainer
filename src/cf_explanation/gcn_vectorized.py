import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

# OKAY
# vectorized => written to more closely recreate original behaviour.

# plan : new cf explainer that uses this object instead
# from edge weights construct edge vector
# grad on edge_vec (grad)
# in forward(s), build edge_weights from edge_vec in forward (?)

# where i, j in edge weight (2-tuple)


def remove_bidirectional_edges(edge_index):
    edges = edge_index.numpy().T

    sorted_edges = np.sort(edges, axis=1)
    unique_edges, indices = np.unique(sorted_edges, axis=0, return_index=True)

    result = edges[indices]

    return torch.tensor(result).t().contiguous()


class GCNSyntheticPerturbEdgeWeight(nn.Module):
    """
    3-layer GCN using PyTorch Geometric's GCNConv with learnable edge weights
    for counterfactual explanations
    """
    def __init__(self, model, index, x, edge_index, beta=.5):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.P_edge = remove_bidirectional_edges(edge_index)
        self.index = index
        self.x = x
        self.beta = beta

        self.P_map = {}
        for i, j in self.P_edge.T:
            i, j = i.item(), j.item()

            # Find edge (i,j) using tensor operations
            edge_mask_ij = (edge_index[0] == i) & (edge_index[1] == j)
            edge_mask_ji = (edge_index[0] == j) & (edge_index[1] == i)

            # Get the index of the first occurrence
            idx_ij = torch.nonzero(edge_mask_ij, as_tuple=True)[0]
            idx_ji = torch.nonzero(edge_mask_ji, as_tuple=True)[0]

            # Convert to scalar or -1 if not found
            idx_ij = idx_ij[0].item() if len(idx_ij) > 0 else -1
            idx_ji = idx_ji[0].item() if len(idx_ji) > 0 else -1

            self.P_map[(i, j)] = (idx_ij, idx_ji)

        with torch.no_grad():
            self.original_class = torch.argmax(model(x, edge_index)[index])

        self.P_vec = Parameter(torch.ones(self.P_edge.shape[1]))
        self.edge_weight_params = torch.ones(edge_index.shape[1])

        self.reset_parameters()

    def reset_parameters(self, eps=1.0, noise=0.0):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            self.edge_weight_params.data.fill_(eps)
            if noise > 0.:
                self.edge_weight_params += (.5 - torch.rand_like(self.edge_weight_params)) * noise

    def reset_dataset(self, index, model=None, x=None, edge_index=None):
        self.index = index

        if model is not model:
            self.model = model
        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
            self.edge_weight_params = Parameter(torch.ones(edge_index.shape[1]))

        self.original_class = torch.argmax(self.model(self.x, self.edge_index)[index])
        self.reset_parameters()

    def forward(self):
        """
        predict class with edges weighted between 0 and 1
        """
        self.edge_weight_params = torch.zeros_like(self.edge_weight_params)
        for (i, j), w in zip(self.P_edge.T, self.P_vec):
            i, j = i.item(), j.item()
            a, b = self.P_map[(i, j)]
            self.edge_weight_params[a] = w
            self.edge_weight_params[b] = w

        return self.model(self.x, self.edge_index,
                          edge_weights=torch.sigmoid(self.edge_weight_params))[self.index]

    def forward_hard(self):
        """
        predict original model with edge deletion
        """

        # Threshold edge weights at 0.5
        edge_weights_soft = torch.sigmoid(self.edge_weight_params)
        self.edge_mask = (edge_weights_soft >= 0.5)
        self.masked_edge_index = self.edge_index[:, self.edge_mask]

        return self.model(self.x, self.masked_edge_index)[self.index]

    def loss(self, output, y_new):
        pred_same = float(y_new == self.original_class)

        # Ensure proper dimensions for loss computation
        output = output.unsqueeze(0)
        y_pred_orig = self.original_class.unsqueeze(0)

        # Get current edge weights
        loss_graph_dist = (~self.edge_mask).sum().float() / 2 # Count removed edges

        # Prediction loss (negative to maximize distance from original prediction)
        loss_pred = -F.nll_loss(output, y_pred_orig)

        # Total loss: only apply pred_same when prediction changes
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, pred_same, loss_graph_dist, self.edge_mask

    def get_weights(self):
        return self.edge_weight_params

    def sample_edge_importance(self, num_samples=5, eps=4.0, noise=0.4):
        """
        Measure average gradient change over multiple samples
        """
        importance_scores = torch.zeros_like(self.edge_weight_params)

        for _ in range(num_samples):
            self.reset_parameters(eps=eps, noise=noise)

            # Forward pass
            output = self.forward()
            self.edge_mask = torch.ones_like(self.edge_weight_params, dtype=bool)
            loss, _, _, _ = self.loss(output, self.original_class)

            self.zero_grad()
            loss.backward()

            importance_scores += self.edge_weight_params.grad

        return importance_scores / num_samples
