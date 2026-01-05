import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def remove_bidirectional_edges(edge_index):
    edges = edge_index.numpy().T

    sorted_edges = np.sort(edges, axis=1)
    _, indices = np.unique(sorted_edges, axis=0, return_index=True)

    result = edges[indices]

    return torch.tensor(result).t().contiguous()


class GCNSyntheticPerturbEdgeWeight(nn.Module):
    """
    3-layer GCN using PyTorch Geometric's GCNConv with learnable edge weights
    for counterfactual explanations
    """
    def __init__(self, model, index, x, edge_index, beta=.5, symmetric=True):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.index = index
        self.x = x
        self.beta = beta
        self.symmetric = symmetric

        if symmetric:
            self.P_edge = remove_bidirectional_edges(edge_index)
            self.matched_edges = torch.concat((self.P_edge, reversed(self.P_edge)), dim=1)
            self.P_middle = self.P_edge.shape[1]

            self.result = torch.ones(edge_index.shape[1])

            self.P_map = []
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

                self.P_map.append((idx_ij, idx_ji))

        with torch.no_grad():
            self.original_class = torch.argmax(model(x, edge_index)[index])

        self.P_vec = Parameter(torch.ones(self.P_edge.shape[1]))
        self.edge_weight_params = torch.ones(edge_index.shape[1])

        self.reset_parameters()

    def reset_parameters(self, eps=1.0, noise=0.0):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            self.P_vec.data.fill_(eps)
            if noise > 0.:
                self.P_vec += (.5 - torch.rand_like(self.P_vec)) * noise

    def reset_dataset(self, index, model=None, x=None, edge_index=None):
        self.index = index

        if model is not model:
            self.model = model
        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
            self.edge_weight_params = torch.ones(edge_index.shape[1])

        self.original_class = torch.argmax(self.model(self.x, self.edge_index)[index])
        self.reset_parameters()

    def forward(self):
        """
        predict class with edges weighted between 0 and 1
        """
        self.edge_weight_params = torch.concat((self.P_vec, self.P_vec))
        return self.model(self.x, self.matched_edges,
                          edge_weights=torch.sigmoid(self.edge_weight_params))[self.index]

    def forward_hard(self):
        """
        predict original model with edge deletion
        """

        # Threshold edge weights at 0 (equivalent to doing sigmoid then 0.5)
        self.edge_mask = (self.edge_weight_params > 0)
        self.masked_edge_index = self.matched_edges[:, self.edge_mask]

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

        if pred_same == 0 and self.symmetric:
            self.removed = torch.where(self.P_vec < 0)[0]
            self.result = torch.ones(self.edge_index.shape[1])

            for i in self.removed:
                a, b = self.P_map[i]
                self.result[a] = 0
                self.result[b] = 0
                assert self.edge_index.T[a][0] == self.edge_index.T[b][1]
                assert self.edge_index.T[a][1] == self.edge_index.T[b][0]

        return loss_total, pred_same, loss_graph_dist, self.result

    def sample_edge_importance(self, num_samples=5, eps=4.0, noise=0.4):
        """
        Measure average gradient change over multiple samples
        """
        importance_scores = torch.zeros_like(self.P_vec)

        for _ in range(num_samples):
            self.reset_parameters(eps=eps, noise=noise)

            # Forward pass
            output = self.forward()
            self.edge_mask = torch.ones_like(self.edge_weight_params, dtype=bool)
            loss, _, _, _ = self.loss(output, self.original_class)

            self.zero_grad()
            loss.backward()

            importance_scores += self.P_vec.grad

        # map matched edge indices to original edge ordering
        if self.symmetric:
            # cast to list to save valeus at the start of iterating
            results = torch.zeros_like(self.edge_weight_params)
            for (i, j), importance in list(zip(self.P_map, importance_scores)):
                results[i] = importance
                results[j] = importance
            return results / num_samples

        return importance_scores / num_samples
