import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv


class GCNSyntheticPerturbEdgeWeight(nn.Module):
    """
    3-layer GCN using PyTorch Geometric's GCNConv with learnable edge weights
    for counterfactual explanations
    """
    def __init__(self, model, index, x, edge_index, beta=.5, edge_additions=False):
        super(GCNSyntheticPerturbEdgeWeight, self).__init__()
        self.model = model
        self.edge_index = edge_index
        self.index = index
        self.x = x
        self.beta = beta

        self.edge_additions = edge_additions

        self.original_class = torch.argmax(model(x, edge_index)[index])

        # Initialize edge weight parameters
        if self.edge_additions:
            # Start from zeros, will learn to add edges
            self.edge_weight_params = Parameter(torch.zeros(edge_index.shape[1]))
        else:
            # Start from ones, will learn to remove edges
            self.edge_weight_params = Parameter(torch.ones(edge_index.shape[1]))
        self.reset_parameters()

    def reset_parameters(self, eps=.5, noise=0.0):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            if self.edge_additions:
                # Start slightly below 0 for additions
                self.edge_weight_params.data.fill_(eps)
            else:
                # Start slightly below 1 for deletions
                self.edge_weight_params.data.fill_(1.0 - eps)

    def forward(self):
        """
        predict class with edges weighted between 0 and 1
        """
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
        loss_graph_dist = (~self.edge_mask).sum().float()  # Count removed edges

        # Prediction loss (negative to maximize distance from original prediction)
        loss_pred = -F.nll_loss(output, y_pred_orig)

        # Total loss: only apply pred_same when prediction changes
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, pred_same, loss_graph_dist, self.edge_mask
