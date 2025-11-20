import math
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
    def __init__(self, nfeat, nhid, nout, nclass, edge_index,
                 dropout, beta, edge_additions=False):
        super(GCNSyntheticPerturbEdgeWeight, self).__init__()
        self.edge_index = edge_index
        self.nclass = nclass
        self.beta = beta
        self.dropout = dropout
        self.edge_additions = edge_additions

        # Number of edges in the graph
        self.num_edges = edge_index.shape[1]

        # Initialize edge weight parameters
        if self.edge_additions:
            # Start from zeros, will learn to add edges
            self.edge_weight_params = Parameter(torch.zeros(self.num_edges) + .01)
        else:
            # Start from ones, will learn to remove edges
            self.edge_weight_params = Parameter(torch.ones(self.num_edges))

        # GCN layers
        self.gc1 = GCNConv(nfeat, nhid, add_self_loops=True)
        self.gc2 = GCNConv(nhid, nhid, add_self_loops=True)
        self.gc3 = GCNConv(nhid, nout, add_self_loops=True)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)

        self.reset_parameters()

    def reset_parameters(self, eps=1e-4):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            if self.edge_additions:
                # Start slightly below 0 for additions
                self.edge_weight_params.data.fill_(eps)
            else:
                # Start slightly below 1 for deletions
                self.edge_weight_params.data.fill_(1.0 - eps)

    def forward(self, x, edge_index=None):
        """
        Forward pass with differentiable edge weights (for training)

        Args:
            x: Node features [num_nodes, nfeat]
            edge_index: Edge indices [2, num_edges] (optional, uses self.edge_index if None)
        """
        if edge_index is None:
            edge_index = self.edge_index

        # Apply sigmoid to bound edge weights in [0, 1]
        edge_weights_soft = torch.sigmoid(self.edge_weight_params)
        edge_weights = edge_weights_soft

        # print(self.edge_weight_params[:20])

        # Pass through GCN layers with learned edge weights
        x1 = F.relu(self.gc1(x, edge_index, edge_weight=edge_weights))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.gc2(x1, edge_index, edge_weight=edge_weights))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = self.gc3(x2, edge_index)

        # Concatenate and classify
        x = self.lin(torch.cat((x1, x2, x3), dim=1))

        # print(edge_weights[:30])

        return F.log_softmax(x, dim=1)

    def get_binary_adjacency(self):
        """
        Forward pass with thresholded edge weights (for evaluation)
        Returns predictions and the binary edge mask

        Args:
            x: Node features [num_nodes, nfeat]
            edge_index: Edge indices [2, num_edges] (optional, uses self.edge_index if None)
        """

        # Threshold edge weights at 0.5
        edge_weights_soft = torch.sigmoid(self.edge_weight_params)
        edge_mask = (edge_weights_soft >= 0.5)

        return edge_mask

    def loss(self, output, y_pred_orig, y_pred_new_actual, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index

        pred_same = float(int(y_pred_new_actual) == int(y_pred_orig))

        # Ensure proper dimensions for loss computation
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        # Get current edge weights
        edge_weights_soft = torch.sigmoid(self.edge_weight_params)
        edge_mask = (edge_weights_soft >= 0.5).float()

        loss_graph_dist = (1.0 - edge_mask).sum()

        # Prediction loss (negative to maximize distance from original prediction)
        loss_pred = -F.nll_loss(output, y_pred_orig)

        # Total loss: only apply pred_same when prediction changes
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        # print(f'{float(loss_total):.2f}')
        return loss_total, loss_pred, loss_graph_dist, edge_mask
