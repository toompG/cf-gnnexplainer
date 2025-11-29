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
    def __init__(self, model, index, x, edge_index, beta=.5):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.index = index
        self.x = x
        self.beta = beta

        self.original_class = torch.argmax(model(x, edge_index)[index])

        # Initialize edge weight parameters
        self.edge_weight_params = Parameter(torch.ones(edge_index.shape[1]))
        self.reset_parameters()

    def reset_parameters(self, eps=1., noise=0.):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            self.edge_weight_params.data.fill_(eps)
            if noise > 0.:
                self.edge_weight_params += torch.rand_like(self.edge_weight_params) * noise

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

    def get_weights(self):
        return self.edge_weight_params

    def compute_edge_importance_gradients(self, num_samples=5, eps=1.0, noise=0.0):
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

            importance_scores += torch.abs(self.edge_weight_params.grad)

        return importance_scores / num_samples
