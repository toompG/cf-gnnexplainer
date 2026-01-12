import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import find_edge_pairs


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

        self.matched_edges = find_edge_pairs(edge_index)

        with torch.no_grad():
            self.original_class = torch.argmax(model(x, edge_index)[index])

        self.P_vec = Parameter(torch.ones(self.edge_index.shape[1] // 2))
        self.reset_parameters()

    def reset_parameters(self, initial=1.0, eps=0.0):
        """Initialize edge weight parameters"""
        with torch.no_grad():
            self.P_vec.data.fill_(initial)
            if eps > 0.:
                self.P_vec += (.5 - torch.rand_like(self.P_vec)) * eps

    def forward(self):
        """
        predict class with edges weighted between 0 and 1
        """
        self.edge_weight_params = torch.empty(self.edge_index.shape[1])
        self.edge_weight_params[self.matched_edges[0]] = self.P_vec
        self.edge_weight_params[self.matched_edges[1]] = self.P_vec

        return self.model(self.x, self.edge_index,
                          edge_weight=torch.sigmoid(self.edge_weight_params))[self.index]

    def forward_hard(self):
        """
        predict original model with edge deletion
        """

        # Threshold edge weights at 0 (equivalent to doing sigmoid then 0.5)
        self.edge_mask = (self.edge_weight_params > 0)
        self.masked_edge_index = self.edge_index[:, self.edge_mask]

        return self.model(self.x, self.masked_edge_index)[self.index]

    def loss(self, output, y_new):
        pred_same = (y_new == self.original_class).float()

        output = output.unsqueeze(0)
        y_pred_orig = self.original_class.unsqueeze(0)

        # Predict loss as described in cf ggnexplainer paper
        loss_pred = -F.nll_loss(output, y_pred_orig)
        loss_graph_dist = (~self.edge_mask).sum() / 2
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, pred_same, loss_graph_dist, self.edge_mask

    def score_edges(self, num_samples=5, initial=1.0, eps=0.0):
        """
        Measure average gradient change over multiple samples
        """
        importance_scores = torch.zeros(self.P_vec.shape)

        for _ in range(num_samples):
            self.reset_parameters(initial=initial, eps=eps)

            # Forward pass
            output = self.forward()
            self.edge_mask = torch.ones(self.edge_index.shape[1], dtype=bool)
            loss, _, _, _ = self.loss(output, self.original_class)

            self.zero_grad()
            loss.backward()

            importance_scores -= self.P_vec.grad

        return list(zip(*self.matched_edges)), importance_scores


