import torch
from utils.utils import get_degree_matrix

def edge_index2norm_adj(edge_index, edge_weights=None, num_nodes=None):
    ''' convert edge_index and edge_weight into normalised form the original model
     expects '''
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    device = edge_index.device

    adj = torch.zeros((num_nodes, num_nodes), device=device)

    if edge_weights is None:
        edge_weights = torch.ones(edge_index.shape[1], device=device)

    row, col = edge_index
    adj[row, col] += edge_weights

    adj = adj + torch.eye(num_nodes, device=device)

    with torch.no_grad():
        deg = get_degree_matrix(adj).detach()
        deg = deg ** (-1 / 2)
        deg[torch.isinf(deg)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(deg, adj), deg)

    return norm_adj


class WrappedOriginalGCN(torch.nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel

    def forward(self, x, edge_index, edge_weights=None):
        num_nodes = x.shape[0]
        return self.submodel(x, edge_index2norm_adj(edge_index, edge_weights, num_nodes))
