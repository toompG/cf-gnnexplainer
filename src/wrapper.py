import torch


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
    adj[row, col] += edge_weights / 2
    adj[col, row] += edge_weights / 2


    adj = adj + torch.eye(num_nodes, device=device)

    with torch.no_grad():
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt = deg_inv_sqrt.detach()

    norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return norm_adj


class WrappedOriginalGCN(torch.nn.Module):
    def __init__(self, submodel, fn_convert=edge_index2norm_adj):
        super().__init__()
        self.submodel = submodel
        self.fn_convert = fn_convert

    def forward(self, x, edge_index, edge_weights=None):
        num_nodes = x.shape[0]
        return self.submodel(x, self.fn_convert(edge_index, edge_weights, num_nodes))
