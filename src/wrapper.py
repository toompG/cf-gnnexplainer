'''
wrapper.py

Contains two interfaces that allow us to use original classifier models
in new cf-gnnexplainer that expects COO formatting.
'''

import torch

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import add_self_loops

from utils.utils import get_degree_matrix


def edge_index2norm_adj(edge_index, edge_weight=None, num_nodes=None):
    ''' Convert edge_index and edge_weight into normalised form the original model
     expects '''
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    device = edge_index.device

    adj = torch.zeros((num_nodes, num_nodes), device=device)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device)

    row, col = edge_index
    adj[row, col] += edge_weight

    adj = adj + torch.eye(num_nodes, device=device)

    with torch.no_grad():
        deg = get_degree_matrix(adj).detach()
        deg = deg ** (-1 / 2)
        deg[torch.isinf(deg)] = 0

    norm_adj = torch.mm(torch.mm(deg, adj), deg)

    return norm_adj


class WrappedOriginalGCN(torch.nn.Module):
    '''
    Interface for original GCNSynthetic model that converts edges and weights
    in coo-format to full normalised adjacency matrix. Results are completely
    identical.
    '''

    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.shape[0]
        out = self.submodel(x, edge_index2norm_adj(edge_index, edge_weight,
                                                   num_nodes))
        return out


class GCNConvGCNSynthetic(torch.nn.Module):
    '''
    Version of GCNSynthetic that uses GCNConv with weights from original
    to calculate forward and backward. Results are within floating point
    error.
    '''
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(num_features, n_hid, add_self_loops=False, normalize=False, bias=False)
        self.conv2 = GCNConv(n_hid, n_hid, add_self_loops=False, normalize=False, bias=False)
        self.conv3 = GCNConv(n_hid, n_out, add_self_loops=False, normalize=False, bias=False)

        self.bias1 = Parameter(torch.zeros(n_hid))
        self.bias2 = Parameter(torch.zeros(n_hid))
        self.bias3 = Parameter(torch.zeros(n_out))

        self.lin = Linear(2 * n_hid + n_out, num_classes)
        self.dropout = dropout

    def normalize_adj(self, edge_index, edge_weight, num_nodes):
        '''
        Normalization weights calculated manually rather than relying on
        GCNConv functionality. Using normalize=True in GCNConv works for forward
        pass but diverges in loss.backward()
        '''
        A_tilde = torch.zeros(num_nodes, num_nodes) + torch.eye(num_nodes)
        A_tilde[edge_index[0], edge_index[1]] = edge_weight
        D_tilde = get_degree_matrix(A_tilde).detach()
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes
        )

        edge_weight = norm_adj[edge_index[0], edge_index[1]]
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
        edge_index_norm, edge_weight_norm = self.normalize_adj(edge_index, edge_weight, num_nodes)

        x1 = F.relu(self.conv1(x, edge_index_norm, edge_weight=edge_weight_norm) + self.bias1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index_norm, edge_weight=edge_weight_norm) + self.bias2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.conv3(x2, edge_index_norm, edge_weight=edge_weight_norm) + self.bias3
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)
