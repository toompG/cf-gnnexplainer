import torch

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, Linear

from gcn import GCNSynthetic
from utils.utils import get_degree_matrix, dense_to_sparse
from utils.test_functions import load_dataset

def edge_index2norm_adj(edge_index, edge_weight=None, num_nodes=None):
    ''' convert edge_index and edge_weight into normalised form the original model
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

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(deg, adj), deg)

    return norm_adj


class WrappedOriginalGCN(torch.nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.shape[0]
        out = self.submodel(x, edge_index2norm_adj(edge_index, edge_weight, num_nodes))
        return out

class GCNSyntheticPyG(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.gc1 = GCNConv(num_features, n_hid, normalize=False, bias=False)
        self.gc2 = GCNConv(n_hid, n_hid, normalize=False, bias=False)
        self.gc3 = GCNConv(n_hid, n_out, normalize=False, bias=False)

        # Manual bias parameters to match original behavior
        self.bias1 = Parameter(torch.zeros(n_hid))
        self.bias2 = Parameter(torch.zeros(n_hid))
        self.bias3 = Parameter(torch.zeros(n_out))

        self.lin = Linear(2 * n_hid + n_out, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = dense_to_sparse(edge_index)

        x1 = F.relu(self.gc1(x, edge_index, edge_weight=edge_weight) + self.bias1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.relu(self.gc2(x1, edge_index, edge_weight=edge_weight) + self.bias2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.gc3(x2, edge_index, edge_weight=edge_weight) + self.bias3
        x3 = F.dropout(x3, training=self.training)

        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

class GCNSyntheticUnNormedPyG(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.gc1 = GCNConv(num_features, n_hid, bias=False)
        self.gc2 = GCNConv(n_hid, n_hid, bias=False)
        self.gc3 = GCNConv(n_hid, n_out, bias=False)

        # Manual bias parameters to match original behavior
        self.bias1 = Parameter(torch.zeros(n_hid))
        self.bias2 = Parameter(torch.zeros(n_hid))
        self.bias3 = Parameter(torch.zeros(n_out))

        self.lin = Linear(2 * n_hid + n_out, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = dense_to_sparse(edge_index)

        x1 = F.relu(self.gc1(x, edge_index, edge_weight=edge_weight) + self.bias1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.relu(self.gc2(x1, edge_index, edge_weight=edge_weight) + self.bias2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.gc3(x2, edge_index, edge_weight=edge_weight) + self.bias3
        x3 = F.dropout(x3, training=self.training)

        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

def main():
    model = GCNSyntheticPyG(10, 2, 20, 20)

    # model_weights = torch.load('../models/gcn_3layer_syn4.pt')

    # Load trained weights from GCNSynthetic
    old_state_dict = torch.load('../models/gcn_3layer_syn4.pt')

    # Create mapping for GCNBruh
    new_state_dict = {
        'gc1.lin.weight': old_state_dict['gc1.weight'].T,
        'gc2.lin.weight': old_state_dict['gc2.weight'].T,
        'gc3.lin.weight': old_state_dict['gc3.weight'].T,
        'bias1': old_state_dict['gc1.bias'],
        'bias2': old_state_dict['gc2.bias'],
        'bias3': old_state_dict['gc3.bias'],
        'lin.weight': old_state_dict['lin.weight'],
        'lin.bias': old_state_dict['lin.bias']
    }
    model.load_state_dict(new_state_dict)
    model.eval()
    model.double()

    torch.set_default_dtype(torch.float64)

    bruhh = GCNSyntheticUnNormedPyG(10, 2, 20, 20)
    bruhh.load_state_dict(new_state_dict)
    bruhh.eval()
    bruhh.double()

    normal_model = GCNSynthetic(10, 20, 20, 2, 0.0)
    normal_model.load_state_dict(old_state_dict)
    normal_model.double()
    # data.norm_adj = data.norm_adj.double()
    data = load_dataset('../data/gnn_explainer/syn4.pickle', device='cpu')

    y_pred = bruhh(data.x.double(), data.adj.double())
    y_pred_normal = model(data.x.double(), data.norm_adj.double())
    difference = abs(y_pred - y_pred_normal)

    # print(y_pred - y_pred_normal)


    print(f'accuracy:   {((torch.argmax(y_pred, dim=1)) == data.y).float().mean()}')
    print(f'similarity: {(torch.argmax(y_pred, dim=1) == torch.argmax(y_pred_normal, dim=1)).float().mean()}')
    print(f'mean difference: {difference.mean()}, std {difference.std()})')


    edge_weight = torch.sigmoid(torch.ones(data.edge_index.shape[1]))
    A_tilde = F.sigmoid(torch.ones_like(data.norm_adj)) * data.adj.double() + torch.eye(data.norm_adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    print(norm_adj)

        # Use sigmoid to bound P_hat in [0,1]
    y_pred = bruhh(data.x.double(), data.adj.double(), edge_weight)
    y_pred_normal = model(data.x.double(), norm_adj)
    difference = abs(y_pred - y_pred_normal)

    print(f'accuracy:   {((torch.argmax(y_pred, dim=1)) == data.y).float().mean()}')
    print(f'similarity: {(torch.argmax(y_pred, dim=1) == torch.argmax(y_pred_normal, dim=1)).float().mean()}')
    print(f'mean difference: {difference.mean()}, std {difference.std()})')


if __name__ == '__main__':
    main()