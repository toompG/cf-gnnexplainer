import torch

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, Linear

from gcn import GCNSynthetic
from utils.utils import get_degree_matrix, dense_to_sparse
from utils.test_functions import load_dataset, explain_new, explain_original

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


class GCNBruh(torch.nn.Module):
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

    def forward(self, x, edge_index, edge_weights=None):
        edge_index, edge_weights = dense_to_sparse(edge_index)

        x1 = self.gc1(x, edge_index, edge_weight=edge_weights) + self.bias1
        x1 = x1.relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.gc2(x1, edge_index, edge_weight=edge_weights) + self.bias2
        x2 = x2.relu()
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.gc3(x2, edge_index, edge_weight=edge_weights) + self.bias3
        x3 = F.dropout(x3, training=self.training)

        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

def main():
    model = GCNBruh(10, 2, 20, 20)

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

    normal_model = GCNSynthetic(10, 20, 20, 2, 0.0)
    normal_model.load_state_dict(old_state_dict)
    data = load_dataset('../data/gnn_explainer/syn4.pickle', device='cpu')

    y_pred = model(data.x, data.norm_adj)

    print((torch.argmax(y_pred, dim=1) == data.y).float().mean())
    y_pred_normal = normal_model(data.x, data.norm_adj)
    print(y_pred - y_pred_normal)

    explain_original(model, data)


if __name__ == '__main__':
    main()