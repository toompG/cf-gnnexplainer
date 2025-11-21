import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from gcn import GraphConvolution, GCNSynthetic

def symm_matrix_from_vec(vec, N, tril_i, tril_j):
    """Reconstruct symmetric N×N matrix from vectorized lower-triangular entries."""
    M = torch.zeros(N, N, device=vec.device, dtype=vec.dtype)
    M[tril_i, tril_j] = vec
    M[tril_j, tril_i] = vec  # Make symmetric
    return M

class GraphConvolutionPerturb(nn.Module):
    """
    Similar to GraphConvolution except includes P_hat
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionPerturb, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias is not None:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCNSyntheticPerturb(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
        super().__init__()

        self.adj = adj.float()
        self.num_nodes = adj.size(0)
        self.nclass = nclass
        self.beta = beta
        self.edge_additions = edge_additions
        self.dropout = dropout

        self.tril_i, self.tril_j = torch.tril_indices(self.num_nodes, self.num_nodes)
        self.P_vec_size = len(self.tril_i)

        if self.edge_additions:
            self.P_vec = nn.Parameter(torch.zeros(self.P_vec_size))
        else:
            self.P_vec = nn.Parameter(torch.ones(self.P_vec_size))

        self.reset_parameters()

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)

        # Final classifier on concatenated GCN outputs
        self.lin = nn.Linear(nhid + nhid + nout, nclass)


    def reset_parameters(self, eps=10**-4):
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = self.adj[self.tril_i, self.tril_j]
                init_vec = adj_vec + eps * torch.randn_like(adj_vec)  # small random noise
            else:
                init_vec = -eps * torch.ones(self.P_vec_size)

            self.P_vec.data.copy_(init_vec)


    def forward(self, x, sub_adj):
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde

        # Use sigmoid to bound P_hat in [0,1]
        P_hat = symm_matrix_from_vec(self.P_vec, self.num_nodes, self.tril_i, self.tril_j)
        P_hat = torch.sigmoid(P_hat)

        if self.edge_additions:
            # Learn new adj matrix directly
            A_tilde = P_hat + torch.eye(self.num_nodes)
        else:
            # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # Use sigmoid to bound P_hat in [0,1]
            A_tilde = P_hat * sub_adj + torch.eye(self.num_nodes)

        D_tilde = A_tilde.sum(dim=1)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_sqrt = D_tilde.pow(-0.5)
        D_tilde_sqrt[torch.isinf(D_tilde_sqrt)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = D_tilde_sqrt[:, None] * A_tilde * D_tilde_sqrt[None, :]

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))

        return F.log_softmax(x, dim=1)


    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions

        P_hat = symm_matrix_from_vec(self.P_vec, self.num_nodes, self.tril_i, self.tril_j)
        self.P = (torch.sigmoid(P_hat) >= 0.5).float()

        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)

        D_tilde = A_tilde.sum(dim=1)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_sqrt = D_tilde.pow(-0.5)
        D_tilde_sqrt[torch.isinf(D_tilde_sqrt)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = D_tilde_sqrt[:, None] * A_tilde * D_tilde_sqrt[None, :]

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))

        return F.log_softmax(x, dim=1), self.P


    def  loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj

        cf_adj = cf_adj.clone().requires_grad_(True)

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)

        # Number of edges changed (symmetrical)
        loss_graph_dist = torch.abs(cf_adj - self.adj).sum() / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist, cf_adj
