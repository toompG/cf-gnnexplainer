from example import *

from pathlib import Path
from gcn import GCNSynthetic
from utils.utils import *
from torch_geometric.utils import to_dense_adj
from cf_explanation.cf_explainer import CFExplainer

from torch_geometric.nn.conv.gcn_conv import gcn_norm



script_dir = Path(__file__).parent
graph_data_path = script_dir / '../data/gnn_explainer/syn1.pickle'




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
    adj[row, col] = edge_weights

    adj = adj + torch.eye(num_nodes, device=device)
    deg = adj.sum(dim=1)

    # Compute D^(-1/2)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: D^(-1/2) @ A @ D^(-1/2)
    norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return norm_adj


class WrappedOriginalGCN(torch.nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel
        self.adj_norm = None

    def forward(self, x, edge_index, edge_weights=None):
        num_nodes = x.shape[0]

        # if edge_weights is None:
        #     if self.adj_norm is None:
        #         self.adj_norm = edge_index2norm_adj(edge_index)
        #     return self.submodel(x, self.adj_norm)

        # Convert to normalized adjacency matrix
        norm_adj = edge_index2norm_adj(
            edge_index,
            edge_weights,
            num_nodes
        )
        # print(norm_adj)
        # if edge_weights is not None:
        #     index, weight = gcn_norm(edge_index, edge_weights)
        #     print(torch.sum(weight - norm_adj[*index]))

        return self.submodel(x, norm_adj)


def main():
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

    data = load_dataset(graph_data_path, device)

    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)

    submodel.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format('syn1')))
    submodel.eval()

    model = WrappedOriginalGCN(submodel).eval()

    weights = torch.sigmoid(torch.ones_like(data.edge_index[0]))
    weights=None
    output = model(data.x, data.edge_index, edge_weights=weights)
    output_real = submodel(data.x, data.norm_adj)

    y_pred_orig = torch.argmax(output, dim=1)
    y_pred_orig_real = torch.argmax(output_real, dim=1)


    train_accuracy = (y_pred_orig == data.y).float().mean()
    train_accuracy_real = (y_pred_orig_real == data.y).float().mean()

    print(f'Wrapped accuracy: {train_accuracy}\nOriginal accuracy: {train_accuracy_real}')
    print(f'Difference: {torch.sum(output - output_real)}')

    #TODO: test difference for non-zero edge_weights

    explain_new(data, model, dst='original_model', beta=.5, lr=.1, epochs=200)


if __name__ == '__main__':
    main()