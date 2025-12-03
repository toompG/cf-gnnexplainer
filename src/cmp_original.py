from example import *

from pathlib import Path
from gcn import GCNSynthetic
from utils.utils import *
from torch_geometric.utils import to_dense_adj
from functools import lru_cache

from cf_explanation.cf_explainer import CFExplainer, CFExplainerOriginal
from cf_explanation.cf_greed import GreedyCFExplainer
from cf_explanation.cf_bruteforce import BFCFExplainer

from torch_geometric.nn.conv.gcn_conv import gcn_norm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='syn1')
parser.add_argument('--dst', type=str, default='results')
parser.add_argument('--lr', type=float, default=.5)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--cf', type=str, default='cf')

args = parser.parse_args()

if args.cf == 'cf':
    cf_model = CFExplainer
elif args.cf == 'greedy':
    cf_model = GreedyCFExplainer
elif args.cf == 'bf':
    cf_model = BFCFExplainer
elif args.cf == 'original':
    cf_model = CFExplainerOriginal
else:
    raise AttributeError('Incorrect cf specified, use cf, greedy or bf')


script_dir = Path(__file__).parent
graph_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
model_path = script_dir / f'../models/gcn_3layer_{args.exp}.pt'


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
        self.edge_cache = {}

    def forward(self, x, edge_index, edge_weights=None):
        num_nodes = x.shape[0]

        # if edge_weights is None:
        #     # print(tuple(edge_index))
        #     if edge_index in self.edge_cache:
        #         print('cache hit')
        #         return self.edge_cache[edge_index]
        #     result = self.submodel(x, edge_index2norm_adj(edge_index,
        #                                                   num_nodes=num_nodes))
        #     self.edge_cache[edge_index] = result
        #     return result

        return self.submodel(x, edge_index2norm_adj(edge_index, edge_weights,
                                                    num_nodes)
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='syn1')
    parser.add_argument('--dst', type=str, default='results')
    parser.add_argument('--lr', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--cf', type=str, default='cf')

    args = parser.parse_args()

    if args.cf == 'cf':
        cf_model = CFExplainer
    elif args.cf == 'greedy':
        cf_model = GreedyCFExplainer
    elif args.cf == 'bf':
        cf_model = BFCFExplainer
    elif args.cf == 'original':
        cf_model = CFExplainerOriginal
    else:
        raise AttributeError('Incorrect cf specified, use cf, greedy or bf')


    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    model_path = script_dir / f'../models/gcn_3layer_{args.exp}.pt'


    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

    data = load_dataset(graph_path, device)

    print(len(data.y.unique()))
    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)

    submodel.load_state_dict(torch.load(model_path))
    submodel.eval()

    if args.cf == 'original':
        # pred = torch.argmax(submodel(data.x, data.norm_adj), dim=1)
        # train_accuracy_real = (pred == data.y).float().mean()
        # print(train_accuracy_real)

        explain_original(submodel, data)
        return

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

    explain_new(data, model, cf_model=cf_model, dst=args.dst, beta=2, lr=args.lr, epochs=500)


if __name__ == '__main__':
    main()
