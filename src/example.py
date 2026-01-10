import torch.nn.utils
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv

from utils.test_functions import explain_new, load_dataset

from cf_explanation.cf_explainer import CFExplainer, GreedyCFExplainer, BFCFExplainer
from torch_geometric.nn import GCNConv

import argparse
import numpy as np

from pathlib import Path


columns = ['node', 'label', 'prediction', 'cf_prediction',
           'distance', 'cf_mask']


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(num_features, n_hid, bias=False)
        self.conv2 = GCNConv(n_hid, n_hid, bias=False)
        self.conv3 = GCNConv(n_hid, n_out, bias=False)
        self.lin = nn.Linear(2 * n_hid + n_out, num_classes)
        self.dropout = dropout

        self.bias1 = nn.Parameter(torch.zeros(n_hid))
        self.bias2 = nn.Parameter(torch.zeros(n_hid))
        self.bias3 = nn.Parameter(torch.zeros(n_out))

    def forward(self, x, edge_index, edge_weights=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weights) + self.bias1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight=edge_weights).relu() + self.bias2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weights) + self.bias3)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)


class SmolGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(num_features, n_hid)
        self.conv2 = GCNConv(n_hid, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weights=None):
        x1 = self.conv1(x, edge_index, edge_weight=edge_weights).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weights)

        return F.log_softmax(x2, dim=1)


def train_model(data, device, end=200, save=False):
    ''' Train GCN model '''
    model = GCN(data.num_features, data.num_classes, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=.001)

    # syn1: lr=0.01 decay=0.001
    # syn4: lr=0.005 decay=0.001


    train_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    train_mask[data.train_set] = True

    for i in range(end):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if i % 50 == 0:
            print(torch.sum(torch.argmax(out, dim=1) == data.y))


    if save == True:
        torch.save(model.state_dict(), "../models/sparse_gcn_3layer_syn1.pt")
    return model

# syn1 .005 .9357
# syn4 .005 .8990
# syn5 .001 .8083

def main():
    script_dir = Path(__file__).parent
    graph_data_path = script_dir / '../data/gnn_explainer/syn1.pickle'
    graph_data_path = graph_data_path.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dst', type=str, default='results')
    parser.add_argument('--cf', type=str, default='cf')

    args = parser.parse_args()

    if args.cf == 'cf':
        cf_model = CFExplainer
    elif args.cf == 'greedy':
        cf_model = GreedyCFExplainer
    elif args.cf == 'bf':
        cf_model = BFCFExplainer
    else:
        raise AttributeError('Incorrect cf specified, use cf, greedy or bf')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   # TODO support device=cuda
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

    data = load_dataset(graph_data_path, device)
    model = train_model(data, device, end=1000, save=False)
    model.eval()

    output = model(data.x, data.edge_index)
    y_pred_orig = torch.argmax(output, dim=1)
    print(y_pred_orig)
    predictions = output.argmax(dim=1)
    train_accuracy = (predictions == data.y).float().mean()
    print("y_true counts: {}".format(np.unique(data.y.numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
    print(f"Training accuracy: {train_accuracy:.4f}")

    explain_new(data, model, cf_model=cf_model, beta=.5, lr=.1, eps=1, epochs=500)
    # explain_original(model, data, predictions, device)


if __name__ == '__main__':
    main()
