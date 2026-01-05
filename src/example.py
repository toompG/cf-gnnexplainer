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
        self.conv1 = GCNConv(num_features, n_hid)
        self.conv2 = GCNConv(n_hid, n_hid)
        self.conv3 = GCNConv(n_hid, n_out)
        self.lin = nn.Linear(2 * n_hid + n_out, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weights=None):
        x1 = self.conv1(x, edge_index, edge_weight=edge_weights).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weights).relu()
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.conv3(x2, edge_index)
        x3 = F.dropout(x3, training=self.training)

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
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)

        return F.log_softmax(x2, dim=1)


def train_model(data, device, end=200):
    ''' Train GCN model '''
    model = GCN(data.num_features, data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=.001)

    # syn1: lr=0.005 decay=0.001
    # syn4: lr=0.005 decay=0.001


    train_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    train_mask[data.train_set] = True

    for _ in range(end):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    return model


def main():
    script_dir = Path(__file__).parent
    graph_data_path = script_dir / '../data/gnn_explainer/syn4.pickle'
    graph_data_path = graph_data_path.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20, help='Random seed.')
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
    model = train_model(data, device, end=1000)
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
