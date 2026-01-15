'''
gcn_sparse.py

gcn_sparse implements a GCN for COO-formatted node classification. Training
settings for every model used in the experiments are stored and may be
re-used using

python3 gcn_sparse.py --exp=[DATASET]
'''


import torch.nn.utils
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv

from utils.test_functions import load_dataset, explain_new

import argparse
import numpy as np

from pathlib import Path


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

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight) + self.bias1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight=edge_weight) + self.bias2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.conv3(x2, edge_index, edge_weight) + self.bias3
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)


class SmolGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_hid=20, n_out=20, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(num_features, n_hid)
        self.conv2 = GCNConv(n_hid, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)

        return F.log_softmax(x2, dim=1)


def train_model(data, device, lr, hidden, dropout, weight_decay, clip,
                end=500, dst='result', save=False, show_progress=True):
    ''' Train GCN model '''
    model = GCN(data.num_features, data.num_classes,
                hidden, hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    train_mask[data.train_set] = True

    for i in range(end):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if show_progress and i % 50 == 0:
            print(torch.sum(torch.argmax(out, dim=1) == data.y))

    if save == True:
        torch.save(model.state_dict(), f"../models/{dst}.pt")
    return model


presets = {
    'syn1': {'seed': 42, 'epochs': 1000, 'lr': 0.001, 'hidden': 20,
             'dropout': 0.0, 'weight_decay': 0.001, 'clip': 2.0},
    'syn2': {'seed': 42, 'epochs': 1000, 'lr': 0.005, 'hidden': 20,
             'dropout': 0.0, 'weight_decay': 0.001, 'clip': 2.0},
    'syn4': {'seed': 42, 'epochs': 1000, 'lr': 0.005, 'hidden': 20,
             'dropout': 0.0, 'weight_decay': 0.001, 'clip': 2.0},
    'syn5': {'seed': 42, 'epochs': 1000, 'lr': 0.002, 'hidden': 20,
             'dropout': 0.0, 'weight_decay': 0.001, 'clip': 2.0},
}


def main():
    script_dir = Path(__file__).parent
    graph_data_path = script_dir / '../data/gnn_explainer/syn1.pickle'
    graph_data_path = graph_data_path.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='syn1')
    parser.add_argument('--dst', type=str, default='result')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--preset', default=True, help='Use hyperparams chosen for experiments (overwrites most options)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')
    parser.add_argument('--verbose', type=bool, default=True, help='Print updates on success rate')


    args = parser.parse_args()

    if args.preset == True:
        seed = presets[args.exp]['seed']
        epochs = presets[args.exp]['epochs']
        lr = presets[args.exp]['lr']
        hidden = presets[args.exp]['hidden']
        dropout=  presets[args.exp]['dropout']
        weight_decay = presets[args.exp]['weight_decay']
        clip = presets[args.exp]['clip']
    else:
        seed = args.seed
        epochs = args.epochs
        lr = args.lr
        hidden = args.hidden
        dropout = args.dropout
        weight_decay = args.weight_decay
        clip = args.clip


    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cpu'

    graph_data_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    graph_data_path = graph_data_path.resolve()

    data = load_dataset(graph_data_path, device)
    model = train_model(data, device, lr=lr, hidden=hidden, dropout=dropout,
                        weight_decay=weight_decay, clip=clip, end=epochs, dst=args.dst, save=args.save)
    model.eval()

    output = model(data.x, data.edge_index)
    y_pred_orig = torch.argmax(output, dim=1)
    predictions = output.argmax(dim=1)
    train_accuracy = (predictions == data.y).float().mean()
    print("y_true counts: {}".format(np.unique(data.y.numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
    print(f"Training accuracy: {train_accuracy:.4f}")

    explain_new(model, data.x, data.edge_index, data.test_set, data.y)


if __name__ == '__main__':
    main()
