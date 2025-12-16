import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv

from utils.utils import get_neighbourhood, normalize_adj
from cf_explanation.cf_explainer import CFExplainer, CFExplainerOriginal
from cf_explanation.cf_greed import GreedyCFExplainer
from cf_explanation.cf_bruteforce import BFCFExplainer
from torch_geometric.nn import GCNConv

import argparse
import numpy as np
import pickle

from tqdm import tqdm
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
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

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
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weights).relu()
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)

        return F.log_softmax(x2, dim=1)


def load_dataset(path, device):
    with open(path, 'rb') as f:
        graphdata = pickle.load(f)

    adj = torch.Tensor(graphdata["adj"]).squeeze()
    features = torch.Tensor(graphdata["feat"]).squeeze()

    labels = torch.tensor(graphdata["labels"]).squeeze()
    idx_train = torch.tensor(graphdata["train_idx"])
    idx_test = torch.tensor(graphdata["test_idx"])
    edge_index, edge_attr = dense_to_sparse(adj)

    norm_adj = normalize_adj(adj)

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        num_features=10,
        num_classes=len(labels.unique()),
        train_set = idx_train,
        test_set = idx_test,
        norm_adj = norm_adj,
        adj = adj
    )

    data.to(device)
    return data


def train_model(data, device, end=200):
    ''' Train GCN model '''
    model = GCN(data.num_features, data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=.001)

    train_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    train_mask[data.train_set] = True

    for _ in range(end):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    return model


def explain_original(model, data, lr=.1, n_momentum=0.9, epochs=500, device='cpu', dst='results'):
    predictions = torch.argmax(model(data.x, data.norm_adj), dim=1)

    test_cf_examples = []
    for i in tqdm(data.test_set):
        explainer = CFExplainerOriginal(model,
                                        data,
                                        i,
                                        n_hid=20,
                                        num_classes = data.num_classes,
                                        dropout=0.0,
                                        beta=.5,
                                        device=device)

        cf_example = explainer.explain(cf_optimizer='SGD', lr=lr,
                                       n_momentum=n_momentum, num_epochs=epochs)

        test_cf_examples.append([i.item(), data.y[i].item(), predictions[i].item()] + cf_example)

    df = pd.DataFrame(test_cf_examples, columns=columns)
    df.to_pickle(f"../results/{dst}.pkl")


def explain_new(data, model, cf_model = CFExplainer, dst='results',
                beta=0.5, lr=0.1, epochs=400, momentum=0.0, eps=1.0, stop=None):
    if stop is None:
        stop = len(data.test_set)

    write_to = [False]

    predictions = torch.argmax(model(data.x, data.edge_index), dim=1)
    explainer = Explainer(
        model=model,
        algorithm=cf_model(epochs=epochs, lr=lr,
                              storage=write_to,
                              beta=beta,
                               n_momentum=momentum,
                               eps=eps),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    test_cf_examples = []
    for n, i in tqdm(list(enumerate(data.test_set))):
        if n == stop:
            break

        _ = explainer(data.x, data.edge_index, index=i)

        if write_to[0]:
            test_cf_examples.append([i.item(), data.y[i].item(),
                                     predictions[i].item()] + write_to[-1])

    df = pd.DataFrame(test_cf_examples, columns=columns)
    df.to_pickle(f"../results/{dst}.pkl")


def main():
    script_dir = Path(__file__).parent
    graph_data_path = script_dir / '../data/gnn_explainer/syn5.pickle'
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
    model = train_model(data, device, end=500)
    model.eval()

    output = model(data.x, data.edge_index)
    y_pred_orig = torch.argmax(output, dim=1)
    predictions = output.argmax(dim=1)
    train_accuracy = (predictions == data.y).float().mean()
    print("y_true counts: {}".format(np.unique(data.y.numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
    print(f"Training accuracy: {train_accuracy:.4f}")

    explain_new(data, model, cf_model=cf_model, beta=.5, lr=.5, eps=0.5, epochs=500, momentum=.9)
    # explain_original(model, data, predictions, device)


if __name__ == '__main__':
    main()
