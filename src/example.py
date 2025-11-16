import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv

from utils.utils import get_neighbourhood
from cf_explanation.cf_explainer import CFExplainer
from torch_geometric.nn import GCNConv
from utils.utils import safe_open

import argparse
import numpy as np
import pickle

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 20)
        self.conv2 = GCNConv(20, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def load_dataset(path, device):
    with open(path, 'rb') as f:
        graphdata = pickle.load(f)

    adj = torch.Tensor(graphdata["adj"]).squeeze()
    features = torch.Tensor(graphdata["feat"]).squeeze()

    labels = torch.tensor(graphdata["labels"]).squeeze()
    idx_train = torch.tensor(graphdata["train_idx"])
    idx_test = torch.tensor(graphdata["test_idx"])
    edge_index, edge_attr = dense_to_sparse(adj)

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        num_features=10,
        num_classes=4,
        train_set = idx_train,
        test_set = idx_test
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


def main():
    # TODO support device=cuda
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

    data = load_dataset('../data/gnn_explainer/syn1.pickle', device)

    model = train_model(data, device, end=1000)
    model.eval()

    output = model(data.x, data.edge_index)
    predictions = output.argmax(dim=1)
    train_accuracy = (predictions == data.y).float().mean()
    print(f"Training accuracy: {train_accuracy:.4f}")

    write_to = [False]
    explainer = Explainer(
        model=model,
        algorithm=CFExplainer(epochs=400, lr=0.001, predictions=predictions,
                              storage=write_to, num_classes=4),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    test_cf_examples = []
    for i in tqdm(data.test_set):
        if data.y[i].item() == 0:
            continue

        # print(f"generating CF for {i} with {predictions[i], data.y[i]}")

        _, _, sub_labels, node_dict = get_neighbourhood(int(i),
                                                        data.edge_index,
                                                        3, # Magic number!
                                                        data.x,
                                                        data.y)
        new_idx = node_dict[int(i)]

        # explainer = CFExplainerOriginal(model=model,
        #                                 sub_adj=sub_adj,
        #                                 sub_feat=sub_feat,
        #                                 n_hid=20,
        #                                 dropout=0.5,
        #                                 sub_labels=sub_labels,
        #                                 y_pred_orig= predictions[i],
        #                                 num_classes = 2,
        #                                 beta=.5,
        #                                 device=device)

        # cf_example = explainer.explain(node_idx=i, cf_optimizer='SGD', new_idx=new_idx, lr=.001,
        #                             n_momentum=0.0, num_epochs=200)

        _ = explainer(data.x, data.edge_index, index=i)

        if write_to[0]:
            test_cf_examples.append([i.item(), sub_labels[new_idx]] + write_to[-1])


    with safe_open("../results/bruh2", "wb") as f:
        pickle.dump(test_cf_examples, f)


if __name__ == '__main__':
    main()
