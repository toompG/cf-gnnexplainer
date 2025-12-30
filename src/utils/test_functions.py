import os
import sys
import torch
import pandas as pd
import pickle
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.utils import dense_to_sparse

from .utils import normalize_adj


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cf_explanation.cf_explainer import CFExplainer, CFExplainerOriginal


columns = ['node', 'label', 'prediction', 'cf_prediction',
           'distance', 'cf_mask']


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


def explain_new(data, model, cf_model=CFExplainer, dst='results', beta=0.5,
                lr=0.1, epochs=500, momentum=0.0, eps=1.0, noise=0.0, stop=None):
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
                           eps=eps,
                           noise=noise),
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
