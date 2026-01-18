import argparse
import numpy as np
import pandas as pd

import torch

from utils.test_functions import load_dataset
from pathlib import Path

from torch_geometric.utils import k_hop_subgraph, mask_select
from gcn import GCNSynthetic
from gcn_sparse import GCN
from cmp_original import WrappedOriginalGCN


def calculate_accuracy_new(df, data, motif_edges_set):
    ''' New accuracy calculation. Uses ground truths and only evaluates
     true positives found by classifier model. '''
    df_motif = df[df["label"] != 0].reset_index(drop=True)
    df_motif = df_motif[df_motif["prediction"] != 0].reset_index(drop=True)

    accuracy = []
    for i in range(len(df_motif)):
        cf_edges = data.edge_index[:, ~df_motif['cf_mask'][i]]
        overlap_count = sum((1 for i, j in cf_edges.T
                             if (i.item(), j.item()) in motif_edges_set or \
                                (j.item(), i.item()) in motif_edges_set))

        accuracy.append(overlap_count / cf_edges.shape[1])
    df_motif['accuracy'] = accuracy
    return df_motif


def calculate_accuracy_original(df, data, motif_nodes):
    ''' Exact recreation of original accuracy function. '''
    df_motif = df[df["prediction"] != 0].reset_index(drop=True)

    accuracy = []
    for i in range(len(df_motif)):
        cf_edges = data.edge_index[:, ~df_motif['cf_mask'][i]]
        nodes_involved = np.unique(np.concatenate((cf_edges[0], cf_edges[1]), axis=0))
        nodes_involved = nodes_involved[nodes_involved != df_motif.iloc[i, 0]]

        overlap_count = sum(1 for i in nodes_involved if i in motif_nodes)
        accuracy.append(overlap_count / len(nodes_involved))

    df_motif['accuracy'] = accuracy
    return df_motif


def main():
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', type=str, default='results.pkl')
    parser.add_argument('--exp', type=str, default='syn1')
    parser.add_argument('--sparse', type=bool, default=False)

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    graph_data_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    graph_data_path = graph_data_path.resolve()

    data = load_dataset(graph_data_path, device)
    if '.pkl' not in args.dst:
        args.dst = args.dst + '.pkl'

    if args.sparse:
        model_path = script_dir / f'../models/sparse_gcn_3layer_{args.exp}.pt'
        classifier = GCN(data.x.shape[1], data.num_classes)
        classifier.load_state_dict(torch.load(model_path))
        classifier.eval()
    else:
        model_path = script_dir / f'../models/gcn_3layer_{args.exp}.pt'
        submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                        nclass=len(data.y.unique()), dropout=0)

        submodel.load_state_dict(torch.load(model_path))
        submodel.eval()
        classifier = WrappedOriginalGCN(submodel).eval()

    with open(f'../results/{args.dst}', "rb") as f:
        df = pd.read_pickle(f)

    # Find subgraph size for sparsity calculation
    df['subgraph_size'] = [(k_hop_subgraph(i, 4, data.edge_index)[1]).shape[1] for i in df['node']]

    # Find edges between motif nodes
    motif_edge_mask = torch.tensor([data.y[i] > 0 and data.y[j] > 0
                                    for i, j in data.edge_index.T])
    motif_edges = mask_select(data.edge_index, 1, motif_edge_mask)
    motif_edges_set = set((i.item(), j.item()) for i, j in motif_edges.T)

    df_motif_new = calculate_accuracy_new(df, data, motif_edges_set)
    cfs = df.dropna().reset_index()

    predictions = torch.argmax(classifier(data.x, data.edge_index), dim=1)
    motif_nodes = set((i.item() for i in torch.where(predictions > 0)[0]))
    df_motif = calculate_accuracy_original(df, data, motif_nodes)

    for i in range(len(cfs)):
        cf_edges = data.edge_index[:, cfs['cf_mask'][i]]
        assert torch.argmax(classifier(data.x, cf_edges)[cfs['node'][i].item()]) == cfs['cf_prediction'][i]

    print(f'{args.exp} tested at {args.dst}')
    print(f'Cf examples found: {len(cfs)}/{len(data.test_set)}')
    print(f'Fidelity: {1 - len(cfs) / len(data.test_set):.3f}')
    print(f'Distance: {cfs["distance"].mean():.3f}, std: {cfs["distance"].std():.3f}')
    print(f'Sparsity: {np.mean(1 - cfs["distance"] / cfs["subgraph_size"] * 2):.3f}, std: {np.std(1 - cfs["distance"] / cfs["subgraph_size"]):.3f}')

    df_motif = df_motif.dropna()
    df_motif_new = df_motif_new.dropna()

    print(f'Accuracy: {np.mean(df_motif["accuracy"]):.3f}, std: {np.std(df_motif["accuracy"]):.3f}')
    print(f'Accuracy (new): {np.mean(df_motif_new["accuracy"]):.3f}, std: {np.std(df_motif_new["accuracy"]):.3f}')

    print('')


if __name__ == '__main__':
    main()
