import argparse
import numpy as np
import pandas as pd

import torch

from example import load_dataset
from pathlib import Path

from torch_geometric.utils import k_hop_subgraph, mask_select

device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--dst', type=str, default='results.pkl')
parser.add_argument('--exp', type=str, default='syn1')
args = parser.parse_args()

# TODO: support different datasets
# TODO: think about subgraph cf method

# Load original graph
script_dir = Path(__file__).parent
graph_data_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
graph_data_path = graph_data_path.resolve()
data = load_dataset(graph_data_path, device)

with open(f'../results/{args.dst}', "rb") as f:
    df = pd.read_pickle(f)

# Find subgraph size for fidelity
df['subgraph_size'] = [(k_hop_subgraph(i, 4, data.edge_index)[1]).shape[1] for i in df['node']]

# Find edges between motif nodes
motif_edge_mask = torch.tensor([data.y[i] > 0 and data.y[j] > 0
                                for i, j in data.edge_index.T])
motif_edges = mask_select(data.edge_index, 1, motif_edge_mask)
motif_edges_set = set((i.item(), j.item()) for i, j in motif_edges.T)

# Filter out nodes with label 0
df_motif = df[df["label"] != 0].reset_index(drop=True)

# Accuracy defined as proportion of removed edges that were originally between motif nodes.
#! Accuracy currently lower because nodes w/o CFs get proportion of all edges in network.
#! Filtering for examples yields accuracy of 1 when explaining original model.
accuracy = []
for i in range(len(df_motif)):
    cf_edges = mask_select(data.edge_index, 1, ~torch.tensor(df_motif['cf_mask'][i]))

    overlap_count = sum((1 for i, j in cf_edges.T if (i.item(), j.item()) in motif_edges_set))
    accuracy.append(overlap_count / cf_edges.shape[1])

    if accuracy[-1] < 1:
        print(cf_edges)
        print(df_motif.iloc[i])

df_motif['accuracy'] = accuracy
# df_motif = df_motif.dropna()
cfs = df.dropna()

print(f'{args.exp} tested at {args.dst}')
print(f'Cf examples found: {len(cfs)}/{len(data.test_set)}, {len(df_motif)} non-zero nodes')
print(f'Fidelity: {1 - len(cfs) / len(data.test_set):.3f}')
print(f'Distance: {cfs["distance"].mean():.3f}, std: {cfs["distance"].std():.3f}')
print(f'Sparsity: {np.mean(1 - cfs["distance"] / cfs["subgraph_size"]):.3f}, std: {np.std(1 - cfs["distance"] / cfs["subgraph_size"]):.3f}')
print(f'Accuracy: {np.mean(df_motif["accuracy"]):.3f}, std: {np.std(df_motif["accuracy"]):.3f}')
print('')
df_motif = df_motif.dropna()
print(f'Distance: {df_motif["distance"].mean():.3f}, std: {df_motif["distance"].std():.3f}')
print(f'Sparsity: {np.mean(1 - df_motif["distance"] / df_motif["subgraph_size"]):.3f}, std: {np.std(1 - df_motif["distance"] / df_motif["subgraph_size"]):.3f}')
print(f'Accuracy: {np.mean(df_motif["accuracy"]):.3f}, std: {np.std(df_motif["accuracy"]):.3f}')
print('')
