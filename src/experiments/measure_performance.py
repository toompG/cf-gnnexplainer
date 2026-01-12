import sys
import os
import time

from pathlib import Path

import torch
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
import pandas as pd
import numpy as np

import psutil
from memory_profiler import memory_usage
from collections import defaultdict
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset
from utils.utils import normalize_adj
from cf_explanation.gcn_vectorized import GCNSyntheticPerturbEdgeWeight
from cf_explanation.gcn_perturb import GCNSyntheticPerturb
from gcn import GCNSynthetic
from example import GCN


def get_process_memory():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_node(benchmark_fun, num_trials=5):
    """Benchmark with full memory tracking including C/C++"""
    results = defaultdict(list)

    # k_hop_edges = edge_index.shape[1]

    for trial in range(num_trials):
        mem_before = get_process_memory()

        # Define the training function to profile


        # Profile memory usage during training
        train_start = time.perf_counter()
        mem_usage = memory_usage(
            (benchmark_fun, ),
            interval=0.01,  # Sample every 10ms
            timeout=None,
            max_usage=False,  # Return full trace
            retval=True,
            include_children=True
        )
        total_train_time = time.perf_counter() - train_start

        # mem_usage is tuple: (memory_samples, return_value)
        memory_samples, (explainer, losses) = mem_usage

        mem_after = get_process_memory()

        # Store results
        results['trial'].append(trial)
        # results['num_edges'].append(k_hop_edges)
        results['total_train_time'].append(total_train_time)
        results['final_loss'].append(losses[-1])

        # Memory metrics (all include C/C++ allocations)
        results['mem_before_mb'].append(mem_before)
        results['mem_after_mb'].append(mem_after)
        results['mem_peak_mb'].append(max(memory_samples))
        results['mem_mean_mb'].append(np.mean(memory_samples))
        results['mem_delta_mb'].append(mem_after - mem_before)
        results['mem_samples'].append(len(memory_samples))

    return pd.DataFrame(results)


def train_explainer_coo(model, node_idx, x, adj, n_classes, num_epochs=100):
    explainer = GCNSyntheticPerturbEdgeWeight(
        model, node_idx, x, adj
    )
    optimizer = torch.optim.SGD([explainer.P_vec], lr=0.1)
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = explainer.forward()
        _ = explainer.forward_hard()
        loss, _, _, _ = explainer.loss(
            output, explainer.original_class
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return explainer, losses


def train_explainer_dense(model, node_idx, x, adj, n_classes, num_epochs=100):
    explainer = GCNSyntheticPerturb(
        10, 20, 20, n_classes, adj, 0.0, 0.5
    )
    explainer.load_state_dict(model.state_dict(), strict=False)
    original_class = torch.argmax(model(x, normalize_adj(adj))[node_idx])

    optimizer = torch.optim.SGD([explainer.P_vec], lr=0.1)
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = explainer.forward(x, adj)[node_idx]
        _ = explainer.forward_prediction(x)
        loss, _, _, _ = explainer.loss(
            output, original_class, original_class
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return explainer, losses


def benchmark_dataset(data, model, benchmark_fun, dense=False):
    results = []
    for i in tqdm(data.test_set):
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            int(i),
            4,
            data.edge_index,
            relabel_nodes=True
        )

        sub_x = data.x[sub_nodes]
        adj = to_dense_adj(sub_edge_index).squeeze() if dense else sub_edge_index

        sub_index = mapping if mapping.dim() > 0 else mapping.unsqueeze(0)
        node_results = benchmark_node(
            partial(benchmark_fun, model=model, node_idx=int(sub_index),
                    x=sub_x, adj=adj, n_classes=data.num_classes),
            num_trials=5
        )

        node_results.insert(0, 'node_idx', int(i))
        node_results['subgraph_size'] = len(sub_nodes)
        node_results['num_edges'] = sub_edge_index.shape[1]

        results.append(node_results)
    return pd.concat(results, ignore_index=True)


def main():
    script_dir = Path(__file__).parent

    datasets = [
        load_dataset(script_dir / '../../data/gnn_explainer/syn1.pickle', device='cpu'),
        load_dataset(script_dir / '../../data/gnn_explainer/syn4.pickle', device='cpu'),
        load_dataset(script_dir / '../../data/gnn_explainer/syn5.pickle', device='cpu')
    ]

    models = []
    for n, i in enumerate(['syn1', 'syn4', 'syn5']):
        model_path = script_dir / f'../../models/sparse_gcn_3layer_{i}.pt'
        data = datasets[n]
        model = GCN(10, data.num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        y_pred = torch.argmax(model(data.x, data.edge_index), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models.append(model)

        model_path = script_dir / f'../../models/gcn_3layer_{i}.pt'
        dense = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
        dense.load_state_dict(torch.load(model_path))
        dense.eval()

        y_pred = torch.argmax(dense(data.x, data.norm_adj), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models.append(dense)

    experiments = ['BAShapes Sparse', 'BAShapes Dense',
                   'TreeGrid Sparse', 'TreeGrid Dense',
                   'TreeCycle Sparse', 'TreeCycle Dense']

    results = []
    for i in range(6):
        benchmark_fun = train_explainer_coo if i % 2 == 0 else train_explainer_dense

        result = benchmark_dataset(datasets[i//2], models[i], benchmark_fun, dense=i%2)
        result.insert(0, 'dataset', experiments[i])
        results.append(result)

    pd.concat(results, ignore_index=True).to_pickle(f"../../results/performance.pkl")


if __name__ == '__main__':
    main()
    