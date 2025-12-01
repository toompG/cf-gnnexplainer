import numpy as np
import pandas as pd
import torch
import sys
import os

from pathlib import Path

from skopt import gp_minimize
from skopt.space import Categorical, Real

from torch_geometric.utils import k_hop_subgraph, mask_select
from itertools import product
from functools import partial
from tqdm import tqdm

# import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example import load_dataset
from gcn import GCNSynthetic
from cf_explanation.gcn_edge_perturb import GCNSyntheticPerturbEdgeWeight
from cmp_original import WrappedOriginalGCN


script_dir = Path(__file__).parent


pair_options = [(24,1), (12,2), (6,4), (4,6), (2,12)]



def edge_importance(cf_model, num_samples=5, steps_per_sample=3, eps=1.0,
                    noise=0.1, lr=0.1):
    ''' Ranking function for edge weights '''
    scores = torch.zeros_like(cf_model.edge_weight_params)

    for _ in range(num_samples):
        cf_model.reset_parameters(eps=eps, noise=noise)

        optimizer = torch.optim.Adam([cf_model.edge_weight_params], lr=lr)

        scores -= cf_model.edge_weight_params
        for step in range(steps_per_sample):
            optimizer.zero_grad()
            output = cf_model.forward()
            # Required to set edge_mask
            cf_model.forward_hard()
            loss, _, _, _ = cf_model.loss(output, cf_model.original_class)
            loss.backward()
            optimizer.step()

        scores += cf_model.edge_weight_params

    return scores / num_samples


def load_datasets(device='cpu'):
    ''' Return a tuple of the 3 original datasets '''
    datasets = []
    for i in ['syn1', 'syn4', 'syn5']:
        data_path = script_dir / f'../../data/gnn_explainer/{i}.pickle'
        data_path = data_path.resolve()
        datasets.append(load_dataset(data_path, device))

    return tuple(datasets)


def set_models(*args):
    ''' Return a tuple of the 3 original models wrapped to accept sparse
     edges adjacency. '''
    models = []
    for i, d in zip(['syn1', 'syn4', 'syn5'], args):
        path = script_dir / f'../../models/gcn_3layer_{i}.pt'

        submodel = GCNSynthetic(nfeat=d.x.shape[1], nhid=20, nout=20,
                                nclass=len(d.y.unique()), dropout=0)

        submodel.load_state_dict(torch.load(path))
        submodel.eval()

        model = WrappedOriginalGCN(submodel)
        model.eval()
        models.append(model)

    return tuple(models)


def score_ranking(ranking, motif_edges):
    ''' Find average index position of first 3 motif edges '''
    scores = []
    for n, (_, i) in enumerate(ranking):
        v1 = i[0].item()
        v2 = i[1].item()

        if (v1, v2) in motif_edges or (v2, v1) in motif_edges:
            scores.append(n)

        if len(scores) > 3:
            break
    return sum(scores) / 6 - 1


def test_accuracy(cf_model, edge_rank, data, motif_edges, avgs={}):
    ''' For a model and edge_importance setup, find  '''
    accuracy = []
    for i in data.test_set[:100]:
        if data.y[i] == 0:
            continue

        if avgs.get((i, len(data.test_set)), 1.0) < .05:
            accuracy.append(0.0)

            continue

        cf_model.reset_dataset(i)
        edge_scores = edge_rank(cf_model)
        ranking = sorted(list(enumerate(data.edge_index.T)),
                         key=lambda x: -edge_scores[x[0]])
        score = score_ranking(ranking, motif_edges)
        avgs[(i, len(data.test_set))] = avgs.get((i, len(data.test_set)), 1.0) * .5 + score
        accuracy.append(score)

    # print(avgs)

    print(sum(accuracy) / len(accuracy))

    return sum(accuracy) / len(accuracy)


def objective(cf_model, models, datasets, motif_sets, params):
    pair, eps, noise, lr = params
    samples, steps = pair_options[pair]

    print(f'{samples}x{steps}, {eps:.3f}+-{noise:.3f}, lr={lr:.2f}')

    edge_rank = partial(
        edge_importance,
        num_samples=samples,
        steps_per_sample=steps,
        eps=eps,
        noise=noise,
        lr=lr,
    )
    skip = 1
    score = 0.0
    for m, d, motif in zip(models, datasets, motif_sets):
        if skip:
            skip = 0
            continue
        cf_model.reset_dataset(
            index=0, model=m,
            x=d.x,
            edge_index=d.edge_index
        )
        score += test_accuracy(cf_model, edge_rank, d, motif)
        # score += test_accuracy(cf_model, edge_rank, d, motif)
        # score += test_accuracy(cf_model, edge_rank, d, motif)
        # score += test_accuracy(cf_model, edge_rank, d, motif)
        # score += test_accuracy(cf_model, edge_rank, d, motif)

    return score


test_values = [
    [(24, 1), (12, 2), (6, 4), (4, 6), (2, 12), (24, 1)], # n_samples and steps
    [1.0, 0.5, 0.2], # starting edge weights before sigmoid
    [0.001, 0.01, 0.1], # starting noise to add
    [0.05, 0.1, 0.5], # learning rates
]

search_space = [
    Categorical(list(range(len(pair_options))), name="pair"),
    Real(-1.0, 5.0, name="eps"),
    Real(1e-4, 0.5, name="noise"),
    Real(5e-4, 1.0, name="lr"),
]

def main():
    # Load original graph
    experiment_names = ['syn1', 'syn4', 'syn5']

    datasets = load_datasets()
    motif_sets = []
    for data in datasets:
        motif_edge_mask = torch.tensor([data.y[i] > 0 and data.y[j] > 0
                                        for i, j in data.edge_index.T])
        motif_edges = mask_select(data.edge_index, 1, motif_edge_mask)
        motif_sets.append(set((i.item(), j.item()) for i, j in motif_edges.T))

    models = set_models(*datasets)

    # test_values.append(list(zip(datasets, models, motif_edge_mask)))
    cf_model = GCNSyntheticPerturbEdgeWeight(models[0], 1,
                                             datasets[0].x, datasets[0].edge_index)


    edge_rank = partial(
        edge_importance,
        num_samples=24,
        steps_per_sample=1,
        eps=1,
        noise=.1,
        lr=.2,
        )

    test_accuracy(cf_model, edge_rank, datasets[0], motif_sets[0])
    edge_rank = partial(
        edge_importance,
        num_samples=24,
        steps_per_sample=3,
        eps=1,
        noise=.1,
        lr=.2,
        )

    test_accuracy(cf_model, edge_rank, datasets[0], motif_sets[0])
    res = gp_minimize(
        func=partial(objective,
                     cf_model,
                     models,
                     datasets,
                     motif_sets),
        dimensions=search_space,
        n_calls=50,
        random_state=0,
        verbose=True
    )

    print(res)
    print("Best score:", res.fun)
    print("Best params:", res.x)

    # results = []
    # for (samples, steps), eps, noise, lr in tqdm(list(product(*test_values))):
    #     # Set params for edge-ranking function
    #     edge_rank = partial(edge_importance, num_samples=samples,
    #                         steps_per_sample=steps,
    #                         eps=eps, noise=noise, lr=lr)

    #     # Test current ranking function for the 3 experiments and save results
    #     for d, m, motif_edges, n in zip(datasets, models, motif_sets, experiment_names):
    #         cf_model.reset_dataset(index=0, model=m, x=d.x, edge_index=d.edge_index)
    #         accuracy = test_accuracy(cf_model, edge_rank, d, motif_edges)
    #         results.append([samples, steps, eps, noise, lr, n, accuracy])

    # print(*results, sep='\n')


if __name__ == '__main__':
    main()
