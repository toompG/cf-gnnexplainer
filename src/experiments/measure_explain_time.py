import sys
import os
from pathlib import Path

import torch
import pandas as pd

from functools import partial

from measure_performance import measure_function_time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, explain_new, explain_original, \
                                 load_sparse_dense_weights
from gcn import GCNSynthetic
from gcn_sparse import GCNReuseNormalisation
from cf_explanation.cf_explainer import CFExplainerNew, GreedyCFExplainer, BFCFExplainer


cf_explainers_sparse = [CFExplainerNew, GreedyCFExplainer, BFCFExplainer]

def measure_sparse(models, exp, data):
    results = []
    if exp == 'syn1' or 'syn2':
        momentum = 0.9
    else:
        momentum = 0.0

    for explain_method in cf_explainers_sparse:
        measurement = measure_function_time(
            partial(
                explain_new,
                model=models[exp]['Sparse'],
                x=data.x,
                edge_index=data.edge_index,
                y=data.y,
                target=data.test_set,
                n_momentum=momentum,
                cf_model=explain_method
            ),
            num_trials=1
        )
        measurement.insert(0, 'method', explain_method.__name__)
        measurement.insert(0, 'dataset', exp)
        # measurement.to_pickle(script_dir / f'../../results/explain_time/{exp}{explain_method.__name__}.pkl')
        results.append(measurement)
    return results


def measure_dense(models, exp, data):
    if exp == 'syn1' or 'syn2':
        momentum = 0.9
    else:
        momentum = 0.0
    results = []

    for s in [True, False]:
        measurement = measure_function_time(partial(
                explain_original,
                model=models[exp]['Dense'],
                data=data,
                n_momentum=momentum,
                skip=s,
                target=data.test_set
            ),
            num_trials=1
        )
        skipped = 'Skip' if s else 'Full'

        measurement.insert(0, 'method', 'Dense' + skipped)
        measurement.insert(0, 'dataset', exp)
        # measurement.to_pickle(script_dir / f'../../results/explain_time/{exp}Dense{skipped}.pkl')
        results.append(measurement)
    return results

def main():
    script_dir = Path(__file__).parent

    datasets = {
        'syn1': load_dataset(script_dir / '../../data/gnn_explainer/syn1.pickle', device='cpu'),
        'syn2': load_dataset(script_dir / '../../data/gnn_explainer/syn2.pickle', device='cpu'),
        'syn4': load_dataset(script_dir / '../../data/gnn_explainer/syn4.pickle', device='cpu'),
        'syn5': load_dataset(script_dir / '../../data/gnn_explainer/syn5.pickle', device='cpu')
    }

    models = {}
    for i in ['syn1', 'syn2', 'syn4', 'syn5']:
        model_path = script_dir / f'../../models/gcn_3layer_{i}.pt'
        models[i] = {'data': datasets[i]}

        data = datasets[i]
        model = GCNReuseNormalisation(10, data.num_classes)
        load_sparse_dense_weights(model, model_path)
        model.eval()

        y_pred = torch.argmax(model(data.x, data.edge_index), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models[i]['Sparse'] = model

        dense = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
        dense.load_state_dict(torch.load(model_path))
        dense.eval()

        y_pred = torch.argmax(dense(data.x, data.norm_adj), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models[i]['Dense'] = dense

    results = []
    for (exp, data) in datasets.items():
        results += measure_sparse(models, exp, data)
        results += measure_dense(models, exp, data)

    pd.concat(results, ignore_index=True).to_pickle(script_dir / f'../../results/explain_time/final_result.pkl')


if __name__ == '__main__':
    main()
