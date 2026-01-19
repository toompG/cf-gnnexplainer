import sys
import os
from pathlib import Path

import torch
import pandas as pd

from itertools import product
from functools import partial

from measure_performance import measure_function_time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, explain_new, explain_original
from gcn import GCNSynthetic
from gcn_sparse import GCN
from wrapper import WrappedOriginalGCN
from cf_explanation.cf_explainer import CFExplainerNew, GreedyCFExplainer, BFCFExplainer


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
        models[i] = {'data': datasets[i]}

        model_path = script_dir / f'../../models/sparse_gcn_3layer_{i}.pt'
        data = datasets[i]
        model = GCN(10, data.num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        y_pred = torch.argmax(model(data.x, data.edge_index), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models[i]['Sparse'] = model

        model_path = script_dir / f'../../models/gcn_3layer_{i}.pt'
        dense = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)
        dense.load_state_dict(torch.load(model_path))
        dense.eval()

        y_pred = torch.argmax(dense(data.x, data.norm_adj), dim=1)
        assert (y_pred == data.y).float().mean() > 0.8
        models[i]['Dense'] = dense

        wrapped = WrappedOriginalGCN(dense)
        models[i]['Wrapped'] = wrapped

    experiments = [*product(
        datasets.items(),
        ['Sparse', 'Wrapped'],
        [CFExplainerNew, GreedyCFExplainer, BFCFExplainer]
    )]

    # explain_new(model, x, edge_index, y, target, cf_model)
    results = []
    for (exp, data), model_type, explain_method in experiments:
        measurement = measure_function_time(
            partial(
                explain_new,
                model=models[exp][model_type],
                x=data.x,
                edge_index=data.edge_index,
                y=data.y,
                target=data.test_set,
                n_momentum = .9 if exp == 'syn1' else 0.0,
                cf_model=explain_method
            ),
            num_trials=1
        )
        measurement.insert(0, 'method', explain_method.__name__)
        measurement.insert(0, 'model', model_type)
        measurement.insert(0, 'dataset', exp)
        # measurement.to_pickle(script_dir / f'../../results/explain_time/{exp}{model_type}{explain_method.__name__}.pkl')
        results.append(measurement)

    for (exp, data), skip in product(datasets.items(), [True, False]):
        measurement = measure_function_time(partial(
                explain_original,
                model=models[exp]['Dense'],
                data=data,
                n_momentum = .9 if exp == 'syn1' else 0.0,
                skip=skip,
                target=data.test_set
            ),
            num_trials=1
        )
        skipped = 'Skip' if skip else 'Full'

        measurement.insert(0, 'method', 'Dense' + skipped)
        measurement.insert(0, 'model', 'Dense')
        measurement.insert(0, 'dataset', exp)
        # measurement.to_pickle(script_dir / f'../../results/explain_time/{exp}Dense{skipped}.pkl')
        results.append(measurement)

    pd.concat(results, ignore_index=True).to_pickle(script_dir / f'../../results/explain_time/result.pkl')


if __name__ == '__main__':
    main()
