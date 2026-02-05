"""
main_explain.py

Explain nodes in test set using different classifiers and cf-methods. Does
not support models trained with gcn_sparse.py

Experimental setup may be found in scripts directory.

Arguments
    --exp: dataset
    --model: model to explain. defaults to models from original paper if empty.
    --dst: location to store results
    --sparse: bool, default=False)
    --cf_method: method choice
        Dense gradients: [original, cf_wrapped, cf_transposed, greedy, bf]
        Sparse gradients: [cf, greedy, bf]
    --lr: learning rate
    --momentum: nesterov momentum
    --epochs: number of epochs
    --eps: edge weight noise
    --seed: random seed
"""


from pathlib import Path
import argparse

import torch

from gcn import GCNSynthetic
from gcn_sparse import GCNReuseNormalisation
from wrapper import WrappedOriginalGCN, GCNConvGCNSynthetic
from cf_explanation.cf_explainer import CFExplainerNew, CFExplainer, \
                                        GreedyCFExplainer, BFCFExplainer
from utils.test_functions import load_dataset, explain_new, explain_original, \
                                 load_sparse_dense_weights


cf_explainers = {
    'original': CFExplainer,
    'cf_wrapped': CFExplainerNew,
    'cf_transposed': CFExplainerNew,
    'greedy': GreedyCFExplainer,
    'bf': BFCFExplainer
}


def explain_original_experiment(args, data):
    script_dir = Path(__file__).parent

    cf_model = cf_explainers.get(args.cf_method, None)
    if cf_model is None:
        raise AssertionError('cf_method must be original, cf_wrapped, \
                             cf_transposed, greedy, bf')

    model_path = script_dir / f'../models/{args.model}'
    if args.cf_method == 'original':
        model = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                                nclass=len(data.y.unique()), dropout=0.0)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        result = explain_original(model, data, lr=args.lr,
                                  n_momentum=args.momentum,
                                  epochs=args.epochs)
        result.to_pickle(script_dir / f"../results/{args.dst}.pkl")
        return
    if args.cf_method == 'cf_transposed':
        model = GCNConvGCNSynthetic(data.x.shape[1], data.num_classes)
        load_sparse_dense_weights(model, model_path)
    else:
        submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                                nclass=len(data.y.unique()), dropout=0.0)
        submodel.load_state_dict(torch.load(model_path))
        submodel.eval()
        model = WrappedOriginalGCN(submodel)

    model.eval()
    result = explain_new(
        model, data.x, data.edge_index, data.y, data.test_set,
        cf_explainers[args.cf_method], epochs=args.epochs, lr=args.lr,
        n_momentum=args.momentum, eps=args.eps
    )
    result.to_pickle(script_dir / f"../results/{args.dst}.pkl")


def explain_sparse(args, data):
    script_dir = Path(__file__).parent
    model_path = script_dir / f'../models/{args.model}'

    cf_model = cf_explainers.get(args.cf_method, CFExplainerNew)
    if cf_model == CFExplainer:
        raise NotImplementedError(
            'Sparse classifier cannot be explained in original framework. \
             Leave cf_method unassigned to default to CFExplainerNew'
        )

    model = GCNReuseNormalisation(data.x.shape[1], data.num_classes)

    try:
        sparse_path = script_dir / f'../models/{args.model}'
        model.load_state_dict(torch.load(sparse_path))
    except RuntimeError:
        load_sparse_dense_weights(model, model_path)

    model.eval()

    result = explain_new(
        model, data.x, data.edge_index, data.y, data.test_set,
        cf_explainers[args.cf_method], epochs=args.epochs, lr=args.lr,
        n_momentum=args.momentum, eps=args.eps
    )
    result.to_pickle(script_dir / f"../results/{args.dst}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='syn1')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dst', type=str, default='results')
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--cf_method', type=str, default='cf_wrapped')
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=20)

    args = parser.parse_args()

    if args.model == '':
        args.model = f'gcn_3layer_{args.exp}.pt'

    script_dir = Path(__file__).parent
    results_dir = script_dir / "../results"
    results_dir.mkdir(parents=True, exist_ok=True)

    graph_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    data = load_dataset(graph_path, 'cpu')

    if args.sparse:
        explain_sparse(args, data)
    else:
        explain_original_experiment(args, data)


if __name__ == '__main__':
    main()
