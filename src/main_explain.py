from pathlib import Path
import argparse

import torch

from gcn import GCNSynthetic
from gcn_sparse import GCN
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

    model_path = script_dir / f'../models/gcn_3layer_{args.exp}.pt'
    if args.cf_method == 'original':
        model = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                                nclass=len(data.y.unique()), dropout=0.0)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        result = explain_original(model, data, lr=args.lr,
                                  n_momentum=args.momentum,
                                  epochs=args.epochs)
        result.to_pickle(f"../results/{args.dst}.pkl")
        return
    if args.cf_method == 'cf_transposed':
        model = GCN(data.x.shape[1], data.num_classes)
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
    result.to_pickle(f"../results/{args.dst}.pkl")


def explain_sparse(args, data):
    script_dir = Path(__file__).parent
    model_path = script_dir / f'../models/sparse_gcn_3layer_{args.exp}.pt'

    cf_model = cf_explainers.get(args.cf_method, CFExplainerNew)
    if cf_model == CFExplainer:
        raise NotImplementedError(
            'Sparse classifier cannot be explained in original framework. \
             Leave cf_method unassigned to default to CFExplainerNew'
        )

    model = GCN(data.x.shape[1], data.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    result = explain_new(
        model, data.x, data.edge_index, data.y, data.test_set,
        cf_explainers[args.cf_method], epochs=args.epochs, lr=args.lr,
        n_momentum=args.momentum, eps=args.eps
    )
    result.to_pickle(f"../results/{args.dst}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='syn1')
    parser.add_argument('--dst', type=str, default='results')
    parser.add_argument('--cf_method', type=str, default='cf_wrapped')
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    data = load_dataset(graph_path, 'cpu')

    if args.sparse:
        explain_sparse(args, data)
    else:
        explain_original_experiment(args, data)


if __name__ == '__main__':
    main()
