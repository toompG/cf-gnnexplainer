from pathlib import Path
import argparse

import torch

from gcn import GCNSynthetic
from wrapper import WrappedOriginalGCN
from cf_explanation.cf_explainer import CFExplainerNew, CFExplainer, \
                                        GreedyCFExplainer, BFCFExplainer
from utils.test_functions import load_dataset, explain_new, explain_original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='syn1')
    parser.add_argument('--dst', type=str, default='results')
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--cf', type=str, default='cf')

    args = parser.parse_args()
    if args.cf == 'cf':
        cf_model = CFExplainerNew
    elif args.cf == 'greedy':
        cf_model = GreedyCFExplainer
    elif args.cf == 'bf':
        cf_model = BFCFExplainer
    elif args.cf == 'original':
        cf_model = CFExplainer
    else:
        raise AttributeError('Incorrect cf specified: use cf, greedy, bf or original')

    script_dir = Path(__file__).parent
    graph_path = script_dir / f'../data/gnn_explainer/{args.exp}.pickle'
    model_path = script_dir / f'../models/gcn_3layer_{args.exp}.pt'


    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data, dataset = get_dataset(nodes=n_nodes_graph, motifs = n_motifs, device=device)

    data = load_dataset(graph_path, device)

    print(len(data.y.unique()))
    submodel = GCNSynthetic(nfeat=data.x.shape[1], nhid=20, nout=20,
                            nclass=len(data.y.unique()), dropout=0)

    submodel.load_state_dict(torch.load(model_path))
    submodel.eval()

    if args.cf == 'original':
        explain_original(submodel, data,
                         lr=args.lr, n_momentum=args.momentum, epochs=args.epochs, dst=args.dst)
        return
    elif args.cf == 'cf':
        model = WrappedOriginalGCN(submodel).eval()
    else:
        model = WrappedOriginalGCN(submodel)

    output = model(data.x, data.edge_index)
    output_real = submodel(data.x, data.norm_adj)

    y_pred_orig = torch.argmax(output, dim=1)
    y_pred_orig_real = torch.argmax(output_real, dim=1)

    train_accuracy = (y_pred_orig == data.y).float().mean()
    train_accuracy_real = (y_pred_orig_real == data.y).float().mean()

    print(f'Wrapped accuracy: {train_accuracy}\nOriginal accuracy: {train_accuracy_real}')
    print(f'Difference: {torch.sum(output - output_real)}')

    result = explain_new(model, data.x, data.edge_index, data.test_set, data.y, cf_model,
                n_hops=4, device='cpu', epochs=args.epochs, lr=args.lr,
                n_momentum=args.momentum, eps=args.eps)

    result.to_pickle(f"../results/{args.dst}.pkl")


if __name__ == '__main__':
    main()
