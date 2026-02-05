"""
analyse_max_err.py

small script to analyse at which bit error occurs for the forward pass.
highest order bit error is shown for value where error magnitude is highest.
it is possible that points with a lower exponent have greater error in
mantissa, however.

"""

import sys
import os
import argparse

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, load_sparse_dense_weights
from gcn import GCNSynthetic
from gcn_sparse import GCNReuseNormalisation
from wrapper import GCNConvGCNSynthetic, WrappedOriginalGCN


def parse_args():
    parser = argparse.ArgumentParser(description='Compare GCN model formats')

    parser.add_argument('--exp', type=str, required=True,
                        choices=['syn1', 'syn2', 'syn4', 'syn5'],
                        help='Dataset to use (syn1, syn2, syn4, or syn5)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['wrapped', 'gcn', 'gcnmm'],
                        help='Interfacing method to use')
    parser.add_argument('--verbose', type=bool, default=False)

    return parser.parse_args()


def analyze_max_error(out_dd, out_sd):
    """
    Find max error location and analyze bit-level precision
    """
    error = (out_dd - out_sd).abs()
    max_error_idx = error.argmax()
    max_error_value = error.flatten()[max_error_idx]

    # Get the actual values at max error location
    if error.dim() == 2:
        row = max_error_idx // error.shape[1]
        col = max_error_idx % error.shape[1]
        val_dd = out_dd[row, col]
        val_sd = out_sd[row, col]
        print(f"Max error at index: [{row}, {col}]")
    else:
        val_dd = out_dd.flatten()[max_error_idx]
        val_sd = out_sd.flatten()[max_error_idx]
        print(f"Max error at index: {max_error_idx.item()}")

    print(f"Absolute max error: {max_error_value:.17e}")
    print(f"Dense value:  {val_dd:.17e}")
    print(f"Sparse value: {val_sd:.17e}")

    # Calculate relative error
    magnitude = max(abs(val_dd), abs(val_sd))
    if magnitude > 0:
        relative_error = max_error_value / magnitude
        print(f"Relative error: {relative_error:.17e}")

        import math
        if relative_error > 0:
            bit_order = -math.ceil(math.log2(relative_error.item()))
            print(f"Max error starts at bit: ~{bit_order:.1f} (out of 53 significand bits)")
            print(f"Significant digits preserved: ~{bit_order * math.log10(2):.1f} decimal digits")


def main():
    args = parse_args()

    method_map = {
        'gcnmm': GCNConvGCNSynthetic,
        'gcn': GCNReuseNormalisation,
        'wrapped': WrappedOriginalGCN
    }

    to_test = method_map[args.method]

    # Dataset configuration
    data_path = f'../../data/gnn_explainer/{args.exp}.pickle'
    model_path = f'../../models/gcn_3layer_{args.exp}.pt'

    print(f"Using: {args.exp}, {args.method}")

    torch.set_default_dtype(torch.float64)

    data = load_dataset(data_path, device='cpu')
    data.norm_adj = data.norm_adj.double()
    data.x = data.x.double()

    dense_dense = GCNSynthetic(10, 20, 20, data.num_classes, 0.0)
    dense_dense.load_state_dict(torch.load(model_path))
    dense_dense.eval()

    if to_test == WrappedOriginalGCN:
        sparse_dense = WrappedOriginalGCN(dense_dense)
    else:
        sparse_dense = to_test(10, data.num_classes)
        load_sparse_dense_weights(sparse_dense, model_path)

    sparse_dense.eval()

    # index = torch.tensor([400, 500, 600])
    index = data.test_set

    out_dd = dense_dense(data.x, data.norm_adj)
    out_sd = sparse_dense(data.x, data.edge_index)

    # Calculate error from forward call
    print('Original weights')
    print(f'Total err: {(out_dd - out_sd).sum():.2e}')
    print(f'Forward max err:   {abs(out_dd - out_sd).max():.2e}')
    print(f'Forward mean err:  {abs(out_dd - out_sd).mean():.2e}')
    analyze_max_error(out_dd, out_sd)


if __name__ == '__main__':
    main()