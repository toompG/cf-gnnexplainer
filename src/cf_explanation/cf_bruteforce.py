# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

from typing import List, Dict, Optional, Tuple, Union

from .cf_explainer import CFExplainer

import time
import torch
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm
from utils.utils import get_degree_matrix
from .gcn_edge_perturb import GCNSyntheticPerturbEdgeWeight
from utils.utils import normalize_adj

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from torch_geometric.explain.config import ExplanationType


class BFCFExplainer(CFExplainer):
    r"""
    ExplainerAlgorithm: Forward, explainer_config and model_config seem fun to use

    Return type Explanation might not be suited for CF example
    _loss_binary_classification etc not for cf objectives


    """
    def _find_cf(self, model, index, x, edge_index):
        '''
        Find counterfactuals by creating a ranking of important edges for the
        node index and removing combinations of them up to self.epochs.

        Combinations are found by converting the current epoch to binary where
        the nth bit removes the n most important edge according to the ranking.
        ie ...000101 removes the 1st and 3rd highest ranked edges.

        '''
        cf_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index,
                                                 beta=self.coeffs['beta'])

        # create ranking of most important edges
        scores = cf_model.sample_edge_importance(num_samples=10,
                                                 eps=self.coeffs['eps'],
                                                 noise=self.coeffs['noise'])
        ranking = sorted(list(enumerate(edge_index.T)),
                            key=lambda x: -scores[x[0]])

        best_distance = np.inf
        best_loss = np.inf
        best_cf_example = []

        mask = torch.tensor(torch.ones(edge_index.shape[1]), dtype=bool)
        for i in range(1, self.epochs):
            binary_mask = bin(i)[2:]
            distance = binary_mask.count('1')

            # Zero out edges according to current epoch
            for bit, j in zip(binary_mask[::-1], ranking):
                mask[j[0]] = bit != '1'

            masked_edge_index = edge_index[:, mask]
            out = model(x, masked_edge_index)[index]
            new_prediction = torch.argmax(out)
            loss = out[self.prediction]

            # break same-distance ties using loss for accuracy
            if new_prediction != self.prediction and \
                       distance <= best_distance and \
                                    loss < best_loss:
                best_distance = distance
                best_loss = loss

                best_cf_example.append([new_prediction.item(), distance, mask.clone()])
                if distance == 1:
                    break
        return best_cf_example