# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

from typing import List, Dict, Optional, Tuple, Union

import time
import torch
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from utils.utils import get_degree_matrix
from .gcn_edge_perturb import GCNSyntheticPerturbEdgeWeight
from utils.utils import normalize_adj

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from torch_geometric.explain.config import ExplanationType


class GreedyCFExplainer(ExplainerAlgorithm):
    r"""
    ExplainerAlgorithm: Forward, explainer_config and model_config seem fun to use

    Return type Explanation might not be suited for CF example
    _loss_binary_classification etc not for cf objectives


    """
    coeffs = {
        'dropout': 0.0,
        'num_layers': None,
        'beta': .5,
    }

    def __init__(
        self,
        # predictions: Tensor,
        epochs: int = 200,
        optimizer: str = 'SGD',
        n_momentum: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # self.predictions = predictions
        self.epochs = epochs
        self.optimizer = optimizer
        self.n_momentum = n_momentum

        self.coeffs.update(kwargs)

    def find_cf(self, model, index, x, edge_index):
        self.prediction = torch.argmax(model(x, edge_index)[index])

        cf_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index,
                                                 beta=self.coeffs['beta'])

        edge_scores = cf_model.compute_edge_importance_gradients(num_samples=10, eps=.2, noise=.05)
        ranking = sorted(list(enumerate(edge_index.T)), key=lambda x: -edge_scores[x[0]])

        best_cf_example = []
        mask = torch.tensor(torch.ones(edge_index.shape[1]), dtype=bool)
        for i in range(1, 13):
            mask[ranking[i][0]] = False

            masked_edge_index = edge_index[:, mask]
            new_prediction = torch.argmax(model(x, masked_edge_index)[index])

            if new_prediction != self.prediction:
                best_cf_example.append([new_prediction.item(), i, mask.clone()])
                break
        return best_cf_example

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if index is None:
            index = Tensor(range(len(x)))

        model.eval()
        best_cf_example = self.find_cf(model, index, x, edge_index)

        # TODO: Figure out the correct output type

        # TODO: Make example communication not shit
        print(f'node {index}: {len(best_cf_example)}')
        if 'storage' in self.coeffs:
            if best_cf_example != []:
                self.coeffs['storage'][0] = True
                self.coeffs['storage'].append(best_cf_example[-1])
            else:
                self.coeffs['storage'][0] = True
                result = [self.prediction.item(),
                         float('nan'),
                          np.zeros(edge_index.shape[1], dtype=bool)
                ]
                self.coeffs['storage'].append(result)

        return Explanation(best_cf_example)

    def supports(self):
        # TODO: check values in Explainer
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.model:
            # logging.error(f"'{self.__class__.__name__}' only supports "
            #               f"model explanations "
            #               f"got (`explanation_type={explanation_type.value}`)")
            return False
        return True
