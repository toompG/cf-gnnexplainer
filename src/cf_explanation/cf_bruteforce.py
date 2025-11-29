# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

from typing import List, Dict, Optional, Tuple, Union

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


def find_cf(model, index, x, edge_index, epochs, beta=.5):
    cf_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index,
                                              beta=beta)

    prediction = torch.argmax(model(x, edge_index)[index])

    scores = cf_model.compute_edge_importance_gradients(num_samples=10, eps=.2, noise=.05)
    ranking = sorted(list(enumerate(edge_index.T)),
                        key=lambda x: -scores[x[0]])

    best_distance = np.inf
    best_loss = np.inf
    best_cf_example = []

    mask = torch.tensor(torch.ones(edge_index.shape[1]), dtype=bool)
    for i in range(1, epochs):
        binary_mask = bin(i)[2:]
        distance = binary_mask.count('1')

        for bit, j in zip(binary_mask[::-1], ranking):
            mask[j[0]] = bit != '1'

        masked_edge_index = edge_index[:, mask]
        out = model(x, masked_edge_index)[index]
        new_prediction = torch.argmax(out)
        loss = out[prediction]

        if new_prediction != prediction and distance <= best_distance and loss < best_loss:
            best_distance = distance
            best_loss = loss

            best_cf_example.append([new_prediction.item(), distance, mask.clone()])
            if distance == 1:
                break
    return best_cf_example



class BFCFExplainer(ExplainerAlgorithm):
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
        lr: float = 0.001,
        optimizer: str = 'SGD',
        n_momentum: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # self.predictions = predictions
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.n_momentum = n_momentum

        self.coeffs.update(kwargs)

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
        # TODO: test if subgraphs are substantially better
        best_cf_example = find_cf(model, index, x, edge_index, self.epochs)
        self.prediction = torch.argmax(model(x, edge_index)[index])

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

    def _initialize_cf_optimizer(
            self, cf_model: GCNSyntheticPerturbEdgeWeight
        ) -> optim.Optimizer:
        if self.optimizer == 'SGD':
            return optim.SGD(
                [cf_model.edge_weight_params],
                self.lr,
                self.n_momentum,
                nesterov=(self.n_momentum != 0.0),
            )
        elif self.optimizer == 'Adadelta':
            return optim.Adadelta(
                [cf_model.edge_weight_params],
                self.lr,
            )
        else:
            raise ValueError('Invalid optimizer value')
