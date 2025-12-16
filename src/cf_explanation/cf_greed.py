# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

from .cf_explainer import CFExplainer

import torch
from .gcn_edge_perturb import GCNSyntheticPerturbEdgeWeight


class GreedyCFExplainer(CFExplainer):
    r"""
    
    """
    def _find_cf(self, model, index, x, edge_index):
        """ Remove edges sequentually in order from most to least important """
        self.prediction = torch.argmax(model(x, edge_index)[index])

        cf_model = GCNSyntheticPerturbEdgeWeight(model, index, x, edge_index,
                                                    beta=self.coeffs['beta'])

        edge_scores = cf_model.sample_edge_importance(10,
                                                      self.coeffs['eps'],
                                                      self.coeffs['noise'])
        ranking = sorted(list(enumerate(edge_index.T)), key=lambda x: -edge_scores[x[0]])

        best_cf_example = []
        mask = torch.tensor(torch.ones(edge_index.shape[1]), dtype=bool)
        for i in range(1, 5):
            mask[ranking[i][0]] = False

            masked_edge_index = edge_index[:, mask]
            new_prediction = torch.argmax(model(x, masked_edge_index)[index])

            if new_prediction != self.prediction:
                best_cf_example.append([new_prediction.item(), i, mask.clone()])
                break
        return best_cf_example