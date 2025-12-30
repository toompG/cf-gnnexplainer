# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

from typing import List, Dict, Optional, Tuple, Union

import time
import torch
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm
from utils.utils import get_degree_matrix
from .gcn_vectorized import GCNSyntheticPerturbEdgeWeight
from .gcn_perturb import GCNSyntheticPerturb
from utils.utils import get_neighbourhood

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from torch_geometric.explain.config import ExplanationType


class CFExplainer(ExplainerAlgorithm):
    r"""
    ExplainerAlgorithm: Forward, explainer_config and model_config seem fun to use

    Return type Explanation might not be suited for CF example
    _loss_binary_classification etc not for cf objectives
    """
    coeffs = {
        'dropout': 0.0,
        'num_layers': None,
        'beta': .5,
        'eps': 1.0,
        'noise': 0.0
    }

    def __init__(
        self,
        # predictions: Tensor,
        epochs: int = 200,
        lr: float = 0.1,
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
            self.index = Tensor(range(len(x)))
        else:
            self.index = index
        self.edge_index = edge_index

        model.eval()
        self.prediction = torch.argmax(model(x, edge_index), dim=1)[self.index]

        # Extract k-hop subgraph
        sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            int(self.index),
            4,
            edge_index,
            relabel_nodes=True
        )

        self.sub_nodes = sub_nodes
        self.sub_to_full_edge_mask = edge_mask
        self.full_edge_count = edge_index.shape[1]

        sub_x = x[sub_nodes]
        sub_index = mapping if mapping.dim() > 0 else mapping.unsqueeze(0)

        best_cf_example = self._find_cf(model, int(sub_index), sub_x, sub_edge_index)
        if best_cf_example != []:
            best_cf_example = self._map_to_full_graph(best_cf_example, sub_edge_index, mapping)

        if 'storage' in self.coeffs:
            self._store_result(best_cf_example)

        return Explanation(best_cf_example)

    def _map_to_full_graph(self, cf_explanation, sub_edge_index, mapping):
        sub_edge_mask = cf_explanation[-1][2]

        full_edge_mask = torch.ones(self.full_edge_count, dtype=torch.bool)
        subgraph_edge_positions = torch.where(self.sub_to_full_edge_mask)[0]

        for sub_idx, full_idx in enumerate(subgraph_edge_positions):
            full_edge_mask[full_idx] = bool(sub_edge_mask[sub_idx])

        cf_explanation[-1][2] = np.array(full_edge_mask)
        return cf_explanation

    def _store_result(self, best_cf_example):
        # TODO : Run from example function, rebuild all values from return explanation
        print(f'node {self.index}: {len(best_cf_example)}')
        if best_cf_example != []:
            self.coeffs['storage'][0] = True
            self.coeffs['storage'].append(best_cf_example[-1])
        else:
            self.coeffs['storage'][0] = True
            result = [self.prediction.item(),
                      float('nan'),
                      np.zeros(self.edge_index.shape[1], dtype=bool)]
            self.coeffs['storage'].append(result)

    def _find_cf(self, model, index, x, edge_index):
        cf_model = GCNSyntheticPerturbEdgeWeight(
            model,
            index,
            x,
            edge_index,
            beta=self.coeffs['beta'],
        )

        # cf_model.original_class = torch.argmax(model(x, edge_index)[index])

        cf_model.reset_parameters(self.coeffs['eps'], 0.0)
        cf_optimizer = self._initialize_cf_optimizer(cf_model)

        best_cf_example = []
        best_distance = np.inf

        for epoch in range(self.epochs):
            new_example, loss_total = self.cf_train(
                cf_model,
                cf_optimizer,
                # x=sub_feat,
                # A_x=edge_subset,
                # y_pred=y_pred,
                # target=target,
                # index=new_index,
                # **kwargs
            )

            if new_example and new_example[1] < best_distance:
                assert(new_example[1] != 0)
                best_cf_example.append(new_example)
                best_distance = new_example[1]

                # skip when optimal cf found
                return best_cf_example
        return []

    def cf_train(
        self,
        cf_model: GCNSyntheticPerturbEdgeWeight,
        cf_optimizer: optim.Optimizer,
        # x: Tensor,
        # A_x: Tensor,
        # y_pred: Tensor,
        # *,
        # target: Tensor,
        # index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r''' e@@@@@@@@@@@@@@@
            @@@""""""""""
           @" ___ ___________
          II__[w] | [i] [z] |
         {======|_|~~~~~~~~~|
        /oO--000""`-OO---OO-'''

        cf_model.train()
        cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = cf_model.forward()
        y_new = torch.argmax(cf_model.forward_hard())

        #// Need to use new_idx from now on since sub_adj is reindexed

        # assert(y_pred_real == y_pred_new_actual)


        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, pred_same, loss_graph_dist, cf_adj = cf_model.loss(
            output,
            y_new
        )

        # print(loss_total)
        # print(output, loss_total)

        loss_total.backward()
        # clip_grad_norm(cf_model.parameters(), 2.0)
        cf_optimizer.step()

        # TODO: This should not be in the train function
        cf_stats = []
        if pred_same == 0:
            cf_stats = [
                y_new.item(),
                loss_graph_dist.item(),
                cf_adj.detach().numpy()
            ]

        return (cf_stats, loss_total.item())

    def supports(self):
        # TODO: check values in Explainer
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.model:
            # logging.error(f"'{self.__class__.__name__}' only supports "
            #               f"model explanations "
            #               f"got (`explanation_type={explanation_type.value}`)")
            return False

        # task_level = self.model_config.task_level
        # if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
        #     logging.error(f"'{self.__class__.__name__}' only supports "
        #                   f"node-level or graph-level explanations "
        #                   f"got (`task_level={task_level.value}`)")
        #     return False

        # node_mask_type = self.explainer_config.node_mask_type
        # if node_mask_type is not None:
        #     logging.error(f"'{self.__class__.__name__}' does not support "
        #                   f"explaining input node features "
        #                   f"got (`node_mask_type={node_mask_type.value}`)")
        #     return False

        return True

    def _get_neighbourhood(
            self, x: Tensor, edge_index: Tensor, index: Tensor, num_hops=4
        ) -> Tuple[Tensor, Tensor, Tensor]:
        nodes, edge_subset, new_index, edge_mask = k_hop_subgraph(
            index.item(),
            num_hops,
            edge_index,
            relabel_nodes=True
        )

        # Get relabelled subset of edges
        edge_subset_relabel = subgraph(nodes, edge_index, relabel_nodes=True)
        # TODO: Why do we need to relabel?
        #? print(edge_subset == edge_index[:, edge_mask])
        #? print(edge_subset_relabel[0])
        #? sub_adj = to_dense_adj(edge_subset).squeeze()
        #? sub_labels = labels[nodes]
        sub_feat = x[nodes, :]
        sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()

        return new_index, edge_subset, sub_feat, sub_adj

    def _initialize_cf_optimizer(
            self, cf_model: GCNSyntheticPerturbEdgeWeight
        ) -> optim.Optimizer:
        if self.optimizer == 'SGD':
            return optim.SGD(
                [cf_model.P_vec],
                self.lr,
                self.n_momentum,
                nesterov=(self.n_momentum != 0.0),
            )
        elif self.optimizer == 'Adadelta':
            return optim.Adadelta(
                [cf_model.P_vec],
                self.lr,
            )
        else:
            raise ValueError('Invalid optimizer value')


def convert_subadj_to_full_mask(node_dict, sub_adj, edge_index):
    vals = []

    for i in edge_index.T:
        v1 = i[0].item()
        v2 = i[1].item()

        if v1 not in node_dict or v2 not in node_dict:
            vals.append(True)
            continue
        v1 = node_dict[v1]
        v2 = node_dict[v2]
        vals.append(bool(sub_adj[v1][v2]))

    return torch.tensor(vals)


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
        scores = cf_model.sample_edge_importance(num_samples=1,
                                                 eps=self.coeffs['eps'],
                                                 noise=self.coeffs['noise'])
        ranking = sorted(list(enumerate(cf_model.edge_index.T)),
                              key=lambda x: -scores[x[0]])
        print(*zip(ranking, sorted(-scores)), sep='\n')

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


# inherit from explainer algorithm pytorch geometric
# implement baseclass functions
class CFExplainerOriginal():
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(self, model, data, index, n_hid, num_classes, dropout, beta, device):
        # super(CFExplainer,      # This does nothing?
        #       self).__init__()  # init superclass, inherits nothing
        self.model = model # trained gcnconv model
        self.model.eval()

        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(index),
                                                data.edge_index,
                                                4,
                                                data.x,
                                                data.y)
        self.new_idx = node_dict[int(index)]
        self.node_idx = int(index)
        self.node_dict = node_dict
        self.y_pred_orig = torch.argmax(model(data.x, data.norm_adj), dim=1)[int(index)]

        self.edge_index = data.edge_index
        self.sub_adj = sub_adj # adjacency matrix to explain
        self.sub_feat = sub_feat # node features of subgraph
        self.n_hid = n_hid # dimension size in gnn
        self.dropout = dropout # dropout rate for model
        self.sub_labels = sub_labels # ground truth sub labels for nodes
        # self.y_pred_orig = y_pred_orig # original prediction
        self.beta = beta # weight balancing loss vs graph distance loss
        self.num_classes = num_classes # number of classes in classification
        self.device = device # device (yknow)

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid,
                                            n_hid, self.num_classes,
                                            self.sub_adj, dropout, beta)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     print("orig model requires_grad: ", name, param.requires_grad)
        # for name, param in self.cf_model.named_parameters():
        #     print("cf model requires_grad: ", name, param.requires_grad)

    ''' This is forward in ExplainerAlgorithm class for PyG '''
    def explain(self, cf_optimizer, lr, n_momentum,
                num_epochs):
        # self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x) # Never used?

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(),
                                          lr=lr,
                                          nesterov=True,
                                          momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(),
                                               lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

                new_example[-1] = convert_subadj_to_full_mask(self.node_dict,
                                                              new_example[-1],
                                                              self.edge_index)
                break
        print("{} CF examples for node_idx = {}".format(
            num_cf_examples, self.node_idx))
        print(" ")
        if best_cf_example == []:
            return [self.y_pred_orig.item(), np.nan, np.zeros(self.edge_index.shape[1], dtype=bool)]
        return best_cf_example[0]

    def train(self, epoch):
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)

        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # print(output[self.new_idx])
        # print(self.cf_model.P_vec)

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        # print(output[self.new_idx], loss_total)
        # print(loss_total)

        # print('Node idx: {}'.format(self.node_idx),
        #       'New idx: {}'.format(self.new_idx),
        #       'Epoch: {:04d}'.format(epoch + 1),
        #       'loss: {:.4f}'.format(loss_total.item()),
        #       'pred loss: {:.4f}'.format(loss_pred.item()),
        #       'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        # print(
        #     'Output: {}\n'.format(output[self.new_idx].data),
        #     'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
        #     'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(
        #         self.y_pred_orig, y_pred_new, y_pred_new_actual))
        # print(" ")
        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [
                # self.node_idx.item(),
                # self.new_idx.item(),
                # cf_adj.detach().numpy(),
                # self.sub_adj.detach().numpy(),
                # self.y_pred_orig.item(),
                # y_pred_new.item(),
                y_pred_new_actual.item(),
                # self.sub_labels[self.new_idx].numpy(), self.sub_adj.shape[0],
                # loss_total.item(),
                # loss_pred.item(),
                loss_graph_dist.item(),
                self.cf_model.P * self.sub_adj
            ]

        return (cf_stats, loss_total.item())
