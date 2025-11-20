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


class CFExplainer(ExplainerAlgorithm):
    r"""
    ExplainerAlgorithm: Forward, explainer_config and model_config seem fun to use

    Return type Explanation might not be suited for CF example
    _loss_binary_classification etc not for cf objectives


    """
    coeffs = {
        'n_hid': 20,
        'dropout': 0.0,
        'num_layers': None,
        'num_classes': 2, #TODO where to put?
        'beta': 1,
    }

    def __init__(
        self,
        predictions: Tensor,
        epochs: int = 200,
        lr: float = 0.001,
        optimizer: str = 'SGD',
        n_momentum: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.predictions = predictions
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


        # if self.coeffs['num_layers'] is not None:
        # # new_index, edge_subset, sub_feat, sub_adj = self._get_neighbourhood(x, edge_index, index)
        #     nodes, edge_subset, mapping, mask = k_hop_subgraph(index,
        #                                                        self.coeffs['num_layers'],
        #                                                        edge_index,
        #                                                        relabel_nodes=True)

        # Instantiate CF model class, load weights from original model
        cf_model = GCNSyntheticPerturbEdgeWeight(
            model,
            index,
            x,
            edge_index,
            beta=self.coeffs['beta'],
            edge_additions=True
        )

        self.prediction = self.predictions[index]


        # cf_model.load_state_dict(model.eval().state_dict(), strict=False)
        cf_optimizer = self._initialize_cf_optimizer(cf_model)
        # y_pred = self.predictions[index]

        # Freeze weights from original model in cf_model
        # for name, param in cf_model.named_parameters():
        #     if name.endswith("weight") or name.endswith("bias"):
        #         param.requires_grad = False

        best_cf_example = []
        best_loss = np.inf

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



            if new_example and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total

        # TODO: Figure out the correct output type

        # TODO: Make example communication not shit
        print(f'node {index}: {len(best_cf_example)}')
        if 'storage' in self.coeffs and best_cf_example != []:
            self.coeffs['storage'][0] = True
            self.coeffs['storage'].append(best_cf_example[-1])
        else:
            self.coeffs['storage'][0] = False

        return Explanation(best_cf_example)

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

        loss_total.backward()
        clip_grad_norm(cf_model.parameters(), 2.0)
        cf_optimizer.step()

        # TODO: This should not be in the train function
        cf_stats = []
        if pred_same == 0.0:
            cf_stats = [
                #! self.node_idx.item(), # this one is actually used
                # index.item(),
                cf_adj.detach().numpy(),
                # A_x.detach().numpy(),
                # y_pred.item(),
                # y_pred_new.item(),
                # y_pred_new_actual.item(),
                #! self.sub_labels[index].numpy(),
                # A_x.shape[0],
                # loss_total.item(),
                # loss_pred.item(),
                loss_graph_dist.item(),
                y_new,
                self.prediction
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



# inherit from explainer algorithm pytorch geometric
# implement baseclass functions
class CFExplainerOriginal():
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(self, model, sub_adj, sub_feat, n_hid, dropout, sub_labels,
                 y_pred_orig, num_classes, beta, device):
        # super(CFExplainer,      # This does nothing?
        #       self).__init__()  # init superclass, inherits nothing
        self.model = model # trained gcnconv model
        self.model.eval()
        self.sub_adj = sub_adj # adjacency matrix to explain
        self.sub_feat = sub_feat # node features of subgraph
        self.n_hid = n_hid # dimension size in gnn
        self.dropout = dropout # dropout rate for model
        self.sub_labels = sub_labels # ground truth sub labels for nodes
        self.y_pred_orig = y_pred_orig # original prediction
        self.beta = beta # weight balancing loss vs graph distance loss
        self.num_classes = num_classes # number of classes in classification
        self.device = device # device (yknow)

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturbEdgeWeight(self.sub_feat.shape[1], n_hid,
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
    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum,
                num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

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
        print("{} CF examples for node_idx = {}".format(
            num_cf_examples, self.node_idx))
        print(" ")
        return (best_cf_example)

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

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
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
                self.node_idx.item(),
                self.new_idx.item(),
                cf_adj.detach().numpy(),
                self.sub_adj.detach().numpy(),
                self.y_pred_orig.item(),
                y_pred_new.item(),
                y_pred_new_actual.item(),
                self.sub_labels[self.new_idx].numpy(), self.sub_adj.shape[0],
                loss_total.item(),
                loss_pred.item(),
                loss_graph_dist.item()
            ]

        return (cf_stats, loss_total.item())
