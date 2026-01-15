import time
import torch
import numpy as np
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import dense_to_sparse
from utils.utils import get_degree_matrix
from .gcn_perturb_coo import GCNSyntheticPerturbEdgeWeight
from .gcn_perturb import GCNSyntheticPerturb

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


class CFExplainerNew:
    '''
    New version of CF Explainer to help in training perturb object for model
    that use COO format.
    '''
    def __init__(self, model, device='cpu', epochs=500, lr=0.1, n_momentum=0.0, eps=0.0):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.n_momentum = n_momentum
        self.eps = eps

        self.device = device

    def __call__(self, index, x, edge_index):
        ''' Find an edge mask that predicts a different label. '''
        cf_model = GCNSyntheticPerturbEdgeWeight(self.model, index, x, edge_index)
        optimizer = torch.optim.SGD([cf_model.P_vec], lr=self.lr,
                                    momentum=self.n_momentum,
                                    nesterov=(self.n_momentum > 0.0))

        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = cf_model.forward()
            new_prediction = torch.argmax(cf_model.forward_hard())
            loss, pred_is_same, _, candidate_adj = cf_model.loss(
                output, new_prediction
            )

            if not pred_is_same:
                return candidate_adj

            loss.backward()
            clip_grad_norm_([cf_model.P_vec], 2.0)
            optimizer.step()

        # No cf found
        return np.zeros(edge_index.shape[1], dtype=bool)


class BFCFExplainer(CFExplainerNew):
    '''
    Find counterfactuals by sampling initial grads to use as priority
    before running exhaustive search of every combination, up to max
    as given by the number of epochs.

    Produces distinct combinations by converting epoch number to binary
    eg: 5 = ...0000101 removes the first and third edges in ranking.

    Best CF is chosen first by distance by lowest score of original prediction.
    '''
    def __call__(self, index, x, edge_index):
        ''' Find an edge mask that predicts a different label. '''
        cf_model = GCNSyntheticPerturbEdgeWeight(self.model, index, x, edge_index,
                                                 beta=0.5)
        prediction = cf_model.original_class

        edges, scores = cf_model.score_edges(num_samples=10,
                                             eps=self.eps)

        # Modify scores to remove positive grads after negatives but before 0.0
        # This was found to increase fidelity
        scores[torch.where(scores > 0.0)] *= -.001
        ranking = sorted(list(enumerate(edges)), key=lambda x: scores[x[0]])

        best_distance = np.inf
        best_loss = np.inf
        best_cf_example = torch.zeros(edge_index.shape[1], dtype=bool)

        mask = torch.ones(edge_index.shape[1], dtype=bool)
        for i in range(1, self.epochs):
            binary_mask = bin(i)[2:]
            distance = binary_mask.count('1')

            # Zero out edges according to current epoch
            for n, bit in enumerate(binary_mask[::-1]):
                if n >= len(ranking):
                    break
                mask[ranking[n][1][0]] = bit != '1'
                mask[ranking[n][1][1]] = bit != '1'

            masked_edge_index = edge_index[:, mask]
            out = self.model(x, masked_edge_index)[index]
            new_prediction = torch.argmax(out)
            loss = out[prediction]

            # Break same-distance ties using loss for accuracy
            if new_prediction != prediction and \
                       distance <= best_distance and \
                                    loss < best_loss:
                best_distance = distance
                best_loss = loss

                best_cf_example = mask.clone()
                if distance == 1:
                    break
        return best_cf_example


class GreedyCFExplainer(CFExplainerNew):
    '''
    Find counterfactuals by sampling initial grads, then removes edges in order
    of importance.

    To avoid skewing mean distance of results, max distance may be passed before
    search is called off.
    '''

    def __call__(self, index, x, edge_index, max_distance=10):
        ''' Find an edge mask that predicts a different label. '''
        cf_model = GCNSyntheticPerturbEdgeWeight(self.model, index, x, edge_index,
                                                 beta=0.5)
        prediction = cf_model.original_class

        # Create ranking of most important edges
        edges, scores = cf_model.score_edges(num_samples=10,
                                                 eps=self.eps)

        # Modify scores to remove positive grads after negatives but before 0.0
        # This was found to increase fidelity
        scores[torch.where(scores > 0.0)] *= -.001
        ranking = sorted(list(enumerate(edges)), key=lambda x: scores[x[0]])

        mask = torch.ones(edge_index.shape[1], dtype=bool)
        for i in range(0, max_distance):
            if i >= len(ranking):
                break
            mask[ranking[i][1][0]] = False
            mask[ranking[i][1][1]] = False

            masked_edge_index = edge_index[:, mask]
            new_prediction = torch.argmax(self.model(x, masked_edge_index)[index])

            if new_prediction != prediction:
                return mask.clone()
        return torch.zeros(edge_index.shape[1], dtype=bool)


# # inherit from explainer algorithm pytorch geometric
# # implement baseclass functions
# class CFExplainer:
#     '''
#     CF Explainer class based on the original version in Ana's paper.

#     Changes:
#      - Output list changed to match current format in evaluate.py
#      - Print statements removed
#      - Return early when CF is found
#     '''

#     def __init__(self, model, data, index, n_hid, num_classes, dropout, beta, device):
#         self.model = model # trained gcnconv model
#         self.model.eval()

#         sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(index),
#                                                 data.edge_index,
#                                                 4,
#                                                 data.x,
#                                                 data.y)
#         self.new_idx = node_dict[int(index)]
#         self.node_idx = int(index)
#         self.node_dict = node_dict
#         self.y_pred_orig = torch.argmax(model(data.x, data.norm_adj), dim=1)[int(index)]

#         self.edge_index = data.edge_index
#         self.sub_adj = sub_adj
#         self.sub_feat = sub_feat
#         self.n_hid = n_hid
#         self.dropout = dropout
#         self.sub_labels = sub_labels
#         self.beta = beta
#         self.num_classes = num_classes
#         self.device = device

#         # Instantiate CF model class, load weights from original model
#         self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid,
#                                             n_hid, self.num_classes,
#                                             self.sub_adj, dropout, beta)

#         self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

#         # Freeze weights from original model in cf_model
#         for name, param in self.cf_model.named_parameters():
#             if name.endswith("weight") or name.endswith("bias"):
#                 param.requires_grad = False

#     ''' This is forward in ExplainerAlgorithm class for PyG '''
#     def explain(self, cf_optimizer, lr, n_momentum,
#                 num_epochs):
#         self.x = self.sub_feat
#         self.A_x = self.sub_adj
#         self.D_x = get_degree_matrix(self.A_x) # Never used?

#         if cf_optimizer == "SGD" and n_momentum == 0.0:
#             self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
#         elif cf_optimizer == "SGD" and n_momentum != 0.0:
#             self.cf_optimizer = optim.SGD(self.cf_model.parameters(),
#                                           lr=lr,
#                                           nesterov=True,
#                                           momentum=n_momentum)
#         elif cf_optimizer == "Adadelta":
#             self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(),
#                                                lr=lr)

#         best_cf_example = []
#         best_loss = np.inf
#         num_cf_examples = 0
#         for epoch in range(num_epochs):
#             new_example, loss_total = self.train(epoch)
#             if new_example != [] and loss_total < best_loss:
#                 best_cf_example.append(new_example)
#                 best_loss = loss_total
#                 num_cf_examples += 1

#                 new_example[-1] = convert_subadj_to_full_mask(self.node_dict,
#                                                               new_example[-1],
#                                                               self.edge_index)
#                 break
#         print("{} CF examples for node_idx = {}".format(
#             num_cf_examples, self.node_idx))
#         print(" ")
#         if best_cf_example == []:
#             return [self.y_pred_orig.item(), np.nan, torch.zeros(self.edge_index.shape[1], dtype=bool)]
#         return best_cf_example[0]

#     def train(self, epoch):
#         t = time.time()
#         self.cf_model.train()
#         self.cf_optimizer.zero_grad()

#         # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
#         # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
#         output = self.cf_model.forward(self.x, self.A_x)
#         output_actual, self.P = self.cf_model.forward_prediction(self.x)

#         # Need to use new_idx from now on since sub_adj is reindexed
#         y_pred_new = torch.argmax(output[self.new_idx])
#         y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

#         # print(output[self.new_idx])
#         # print(self.cf_model.P_vec)

#         # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
#         loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
#             output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
#         loss_total.backward()
#         clip_grad_norm_(self.cf_model.parameters(), 2.0)
#         self.cf_optimizer.step()

#         cf_stats = []
#         if y_pred_new_actual != self.y_pred_orig:
#             cf_stats = [
#                 y_pred_new_actual.item(),
#                 loss_graph_dist.item(),
#                 self.cf_model.P * self.sub_adj
#             ]

#         return (cf_stats, loss_total.item())


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph

    Changes made from original:
     - changed return object to match expected format in evaluation.py
     - eary break when first counterfactual found
     - removed print statements
    """
    def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
                  sub_labels, y_pred_orig, num_classes, beta, device):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.device = device

        self.edge_index = dense_to_sparse(sub_adj)[0]

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
                                            self.num_classes, self.sub_adj, dropout, beta)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
                return new_example

        return [self.y_pred_orig.item(), np.nan,
                torch.zeros(self.edge_index.shape[1], dtype=bool)]

    def train(self, epoch):
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)
        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, _, loss_graph_dist, _ = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [
                y_pred_new_actual.item(),
                loss_graph_dist.item(),
                self.cf_model.P * self.sub_adj
            ]

        return (cf_stats, loss_total.item())
