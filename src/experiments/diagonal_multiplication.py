import torch

import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_degree_matrix
from utils.test_functions import load_dataset

def main():

    # model_weights = torch.load('../models/gcn_3layer_syn4.pt')

    # Load trained weights from GCNSynthetic
    # old_state_dict = torch.load('../models/gcn_3layer_syn4.pt')

    data = load_dataset('../../data/gnn_explainer/syn4.pickle', device='cpu')

    deg = get_degree_matrix(data.adj).detach()
    deg = deg ** (-1 / 2)
    deg[torch.isinf(deg)] = 0

    deg[0][1] = 2

    deg = torch.ones_like(deg) / deg.sum(dim=1) + torch.rand_like(deg) / 10

    for i in range(1000):
        _ = torch.mm(torch.mm(deg, data.adj + .0001), deg)





if __name__ == '__main__':
    main()