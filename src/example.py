import torch
import torch.nn.functional as F

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GraphMaskExplainer
from torch_geometric.utils import dense_to_sparse
from utils.utils import get_neighbourhood

from cf_explanation.cf_explainer import CFExplainer, CFExplainerOriginal
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    ''' Standard GCN '''
    def __init__(self, ds):
        super().__init__()
        self.conv1 = GCNConv(ds.num_features, 16)
        self.conv2 = GCNConv(16, ds.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_dataset(device, nodes = 300, motifs = 20):
    ''' Create BA_Graph dataset '''
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=nodes, num_edges=5),
        motif_generator='house',
        num_motifs= motifs
    )

    # Set node features to all ones
    data = dataset[0].to(device)
    data.x = torch.ones((data.num_nodes, 1), device=device)

    # Set ground truth to boolean prediction
    data.y[data.y > 0] = 1

    train_size = int(.6 * data.num_nodes)
    test_size = int(.2 * data.num_nodes)

    # Select subset of nodes for training mask
    permutation = torch.randperm(data.num_nodes)

    data.train_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    data.train_mask[permutation[:train_size]] = True

    data.test_mask = torch.zeros(data.num_nodes , dtype=torch.bool, device=device)
    data.test_mask[permutation[train_size:train_size + test_size]] = True

    return data, dataset


def train_model(data, dataset, device, end=200):
    ''' Train GCN model '''
    model = GCN(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for i in range(end):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # if i % 20 == 0:
        #     print(f'Epoch {i}, Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, dataset = get_dataset(nodes=150, motifs = 7, device=device)

    model = train_model(data, dataset, device, end=1000)
    model.eval()

    output = model(data.x, data.edge_index)

    predictions = output.argmax(dim=1)
    train_accuracy = (predictions[data.train_mask] == data.y[data.train_mask]).float().mean()
    print(f"Training accuracy: {train_accuracy:.4f}")

    # print(data.edge_index)


    explainer = Explainer(
        model=model,
        algorithm=CFExplainer(epochs=200, lr=0.001),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    test_cf_examples = []
    print(torch.where(data.test_mask))
    for i in torch.where(data.test_mask)[0]:
        print(f"generating CF for {i} with {predictions[i], data.y[i]}")

        # sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i),
        #                                                              data.edge_index,
        #                                                              2, # Magic number!
        #                                                              data.x,
        #                                                              data.y)
        # new_idx = node_dict[int(i)]

        # explainer = CFExplainerOriginal(model=model,
        #                                 sub_adj=sub_adj,
        #                                 sub_feat=sub_feat,
        #                                 n_hid=20,
        #                                 dropout=0.5,
        #                                 sub_labels=sub_labels,
        #                                 y_pred_orig= predictions[i],
        #                                 num_classes = 2,
        #                                 beta=.5,
        #                                 device=device)

        # cf_example = explainer.explain(node_idx=i, cf_optimizer='SGD', new_idx=new_idx, lr=.001,
        #                             n_momentum=0.0, num_epochs=200)

        cf_explanation = explainer(data.x, data.edge_index, index=int(i))

        test_cf_examples.append(cf_explanation)


    print(test_cf_examples)


if __name__ == '__main__':
    main()
