'''
display.py

visualise results of cf-gnnexplainer for different datasets and different
versions. use command line --exp=syn[DATASET] to select which graph to use.

'''

import sys
import os

import torch
from torch_geometric.utils import k_hop_subgraph
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.test_functions import load_dataset, explain_new

from gcn import GCNSynthetic
from wrapper import WrappedOriginalGCN
from cf_explanation.cf_explainer import CFExplainerNew, GreedyCFExplainer, BFCFExplainer


DATASETS = {
    'syn1': {'file': 'syn1.pickle', 'model': 'gcn_3layer_syn1.pt'},
    'syn2': {'file': 'syn2.pickle', 'model': 'gcn_3layer_syn2.pt'},
    'syn4': {'file': 'syn4.pickle', 'model': 'gcn_3layer_syn4.pt'},
    'syn5': {'file': 'syn5.pickle', 'model': 'gcn_3layer_syn5.pt'}
}


EXPLAINERS = {
    'CFExplainer': CFExplainerNew,
    'BFCFExplainer': BFCFExplainer,
    'GreedyCFExplainer': GreedyCFExplainer
}

CLASS_COLORS = ['blue', 'green', 'red']


def draw_node(node, data, counterfactual_fn, model, ax, explainer_cls, num_hops):
    ax.clear()

    try:
        counterfactual = counterfactual_fn(model, data.x, data.edge_index, data.y, [node], explainer_cls)
        cf_edges = data.edge_index[:, ~counterfactual.loc[0, 'cf_mask']]
    except Exception as e:
        ax.set_title(f'Node {node} Error: {e}', fontsize=10)
        return

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(subset.size(0)))
    orig_to_sub = {orig.item(): sub_idx for sub_idx, orig in enumerate(subset)}
    G.add_edges_from(sub_edge_index.t().tolist())

    # CF edges set
    cf_edges_set = set()
    for i in range(cf_edges.size(1)):
        u, v = cf_edges[0, i].item(), cf_edges[1, i].item()
        if u in orig_to_sub and v in orig_to_sub:
            su, sv = orig_to_sub[u], orig_to_sub[v]
            cf_edges_set.add((min(su, sv), max(su, sv)))

    y = data.y.clone()
    y %= 4

    y[torch.where(y > 0)] = 1
    y[node] = 2
    labels = y[subset]

    node_colors = [CLASS_COLORS[int(lbl) % len(CLASS_COLORS)] for lbl in labels]

    # Edge colours
    edge_colors = []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        edge_colors.append('black' if key in cf_edges_set else 'lightgrey')
    edge_widths = [2.5 if c == 'black' else 1.0 for c in edge_colors]

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax)

    ax.set_title(
        f'Node {node}  |  {explainer_cls.__name__}  |  {num_hops}-hop',
        fontsize=11
    )
    ax.axis('off')


def parse_args():
    parser = argparse.ArgumentParser(description='CF-GNNExplainer interactive viewer')
    parser.add_argument(
        '--exp',
        choices=list(DATASETS.keys()),
        default='syn1',
        help='Dataset to load (default: syn1)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_cfg = DATASETS[args.exp]

    device = 'cpu'
    script_dir = Path(__file__).parent

    graph_data_path = (script_dir / f'../../data/gnn_explainer/{dataset_cfg["file"]}').resolve()
    data = load_dataset(graph_data_path, device)

    num_classes = int(data.num_classes)

    model_path = script_dir / f'../../models/{dataset_cfg["model"]}'
    gcn = GCNSynthetic(10, 20, 20, num_classes, 0)
    gcn.load_state_dict(torch.load(model_path))
    gcn.eval()
    wrapped = WrappedOriginalGCN(gcn)

    node_ids = list(range(data.num_nodes))
    default_node = 515 if 515 in node_ids else node_ids[0]
    state = {
        'idx': node_ids.index(default_node),
        'explainer': list(EXPLAINERS.values())[0],
        'num_hops': 2,
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f'Dataset: {args.exp}', fontsize=10, color='grey', y=0.99)
    plt.subplots_adjust(bottom=0.28)

    # Node slider
    ax_node_slider = plt.axes([0.15, 0.20, 0.55, 0.03])
    node_slider = widgets.Slider(
        ax_node_slider, 'Node',
        node_ids[0], node_ids[-1],
        valinit=default_node,
        valstep=1
    )

    # k-hop slider
    ax_hop_slider = plt.axes([0.15, 0.14, 0.55, 0.03])
    hop_slider = widgets.Slider(
        ax_hop_slider, 'k-hop',
        1, 5,
        valinit=state['num_hops'],
        valstep=1
    )

    # Explainer radio buttons
    explainer_names = list(EXPLAINERS.keys())
    ax_radio = plt.axes([0.15, 0.02, 0.45, 0.10])
    ax_radio.set_title('Explainer', fontsize=9, pad=2)
    radio = widgets.RadioButtons(ax_radio, explainer_names, active=0)

    # Prev / Next buttons
    ax_prev = plt.axes([0.63, 0.02, 0.09, 0.05])
    ax_next = plt.axes([0.74, 0.02, 0.09, 0.05])
    btn_prev = widgets.Button(ax_prev, '◀')
    btn_next = widgets.Button(ax_next, '▶')

    # Jump text box
    ax_text = plt.axes([0.85, 0.02, 0.10, 0.05])
    text_box = widgets.TextBox(ax_text, '', initial=str(default_node))

    ax_save = plt.axes([0.63, 0.09, 0.20, 0.05])
    btn_save = widgets.Button(ax_save, 'Save Image')

    def refresh():
        node = node_ids[state['idx']]
        draw_node(node, data, explain_new, wrapped, ax,
                  state['explainer'], state['num_hops'])
        fig.canvas.draw_idle()

    def on_node_slider_drag(val):
        pass

    def on_mouse_release(event):
        if event.inaxes == ax_node_slider:
            node = int(node_slider.val)
            state['idx'] = node_ids.index(node) if node in node_ids else 0
            refresh()

    def on_hop_slider(val):
        state['num_hops'] = int(hop_slider.val)
        refresh()

    def on_radio(label):
        state['explainer'] = EXPLAINERS[label]
        refresh()

    def on_prev(_):
        state['idx'] = max(0, state['idx'] - 1)
        node_slider.set_val(node_ids[state['idx']])

    def on_next(_):
        state['idx'] = min(len(node_ids) - 1, state['idx'] + 1)
        node_slider.set_val(node_ids[state['idx']])

    def on_save(_):
        node = node_ids[state['idx']]
        save_fig, save_ax = plt.subplots(figsize=(6, 6))
        draw_node(node, data, explain_new, wrapped, save_ax,
                state['explainer'], state['num_hops'])
        out_path = script_dir / f'node{node}_{state["explainer"].__name__}_{state["num_hops"]}hop.png'
        save_fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(save_fig)
        print(f'Saved to {out_path}')

    def on_submit(text):
        try:
            node = int(text)
            if node in node_ids:
                state['idx'] = node_ids.index(node)
                node_slider.set_val(node)
                refresh()
            else:
                ax.set_title(f'Node {node} not in range', fontsize=12)
                fig.canvas.draw_idle()
        except ValueError:
            pass

    node_slider.on_changed(on_node_slider_drag)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    hop_slider.on_changed(on_hop_slider)
    radio.on_clicked(on_radio)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    text_box.on_submit(on_submit)
    btn_save.on_clicked(on_save)

    refresh()
    plt.show()


if __name__ == '__main__':
    main()
