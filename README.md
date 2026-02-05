# CF-GNNExplainer in PyTorch Geometric


This repository reimplements CF-GNNExplainer based on [AISTATS 2022 paper CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks](https://arxiv.org/abs/2102.03322).

CF-GNNExplainer was initially purely based on PyTorch, implementing GCN using
matrix multiplication and performing operations on the full adjacency matrix.
We reworked the method to use functionality from PyTorch Geometric, improving
the algorithmic complexity in the process.

Additionally we implement two new methods, greedy and brute-force, that
generate candidate counterfactuals differently for greater exploration and
performance.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt

```

## Recreation

The primary difference between versions is that PyG stores edge information
in COO-format, 2xE tensor of just the edges that are present whereas the
original implementation used full adjacency matrices. Original is known as
dense whereas our implementation is called sparse. In our experiments, we
want to explain the same classifiers as before despite this difference.
For this, we may use an interface to convert between edge formats.

## Training node classifiers

Pytorch Geometric's GCNConv uses a different format to store weights compared
to the original version. Because of this, there are two function with which to
train the classifiers.

Dense:
```train
python train.py --dataset=syn1
```

Sparse:
```gcn_sparce
python gcn_sparce --exp=syn1
```

> Datasets are "syn1" for BAShapes, "syn2" for BACommunity, "syn4" for Tree-Grid,
and "syn5" for Tree-Cycle ([source](https://github.com/RexYing/gnn-model-explainer)).
All hyperparameter settings are listed in the defaults.

## Training CF-GNNExplainer

To explain nodes for each dataset

```train
python main_explain.py --exp=syn1 --method=original --momentum=0.9 --epochs=100
python main_explain.py --exp=syn4 --sparse=True --model=sparse_gcn_3layer_syn4.pt
python main_explain.py --exp=syn5
```

> results are saved to results folder. use --sparse=True to select new framework.
other options are explain in main_explain.py


## Evaluation

To evaluate the CF examples, run the following command:

```eval
python evaluate.py --exp=syn1 --path=../results/<NAME OF RESULTS FILE>
```
> evaluate has an inbuilt check to verify that counterfactual examples lead
to the prediction in the dataframe. Use --sparse=True for classifiers trained
using gcn_sparse.py

## Pre-trained Models

The pretrained models are available in the models folder. Sparse models are
provided, but are unused in any thesis experiments.

## Experiments

In the thesis we perform experiments to show: (1) if we faithfully recreated
the original framework, (2) how our implementation affected the scalability, and
(3) how our newly proposed methods affect counterfactual quality and model performance.

Experiment 1 is found in scripts/eval_frameworks.sh for counterfactual evaluation, and src/experimetns/compare_all_formats.py for error measurements
Experiment 2 is found in src/experiments/measure_performance.py
Experiment 3 is found in scripts/eval_methods.sh for counterfactual evaluation, and src/experiments/measure_explain_time.py for performance

## Results

1: We recreate identical behaviour when using wrapper to interface. Using PyG's
gcn_norm leads to backwards graphs that produce different gradients that still
create counterfactuals of decent quality.
2: The new framework, when used on CPU, improves the algorithmic runtime complexity of cf-gnnexplainer. It is not faster for sparsely-connected networks from syn4 and syn5 datasets.
3: Greedy is very fast without compromising counterfactual quality. Bruteforce is
slightly faster and produces very minimal counterfactual examples.
