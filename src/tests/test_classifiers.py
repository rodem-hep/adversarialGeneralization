import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from functools import partial

import torch as T
import torch.nn as nn
import torchmetrics

from src.models.pc_classifiers import (
    DenseClassifier,
    GraphNetClassifier,
    TransformerClassifier,
)


def default_sizes() -> tuple:
    batch_size = 6
    n_nodes = 5
    edge_dim = 2
    node_dim = 3
    high_dim = 4
    n_classes = 2
    return (batch_size, n_nodes, edge_dim, node_dim, high_dim, n_classes)


def dummy_inputs() -> tuple:
    batch_size, n_nodes, edge_dim, node_dim, high_dim, n_classes = default_sizes()
    edges = T.randn(batch_size, n_nodes, n_nodes, edge_dim)
    nodes = T.randn(batch_size, n_nodes, node_dim)
    high = T.randn(batch_size, high_dim)
    mask = T.rand(batch_size, n_nodes).bernoulli_().bool()
    adjmat = mask.unsqueeze(-2) & mask.unsqueeze(-1)
    labels = T.randint(n_classes, (batch_size,))
    return edges, nodes, high, adjmat, mask, labels


def network_steps(net: nn.Module) -> None:
    # Create some dummy inputs
    inpts = dummy_inputs()

    # Test the three steps
    net.training_step(inpts, 0)
    net.validation_step(inpts, 0)
    net.predict_step(inpts, 0)


def test_dense() -> None:
    """Initialise a DenseClassifier and test it using an input of correct
    shape."""

    # Get the default sizes
    batch_size, n_nodes, edge_dim, node_dim, high_dim, n_classes = default_sizes()

    # Create the classifier
    net = DenseClassifier(
        inpt_dim=[edge_dim, node_dim, high_dim],
        n_nodes=n_nodes,
        n_classes=n_classes,
        loss_name="bcewithlogit",
        normaliser_config={},
        dense_config={
            "hddn_dim": [128, 64, 32],
            "nrm": "batch",
            "act_h": "silu",
        },
        optimizer=partial(T.optim.AdamW, lr=0.001),
        scheduler={
            "mattstools": {"name": "none"},
            "lightning": {},
        },
        accuracy=partial(torchmetrics.Accuracy, task="binary"),
    )

    # Run the step tests
    network_steps(net)


def test_transformer() -> None:
    """Initialise a TransformerClassifier and test it using an input of correct
    shape."""

    # Get the default sizes
    batch_size, n_nodes, edge_dim, node_dim, high_dim, n_classes = default_sizes()

    # Create the classifier
    net = TransformerClassifier(
        inpt_dim=[edge_dim, node_dim, high_dim],
        n_nodes=n_nodes,
        n_classes=n_classes,
        loss_name="bcewithlogit",
        normaliser_config={},
        ftve_config={
            "node_embd_config": {"hddn_dim": 64},
            "edge_embd_config": {"hddn_dim": 64},
            "tve_config": {
                "model_dim": 32,
                "num_sa_layers": 2,
                "num_ca_layers": 1,
                "mha_config": {
                    "num_heads": 2,
                },
                "dense_config": {"hddn_dim": 64},
            },
            "outp_embd_config": {"hddn_dim": 64},
        },
        optimizer=partial(T.optim.AdamW, lr=0.001),
        scheduler={
            "mattstools": {"name": "none"},
            "lightning": {},
        },
        accuracy=partial(torchmetrics.Accuracy, task="binary"),
    )

    # Run the step tests
    network_steps(net)


def test_graphnet() -> None:
    """Initialise a TransformerClassifier and test it using an input of correct
    shape."""

    # Get the default sizes
    batch_size, n_nodes, edge_dim, node_dim, high_dim, n_classes = default_sizes()

    # Create the classifier
    net = GraphNetClassifier(
        inpt_dim=[edge_dim, node_dim, high_dim],
        n_nodes=n_nodes,
        n_classes=n_classes,
        loss_name="bcewithlogit",
        normaliser_config={},
        fgve_config={
            "gnn_kwargs": {
                "num_blocks": 2,
                "gnb_kwargs": {},
            },
            "dns_kwargs": {"hddn_dim": 256},
        },
        optimizer=partial(T.optim.AdamW, lr=0.001),
        scheduler={
            "mattstools": {"name": "none"},
            "lightning": {},
        },
        accuracy=partial(torchmetrics.Accuracy, task="binary"),
    )

    # Run the step tests
    network_steps(net)
