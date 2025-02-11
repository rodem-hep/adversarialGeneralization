from functools import partial
from typing import Mapping

import torch as T

from mattstools.mattstools.gnets.graphs import Graph
from mattstools.mattstools.gnets.modules import FullGraphVectorEncoder
from mattstools.mattstools.modules import DenseNetwork, IterativeNormLayer
from mattstools.mattstools.torch_utils import get_loss_fn
from mattstools.mattstools.transformers import FullTransformerVectorEncoder
from src.models.classifier_base import JetPCClassifier

from franckstools.franckstools.modules import LipschitzDenseNetwork
from franckstools.franckstools.vendor.LorentzNet.models import LorentzNet
from franckstools.franckstools.utils import AdMat2Tuple
from franckstools.franckstools.transformers import SimpleTransformer


class TransformerClassifier(JetPCClassifier):
    """Graph classifier using a transformer arcitecture."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        loss_name: str,
        normaliser_config: Mapping,
        ftve_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        accuracy: Mapping,
    ) -> None:
        """
        Args:
            inpt_dim: Number of edge, node and high level features
            n_nodes: Number of nodes used as inputs
            n_classes: Number of classes
            loss_name: Loss function to use in classification
            normaliser_config: Config for the IterativeNormLayer
            ftve_config: Config for the TransformerVectorEncoder
            optimizer: Partially initialised optimiser
            scheduler: How the scheduler should be used
            accuracy: Config for the accuracy torch metric
        """
        super().__init__()

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.loss_fn = get_loss_fn(loss_name)

        # The layers which normalise the input data
        self.node_norm = IterativeNormLayer(
            self.node_dim, **normaliser_config, track_grad_forward=True
        )  # track_grad_forward=True is required for adversarial training
        if self.edge_dim:
            self.edge_norm = IterativeNormLayer(
                self.edge_dim, **normaliser_config, track_grad_forward=True
            )
        if self.high_dim:
            self.high_norm = IterativeNormLayer(
                self.high_dim, **normaliser_config, track_grad_forward=True
            )

        # The transformer encoder
        self.ftve = FullTransformerVectorEncoder(
            inpt_dim=self.node_dim,
            outp_dim=n_classes if n_classes > 2 else 1,  # logistic regression
            edge_dim=self.edge_dim,
            ctxt_dim=self.high_dim,
            **ftve_config,
        )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> T.Tensor:
        # Pass through the normalisation layers
        nodes = self.node_norm(nodes, mask)
        if self.edge_dim:
            edges = self.edge_norm(edges, adjmat)
        if self.high_dim:
            high = self.high_norm(high)

        # Transformers create their attention matrices as: recv x send
        # This opposite to the GNNs adjmat which is: send x recv, rectify
        # Transformers also need the edges to be filled or None
        adjmat = adjmat.transpose(-1, -2)
        edges = edges.transpose(-2, -3) if edges.nelement() > 0 else None

        # Transformers also need the attention mask to not fully block all queries
        # Meaning that padded nodes need to be able to receive information
        # Even if the padded node will be ignored
        if adjmat.all() or not adjmat.any():
            adjmat = None
        else:
            adjmat = adjmat | ~mask.unsqueeze(-1)

        # Pass through the transformer and return
        return self.ftve(nodes, mask, ctxt=high, attn_mask=adjmat, attn_bias=edges)


class GraphNetClassifier(JetPCClassifier):
    """Graph classifier using a GraphNet arcitecture."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        loss_name: str,
        normaliser_config: Mapping,
        fgve_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        accuracy: Mapping,
    ) -> None:
        """
        Args:
            inpt_dim: Number of edge, node and high level features
            n_nodes: Number of nodes used as inputs
            n_classes: Number of classes
            loss_name: Loss function to use in classification
            normaliser_config: Config for the IterativeNormLayer
            fgve_config: Config for the FullGraphVectorEncoder
            optimizer: Partially initialised optimiser
            scheduler: How the scheduler should be used
            accuracy: Config for the accuracy torch metric
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.loss_fn = get_loss_fn(loss_name)

        # The layers which normalise the input data
        self.node_norm = IterativeNormLayer(
            self.node_dim, **normaliser_config, track_grad_forward=True
        )  # track_grad_forward=True is required for adversarial training
        if self.edge_dim:
            self.edge_norm = IterativeNormLayer(
                self.edge_dim, **normaliser_config, track_grad_forward=True
            )
        if self.high_dim:
            self.high_norm = IterativeNormLayer(
                self.high_dim, **normaliser_config, track_grad_forward=True
            )

        # The transformer encoder
        self.graphnet = FullGraphVectorEncoder(
            inpt_dim=[self.edge_dim, self.node_dim, 0],
            outp_dim=n_classes if n_classes > 2 else 1,  # logistic regression
            ctxt_dim=self.high_dim,
            **fgve_config,
        )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> T.Tensor:
        # Pass through the normalisation layers
        nodes = self.node_norm(nodes, mask)
        if self.edge_dim:
            edges = self.edge_norm(edges, adjmat)
        if self.high_dim:
            high = self.high_norm(high)

        # Create a graph object using your tensors and blank input globals
        globs = T.zeros((mask.shape[0], 0), dtype=nodes.dtype, device=nodes.device)
        graph = Graph(edges, nodes, globs, adjmat, mask)  # Edges are compressed here

        # Pass through the gnn and return
        return self.graphnet(graph, high)


class DenseClassifier(JetPCClassifier):
    """Dense network classifier acting on point cloud data for simplicity."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        loss_name: str,
        normaliser_config: Mapping,
        use_lip_dense: bool = False,
        lipschitz_const: float = 1.0,
        dense_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        accuracy: Mapping,
    ) -> None:
        """
        Args: same as above
        """
        super().__init__()

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.loss_fn = get_loss_fn(loss_name)
        self.use_lip_dense = use_lip_dense
        self.normaliser_config = normaliser_config

        # The dense encoder
        # if use_adversarial_samples:
        self.node_norm = IterativeNormLayer(
            self.node_dim * n_nodes,
            **normaliser_config,
            track_grad_forward=True,
        )
        # else:
        #     self.node_norm = IterativeNormLayer(
        #         self.node_dim * n_nodes, **normaliser_config, track_grad_forward=True,
        #     )

        if self.high_dim:
            self.high_norm = IterativeNormLayer(
                self.high_dim, **normaliser_config, track_grad_forward=True
            )

        if self.use_lip_dense:
            self.dense = LipschitzDenseNetwork(
                inpt_dim=self.node_dim * n_nodes,
                outp_dim=n_classes if n_classes > 2 else 1,  # logistic regression
                ctxt_dim=self.high_dim,
                lipschitz_const=lipschitz_const,
                **dense_config,
            )
        else:
            self.dense = DenseNetwork(
                inpt_dim=self.node_dim * n_nodes,
                outp_dim=n_classes if n_classes > 2 else 1,  # logistic regression
                ctxt_dim=self.high_dim,
                **dense_config,
            )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> T.Tensor:
        # Flatten nodes and pass
        nodes = nodes.view(nodes.shape[0], -1)
        nodes = self.node_norm(nodes)
        return self.dense(nodes, high)


class LorentzClassifier(JetPCClassifier):
    """Lorentz network classifier acting on point cloud data for simplicity."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        loss_name: str,
        normaliser_config: Mapping,
        use_adversarial_samples: bool = False,
        adversarial_fraction: float = 0,
        adversarial_epsilon: float = 0.007,
        grad_align_lambda: float = 0.0,
        use_lip_dense: bool = False,
        lipschitz_const: float = 1.0,
        lorentz_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        accuracy: Mapping,
    ) -> None:
        """
        Args: same as above
        """
        super().__init__()

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.loss_fn = get_loss_fn(loss_name)
        self.use_lip_dense = use_lip_dense
        self.normaliser_config = normaliser_config

        # The dense encoder
        if use_adversarial_samples:
            self.node_norm = IterativeNormLayer(
                self.node_dim * n_nodes,
                **normaliser_config,
                track_grad_forward=True,
            )
        else:
            self.node_norm = IterativeNormLayer(
                self.node_dim * n_nodes,
                **normaliser_config,
            )

        self.lorentz = LorentzNet(
            # n_scalar=self.node_dim * n_nodes,
            n_scalar=self.high_dim,
            # n_hidden=self.node_dim*n_nodes,
            n_class=n_classes if n_classes > 2 else 1,  # logistic regression
            **lorentz_config,
        )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> T.Tensor:
        # Flatten nodes and pass
        nodes = nodes.view(nodes.shape[0], -1)
        nodes = self.node_norm(nodes)

        edge_indices = AdMat2Tuple(batch_adjacency_tensor=adjmat)

        return self.lorentz(
            scalars=high,
            x=nodes,
            edges=edge_indices,
            node_mask=mask,
            edge_mask=None,
            n_nodes=self.n_nodes,
        )
        # return self.lorentz(nodes, high)


class SimpleTransformerClassifier(JetPCClassifier):
    """Simplified transformer classifier acting on point cloud data"""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        loss_name: str,
        normaliser_config: Mapping,
        transformer_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        accuracy: Mapping,
    ) -> None:
        """
        Args: same as above
        """
        super().__init__()

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.loss_fn = get_loss_fn(loss_name)
        self.normaliser_config = normaliser_config

        # The layers which normalise the input data
        self.node_norm = IterativeNormLayer(
            self.node_dim, **normaliser_config, track_grad_forward=True
        )

        if self.high_dim:
            self.high_norm = IterativeNormLayer(
                self.high_dim, **normaliser_config, track_grad_forward=True
            )

        # The transformer encoder
        self.transformer = SimpleTransformer(
            output_dim=n_classes if n_classes > 2 else 1,  # logistic regression
            input_dim=self.node_dim,
            dim=transformer_config["dim"],
            num_heads=transformer_config["num_heads"],
            num_layers=transformer_config["num_layers"],
            ff_mult=transformer_config["ff_mult"],
            dropout=transformer_config["dropout"],
            do_final_norm=transformer_config["do_final_norm"],
        )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> T.Tensor:
        # Pass through the normalisation layers
        nodes = self.node_norm(nodes, mask)
        if self.high_dim:
            high = self.high_norm(high)

        # Pass through the transformer and return
        return self.transformer(nodes, kv_mask=mask, ctxt=high)
