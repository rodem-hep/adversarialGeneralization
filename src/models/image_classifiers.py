from functools import partial
from typing import Mapping

import torch as T
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

from mattstools.mattstools.cnns import DoublingConvNet
from mattstools.mattstools.torch_utils import get_loss_fn
from src.models.classifier_base import JetImageClassifier


class ConvNet(LightningModule):
    """Simple convolutional network to be used as classifier or as encoder for
    jet images."""

    def __init__(self, *, image_dim: int, n_outputs: int, convnet_config: Mapping):
        super().__init__()
        self.image_dim = image_dim
        self.n_outputs = n_outputs
        self.convnet_config = convnet_config

        cnn_layers = [
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.cnn_sequential = nn.Sequential(*cnn_layers)

        fc_layers = [
            nn.Linear(16 * 5 * 5, 120),
            nn.LeakyReLU(inplace=True),
            nn.Linear(120, self.n_outputs),
        ]
        self.fc_sequential = nn.Sequential(*fc_layers)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.cnn_sequential(x)
        x = T.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc_sequential(x)
        return x


class ImageConvClassifier(JetImageClassifier, LightningModule):
    """Simple conv classifier for jet images."""

    def __init__(
        self,
        *,
        image_dim: int,
        n_classes: int,
        inpt_dim: tuple,
        n_nodes: int,
        loss_name: str,
        convnet_config: Mapping,
        optimizer: partial,
        scheduler: Mapping
    ) -> None:
        super().__init__()
        self.net = ConvNet(
            image_dim=image_dim, n_outputs=n_classes, convnet_config=convnet_config
        )
        self.accuracy_method = torchmetrics.Accuracy(
            "multiclass", num_classes=n_classes
        )
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.loss_fn = get_loss_fn(loss_name)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class JIDoublingNetClassifier(JetImageClassifier, LightningModule):
    def __init__(
        self,
        *,
        n_classes: int,
        inpt_dim: tuple,
        n_nodes: int,
        loss_name: str,
        doublinconvnet_config: Mapping,
        optimizer: partial,
        scheduler: Mapping
    ) -> None:
        super().__init__()
        doublinconvnet_config["outp_dim"] = n_classes
        self.net = DoublingConvNet(**doublinconvnet_config)
        self.accuracy_method = torchmetrics.Accuracy(
            "multiclass", num_classes=n_classes
        )
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.loss_fn = get_loss_fn(loss_name)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)
