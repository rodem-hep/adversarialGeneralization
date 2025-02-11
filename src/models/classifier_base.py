import torch as T
import wandb
from pytorch_lightning import LightningModule

from mattstools.mattstools.torch_utils import get_sched

class JetClassifier(LightningModule):
    """Base class containing common methods needed for all jet classifiers."""

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def _shared_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        raise NotImplementedError
    
    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss, acc = self._shared_step(sample, batch_idx)
        self.log("train/total_loss", loss)
        self.log("train/acc", acc)
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss, acc = self._shared_step(sample, batch_idx)
        
        self.log("valid/total_loss", loss)
        self.log("valid/acc", acc)

        return loss
    
    def test_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        edges, nodes, high, adjmat, mask, label = sample
        outputs = self.forward(edges, nodes, high, adjmat, mask)
        losses = self.loss_fn(outputs, label)
        return {'losses': losses}
    
    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            wandb.define_metric("train/acc", summary="max")
            wandb.define_metric("valid/acc", summary="max")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.scheduler.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
        }
    
class JetPCClassifier(JetClassifier):
    """Base class common methods needed for all jet point cloud classifiers."""

    def __init__(self) -> None:
        super().__init__()

        # All point cloud classifiers have the same accuracy tracker
        self.accuracy_method = self.hparams.accuracy(
            num_classes=self.hparams.n_classes,
            task="multiclass" if self.hparams.n_classes > 2 else "binary",
        )

    def _shared_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        edges, nodes, high, adjmat, mask, label = sample
        outputs = self.forward(edges, nodes, high, adjmat, mask)
        loss = self.loss_fn(outputs, label).mean()
        acc = self.accuracy_method(outputs.squeeze(), label)
        return loss, acc
    
    def predict_step(self, sample: tuple, _batch_idx: int) -> None:
        """Single step which produces the tagger outputs for a single test
        batch Must be as a dictionary to generalise to models with multiple
        tagging methods."""
        edges, nodes, high, adjmat, mask, label = sample
        outputs = self.forward(edges, nodes, high, adjmat, mask)
        return {"output": outputs}
    
class JetImageClassifier(JetClassifier):
    """Base class containing common methods needed for all jet image
    classifiers."""

    def _shared_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        image, label = sample
        outputs = self.forward(image)
        loss = self.loss_fn(outputs, label).mean()
        acc = self.accuracy_method(outputs, label)
        return loss, acc

    def predict_step(self, sample: tuple, _batch_idx: int) -> None:
        """Single step which produces the tagger outputs for a single test
        batch Must be as a dictionary to generalise to models with multiple
        tagging methods."""
        image, label = sample
        outputs = T.nn.functional.softmax(self.forward(image), dim=1)
        return {"1-0class_prob": 1 - outputs[:, :1]}
