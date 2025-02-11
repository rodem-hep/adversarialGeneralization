import torch 
from torch.nn.utils import clip_grad_norm_

from .UniversalInheritor import PlWrapper
from mattstools.mattstools.torch_utils import get_sched
import logging
logger = logging.getLogger(__name__)


class SAMWrapper(PlWrapper):
    def __init__(self,
                 parent,
                 rho = 0.05,
                 method_type = "SAM",
                 sparsity = 0.5,
                 num_samples = 1,
                 update_freq = 1,
                 drop_rate = 0.5,
                 drop_strategy = "weight",
                 growth_strategy = "weight",
                 T_start = 0.0,
                 T_end = 1.0,
                 clip_gradient_value = 5.0):
        
        super().__init__(parent=parent)
        self.automatic_optimization = False
        self.parent.automatic_optimization = False
        self.method_type = method_type 
        self.rho = rho
        self.sparsity = sparsity
        self.num_samples = num_samples 
        self.update_freq = update_freq
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy 
        self.T_start = T_start
        self.T_end = T_end
        self.clip_gradient_value = clip_gradient_value



        # check if method_type is valid
        if method_type not in ["SAM", "SSAMF", "SSAMD"]:
            raise ValueError(f"method_type must be either SAM, SSAMF or SSAMD, not {method_type}")
        
        # set associated training step
        if method_type == "SAM":
            self.training_step = self.training_step_SAM
        elif method_type == "SSAMF":
            self.training_step = self.training_step_SSAMF
        elif method_type == "SSAMD":
            self.training_step = self.training_step_SSAMD
        
    def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(f"This method should have been overwritten by the {self.method_type} method.")
    
    def training_step_SAM(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        optimizer = self.optimizers()
        # first forward-backward pass
        def closureSAM():
            loss_1, acc = self.parent._shared_step(sample, batch_idx)
            self.manual_backward(loss_1)
            return loss_1, acc
        
        # second forward-backward pass
        loss_2, acc = self.parent._shared_step(sample, batch_idx)
        self.manual_backward(loss_2)
        # Clip gradients
        clip_grad_norm_(self.parameters(), max_norm=self.clip_gradient_value)

        optimizer.step(closure=closureSAM)
        optimizer.zero_grad()

        self.log("train/total_loss", loss_2) 
        self.log("train/acc", acc)

        self.lr_schedulers().step()

        return loss_2

    def training_step_SSAMF(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        optimizer = self.optimizers()
        # first forward-backward pass
        def closureSSAMF():
            loss_1, acc = self.parent._shared_step(sample, batch_idx)
            self.manual_backward(loss_1)
            return loss_1, acc
        
        # second forward-backward pass
        loss_2, acc = self.parent._shared_step(sample, batch_idx)
        self.manual_backward(loss_2)

        # Clip gradients
        clip_grad_norm_(self.parameters(), max_norm=self.clip_gradient_value)

        optimizer.step(closure=closureSSAMF, pl_model=self, epoch=self.trainer.current_epoch, batch_idx=batch_idx, train_data=self.trainer.datamodule.train_dataloader(), logger = logger)
        optimizer.zero_grad()

        self.log("train/total_loss", loss_2)
        self.log("train/acc", acc)

        self.lr_schedulers().step()

        return loss_2
    
    def training_step_SSAMD(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        
        optimizer = self.optimizers()
        # first forward-backward pass
        def closureSSAMD():
            loss_1, acc = self.parent._shared_step(sample, batch_idx)
            self.manual_backward(loss_1)
            return loss_1, acc
        
        # second forward-backward pass
        loss_2, acc = self.parent._shared_step(sample, batch_idx)
        self.manual_backward(loss_2)

        # Clip gradients
        clip_grad_norm_(self.parameters(), max_norm=self.clip_gradient_value)

        optimizer.step(closure=closureSSAMD, epoch=self.trainer.current_epoch, batch_idx=batch_idx, logger = logger)
        optimizer.zero_grad()

        self.log("train/total_loss", loss_2)
        self.log("train/acc", acc)

        self.lr_schedulers().step()

        return loss_2

    def configure_optimizers(self) -> dict:
        base_model = self
        for i in range(self.wrapperID):
            base_model = base_model.parent

        base_opt = (super().configure_optimizers())["optimizer"]
        
        logger.info(f"Using {self.method_type} method")
        if self.method_type == "SAM":
            from vendor.ssam import SAM
            opt = SAM(self.parameters(), #base_model.parameters() (?)
                      base_opt,
                      rho = self.rho,
                      **base_model.hparams.optimizer.keywords) #base_model.hparams.optimizer.func #TODO: Ask matthew if this is important

        elif self.method_type == "SSAMF":
            from vendor.ssam import SSAMF
            opt = SSAMF(self.parameters(),
                        base_opt,
                        rho = self.rho,
                        sparsity=self.sparsity,
                        num_samples=self.num_samples,
                        update_freq=self.update_freq, 
                        **base_model.hparams.optimizer.keywords)

        elif self.method_type == "SSAMD":
            from vendor.ssam import SSAMD
            opt = SSAMD(self.parameters(),
                        base_opt,
                        rho = self.rho,
                        sparsity=self.sparsity,
                        drop_rate=self.drop_rate,
                        drop_strategy=self.drop_strategy,
                        growth_strategy=self.growth_strategy,
                        update_freq=self.update_freq,
                        T_start=self.T_start,
                        T_end=self.T_end,
                        **base_model.hparams.optimizer.keywords)
        else:
            raise ValueError(f"method_type must be either SAM, SSAMF or SSAMD, not {self.method_type}")
        

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            base_model.hparams.scheduler.mattstools,
            opt,
            steps_per_epoch=len(base_model.trainer.datamodule.train_dataloader()),
            max_epochs=base_model.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **base_model.hparams.scheduler.lightning},
        }

