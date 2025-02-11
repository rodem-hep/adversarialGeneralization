# swag.py - Franck Rothen

import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from collections import OrderedDict
import logging

from typing import Mapping, Any


from .UniversalInheritor import PlWrapper

log = logging.getLogger(__name__)

class SWAGWrapper(PlWrapper):
    def __init__(self, parent, max_samples_to_record = 20, number_epoch_before_new_record = 2, scale = 0.5, isRecordCyclic=False, cycle_period = 10, cycle_min_lr_ratio = 0.1, cycle_max_lr_ratio = 3.0):
        super().__init__(parent=parent)
        self.flatten_skip_keys = ['w_avg', 'w2_avg', 'pre_D', 'is_w_avg_init', 'is_w2_avg_init', 'is_pre_D_init', 'isSwaRecording', 'n_models']
        self.register_buffer('w_avg', self.flatten())
        self.register_buffer('w2_avg', self.flatten())
        self.register_buffer('pre_D', self.flatten().clone()[:,None])
        self.register_buffer('is_w_avg_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('is_w2_avg_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('is_pre_D_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('isSwaRecording', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('n_models', torch.tensor(0, dtype=torch.int32))
        self.K = max_samples_to_record
        self.c = number_epoch_before_new_record
        self.scale = scale
        self.isRecordCyclic = isRecordCyclic
        self.cycle_period = cycle_period
        self.cycle_min_lr_ratio = cycle_min_lr_ratio
        self.cycle_max_lr_ratio = cycle_max_lr_ratio
        self.remaining_step_in_cycle = cycle_period
        if self.isRecordCyclic and self.c != 1:
            log.warning('"isRecordCylic" is active but "c" (number_epoch_before_new_record) is not equal to one. Reverting "c" to 1')
            self.c = 1

    def validation_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        loss, acc = self.parent._shared_step(sample, batch_idx)
        
        self.log("valid/total_loss", loss)
        self.log("valid/acc", acc)

        if not self.is_w_avg_init:
            swa_loss = loss
            swa_acc = acc
        else:
            tmp = self.flatten()
            self.load(self.w_avg)
            swa_loss, swa_acc = self.parent._shared_step(sample, batch_idx)
            self.load(tmp)
        
        self.log("valid/swa_total_loss", swa_loss)
        self.log("valid/swa_acc", swa_acc)
        
        return {'val_loss': loss, 'swa_loss': swa_loss}
    

    def forward_SWAG(self, edges, nodes, high, adjmat, mask):
        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=self.scale)
        outputs = self.parent.forward(edges, nodes, high, adjmat, mask)
        return outputs
    
    def predict_step_SWAG(self, sample: tuple, _batch_idx: int) -> None:
        edges, nodes, high, adjmat, mask, label = sample
        outputs = self.forward_SWAG(edges, nodes, high, adjmat, mask) # WARNING: Extremly slow since it samples for each step
        # outputs = self.parent.forward(edges, nodes, high, adjmat, mask)

        while torch.isnan(outputs).any():
            print("Outputs contain NaN values. Handling the case...") #TODO: Sometimes, the ouput contains nans.. why?
            outputs = self.forward_SWAG(edges, nodes, high, adjmat, mask)
        return {"output": outputs}
    
    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""
        self.parent.on_fit_start(*_args)
        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("valid/swa_total_loss", summary="min")
            wandb.define_metric("valid/swa_acc", summary="max")

    def init_SWA_export(self) -> None:
        self.load(self.w_avg)

    def init_SWAG_export(self) -> None:
        self.predict_step = self.predict_step_SWAG
        self.sample_weights(scale=self.scale)


    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        for key in self.flatten_skip_keys:
            # if hasattr(state_dict, key):
            if key in state_dict:
                continue
            state_dict[key] = self.__getattr__(key)

        if self.pre_D.shape != state_dict['pre_D'].shape:
            self.register_buffer('pre_D', torch.zeros(state_dict['pre_D'].shape))
        super().load_state_dict(state_dict)

    def transfer_checkpoint(self, ckpt_path: str):
        self.isSwaRecording = torch.tensor(True, dtype=torch.bool)
        checkpoint = torch.load(ckpt_path, map_location ='cpu')

        self.parent.load_state_dict(checkpoint['state_dict']) 

    def on_train_epoch_end(self) -> None:
        if self.isSwaRecording:
            if self.isRecordCyclic:
                self.trainer.lr_scheduler_configs[0].scheduler.min_lr_ratio = self.cycle_min_lr_ratio
                self.trainer.lr_scheduler_configs[0].scheduler.max_lr_ratio = self.cycle_max_lr_ratio
                self.trainer.lr_scheduler_configs[0].scheduler.cycle_fraction = (self.cycle_period-self.remaining_step_in_cycle)/self.cycle_period
                
                if self.remaining_step_in_cycle == 0:
                    self.aggregate_model()
                    self.remaining_step_in_cycle = self.cycle_period
                else:
                    self.remaining_step_in_cycle -= 1
            else:
                self.aggregate_model()
        return super().on_train_epoch_end()           
        

    # https://github.com/Lightning-AI/lightning/issues/1894
    # https://github.com/MilesCranmer/bnn_chaos_model/blob/master/spock_reg_model.py
    # Miles Cranmer

    def aggregate_model(self):
        """Aggregate parameters for SWA/SWAG"""

        cur_w = self.flatten()
        cur_w2 = cur_w ** 2
        with torch.no_grad():
            if not self.is_w_avg_init:
                self.is_w_avg_init = torch.tensor(True, dtype=torch.bool)
                self.is_w2_avg_init = torch.tensor(True, dtype=torch.bool)
                self.w_avg = cur_w
                self.w2_avg = cur_w2
            else:
                self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
                self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

            if not self.is_pre_D_init:
                self.is_pre_D_init = torch.tensor(True, dtype=torch.bool)
                self.pre_D = cur_w.clone()[:, None]
            elif self.current_epoch % self.c == 0:
                #Record weights, measure discrepancy with average later
                self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
                if self.pre_D.shape[1] > self.K:
                    self.pre_D = self.pre_D[:, 1:]
                    
        self.n_models += 1
    
    def sample_weights(self, scale=1):
        """Sample weights using SWAG:
        - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
            - This can be done with the following matrices:
                - z_1 ~ N(0, I_d); d the number of parameters
                - z_2 ~ N(0, I_K)
            - Then, compute:
            - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        """
        with torch.no_grad():
            avg_w = self.w_avg #[K]
            avg_w2 = self.w2_avg #[K]
            D = self.pre_D - avg_w[:, None]#[d, K]
            d = avg_w.shape[0]
            K = self.pre_D.shape[1]
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((K, 1), device=self.device)

            # sigma = torch.abs(torch.diag(avg_w2 - avg_w**2)) #STUPID
            # w = avg_w[None] + scale * (1.0/np.sqrt(2.0)) * z_1 @ sigma**0.5

            sigma = torch.abs(avg_w2 - avg_w**2)
            w = avg_w[None] + scale * (1.0/np.sqrt(2.0)) * z_1 * torch.sqrt(sigma)

            w += scale * (D @ z_2).T / np.sqrt(2*(K-1))
            w = w[0]

        self.load(w)
    
    def flatten(self):
        """Convert state dict into a vector, optionally skipping specified keys"""
        ps = self.state_dict()
        p_vec = None
        for key in ps.keys():
            if key in self.flatten_skip_keys:
                continue
            p = ps[key]
            if p_vec is None:
                p_vec = p.reshape(-1)
            else:
                p_vec = torch.cat((p_vec, p.reshape(-1)))
        return p_vec

    # The opposite of this is self.load(p_vec), defined as follows:
    def load(self, p_vec):
        """Load a vector into the state dict"""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key in cur_state_dict.keys():
            if key in self.flatten_skip_keys:
                continue
            old_p = cur_state_dict[key]
            size = old_p.numel()
            shape = old_p.shape

            # Check if the loaded tensor is scalar
            if size == 1:
                new_p = p_vec[i].view_as(old_p) # Just take the single value
            else:
                new_p = p_vec[i:i+size].reshape(*shape)

            new_state_dict[key] = new_p
            i += size
        
        # Add skipped key values back to the state dict
        for key in self.flatten_skip_keys:
            new_state_dict[key] = cur_state_dict[key]

        self.load_state_dict(new_state_dict)
        

class EarlyStoppingSWAG(EarlyStopping):
    def __init__(self, *args, n_extra_epochs=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_extra_epochs = n_extra_epochs
        self.trainer = None
        self.pl_module = None

        if n_extra_epochs==0:
            return ValueError('n_extra_epochs must have a value greater than 0')
    
    def on_train_start(self, trainer, pl_module):
        if self.trainer == None:
            self.trainer = trainer
            self.pl_module = pl_module
            self.trainer.lr_scheduler_configs[0].scheduler.isSwaRecording = self.pl_module.isSwaRecording.item() #TODO: needs to be wrapper permutative (self.pl_module.parent) if not top wrapper
            
        super().on_train_start(trainer, pl_module)
    
    def _evaluate_stopping_criteria(self, current):
        # Check if early stopping conditions are met
        
        if self.pl_module.isSwaRecording:
            self.n_extra_epochs -= 1
        else:
            should_stop_default, _ = super()._evaluate_stopping_criteria(current)
            if should_stop_default:
                self.trainer.lr_scheduler_configs[0].scheduler.isSwaRecording = True
                self.pl_module.isSwaRecording = torch.tensor(True, dtype=torch.bool)

        return (self.n_extra_epochs == 0), ('Reasons not implemented')
