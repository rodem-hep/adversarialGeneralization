from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupToConstantSwag(_LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value. When reaching epoch=swa_start, decays the learning rate by gamma """

    def __init__(self, 
                 optimizer: Optimizer, 
                 num_steps: int = 100, 
                 swa_recording_lr_factor: float = 1.2, 
                 min_lr_ratio: float = 0.1, 
                 max_lr_ratio: float = 3.0):
        """
        args:
            optimizer (Optimizer): Wrapped optimizer.
            num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.swa_recording_lr_factor = swa_recording_lr_factor
        self.isSwaRecording = False
        self.cycle_fraction = -1
        self.min_lr_ratio = min_lr_ratio
        self.max_lr_ratio = max_lr_ratio
        
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.num_steps or self.isSwaRecording:
            lr = [base_lr for base_lr in self.base_lrs]
        else: 
            lr = [(base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs]
        
        if self.isSwaRecording:
            if self.cycle_fraction != -1:
                lr = [lr_val * (self.min_lr_ratio+(1-self.cycle_fraction)*(self.max_lr_ratio-self.min_lr_ratio)) for lr_val in lr]
            else:
                lr = [lr_val * self.swa_recording_lr_factor for lr_val in lr]

        return lr
