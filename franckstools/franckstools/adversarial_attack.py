#adversial_attack.py - Franck Rothen
import torch
import numpy as np
from typing import Mapping

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import logging

from .UniversalInheritor import PlWrapper

log = logging.getLogger(__name__)

class AdversarialWrapper(PlWrapper):
    def __init__(self, 
                 parent, 
                 adversarial_scheduler_config: Mapping,
                 attack_type: str = 'FGSM', 
                 epsilon: float = 0.007, 
                 PGD_num_steps: int = 10, 
                 PGD_step_size: float = 0.002,
    ):
        super().__init__(parent=parent)

        self.adversarial_scheduler_config = adversarial_scheduler_config
        self.adversarial_fraction = adversarial_scheduler_config["min_frac"]
        self.epsilon = epsilon
        self.attack_type = attack_type

        if self.attack_type == 'FGSM':
            self._shared_step = self._shared_step_FGSM


        elif self.attack_type == 'PGD':
            self._shared_step = self._shared_step_PGD
            self.PGD_num_steps = PGD_num_steps
            self.PGD_step_size = PGD_step_size
        else:
            return ValueError(f'Invalid attack type: {self.attack_type}, please try FGSM or PGD')
    
    def _shared_step(self, sample: tuple, _batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        #Hacky solutions to a bug. The _shared_step has to be called once before _shared_step_adversariale (WIP)
        self.parent._current_fx_name = self._current_fx_name
        opt = self.optimizers()
        opt.zero_grad()
        loss, acc = self.parent._shared_step(batch, batch_idx)

        self.log("train/total_loss", loss)
        self.log("train/acc", acc) 
        self.log("adversarial_fraction", self.adversarial_fraction)
       
        self.training_step = self.training_step_adversarial
        return loss

    def training_step_adversarial(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        opt = self.optimizers()
        opt.zero_grad()
        loss, acc = self._shared_step(batch, batch_idx)

        self.log("train/total_loss", loss)
        self.log("train/acc", acc) 
        self.log("adversarial_fraction", self.adversarial_fraction)

        return loss
    
    def _shared_step_FGSM(self, sample: tuple, _batch_idx: int) -> torch.Tensor:
        edges, nodes, high, adjmat, mask, label = sample
        
        torch.set_grad_enabled(True)
        
        nodes.requires_grad = True
        
        outputs = self.parent.forward(edges, nodes, high, adjmat, mask)
        loss = self.parent.loss_fn(outputs, label).mean()
        self.zero_grad()
        #loss.backward()
        self.backward(loss)
        nodes_grad = nodes.grad.data 
        n_perturbed = int(self.adversarial_fraction*len(nodes))
        nodes.requires_grad = False
        nodes[:n_perturbed] = fgsm_attack(data_sample=nodes[:n_perturbed], epsilon=self.epsilon, data_grad=nodes_grad[:n_perturbed])

        self.optimizers().zero_grad()

        outputs = self.parent.forward(edges, nodes.detach(), high, adjmat, mask)
        loss = self.parent.loss_fn(outputs, label).mean()
        
        acc = self.parent.accuracy_method(outputs.squeeze(), label)

        # if self.hparams.grad_align_lambda != 0.0:
        #     loss += getGradAlignReg(self, edges, nodes, high, adjmat, mask, label, self.optimizers(), eps=self.epsilon, half_prec=False, grad_align_lambda=self.hparams.grad_align_lambda)

        return loss, acc
    
    def _shared_step_PGD(self, sample: tuple, _batch_idx: int) -> torch.Tensor:
        edges, nodes, high, adjmat, mask, label = sample
        
        torch.set_grad_enabled(True)
        
        nodes.requires_grad = True
        original_nodes = nodes.clone().detach()

        n_perturbed = int(self.adversarial_fraction * len(nodes))
        
        for _ in range(self.PGD_num_steps):

            nodes.requires_grad_()

            outputs = self.parent.forward(edges, nodes, high, adjmat, mask)
            loss = self.parent.loss_fn(outputs, label).mean()
            
            self.zero_grad()
            self.backward(loss)
            
            nodes_grad = nodes.grad.data
            
            # PGD step: perturb the nodes using gradient direction
            nodes.requires_grad = False
            nodes_perturbed = nodes
            nodes_perturbed[:n_perturbed] = nodes[:n_perturbed] + self.PGD_step_size * nodes_grad[:n_perturbed].sign()
            
            # Project nodes_perturbed back to the epsilon ball around the original nodes
            diff = nodes_perturbed - original_nodes
            diff = torch.clamp(diff, -self.epsilon, self.epsilon)
            nodes = original_nodes + diff.detach() 

            nodes.requires_grad = False

        nodes.requires_grad = False
        
        self.optimizers().zero_grad()
        
        outputs = self.forward(edges, nodes.detach(), high, adjmat, mask)
        loss = self.parent.loss_fn(outputs, label).mean()
        
        acc = self.parent.accuracy_method(outputs.squeeze(), label)

        return loss, acc
    
    def predict_step_adversarial(self, sample: tuple, _batch_idx: int) -> None:
        """Single step which produces the tagger outputs for a single test
        adversarial batch Must be as a dictionary to generalise to models with multiple
        tagging methods."""
        edges, nodes, high, adjmat, mask, label = sample

        torch.set_grad_enabled(True)
        
        nodes.requires_grad = True

        outputs = self.forward(edges, nodes, high, adjmat, mask)
        loss = self.loss_fn(outputs, label).mean()
        self.zero_grad()
        self.manual_backward(loss)
        nodes_grad = nodes.grad.data 
        # n_perturbed = int(self.hparams.adversarial_fraction*len(nodes))
        nodes.requires_grad = False
        # nodes[:n_perturbed] = fgsm_attack(data_sample=nodes[:n_perturbed], epsilon=0.007, data_grad=nodes_grad[:n_perturbed])
        nodes = fgsm_attack(data_sample=nodes, epsilon=self.epsilon, data_grad=nodes_grad)
        outputs = self.parent.forward(edges, nodes, high, adjmat, mask)
        return {"output": outputs}
    
    def on_train_epoch_end(self) -> None:
        self.adversarial_fraction = np.clip(self.adversarial_scheduler_config["min_frac"] + (self.adversarial_scheduler_config["max_frac"]-self.adversarial_scheduler_config["min_frac"]) * (self.current_epoch-self.adversarial_scheduler_config["min_steps"])/self.adversarial_scheduler_config["n_warmup_steps"], a_min=self.adversarial_scheduler_config["min_frac"], a_max=self.adversarial_scheduler_config["max_frac"])
        return super().on_train_epoch_end()
      

def fgsm_attack(data_sample, epsilon, data_grad, doClip=False):
    """
    Fast Gradient Sign Method (FGSM) attack to generate adversarial examples.

    Given a data sample, its corresponding gradient, and a small perturbation value epsilon,
    this function generates an adversarial example by slightly perturbing the data sample
    in the direction of the gradient. The perturbation is constrained by epsilon to ensure
    that the adversarial example is visually similar to the original sample, yet misclassifies
    the model's prediction.

    Parameters:
        data_sample (torch.Tensor): The original data sample, usually a tensor representing an image.
        epsilon (float): The perturbation limit, controlling the magnitude of the attack.
                         A higher epsilon results in a more noticeable perturbation.
        data_grad (torch.Tensor): The gradient of the loss function w.r.t. the data sample.
                                  This gradient is used to determine the direction of the perturbation.
        doClip (bool, optional): Whether to clip the perturbed data to stay within the valid data range.
                                 Defaults to False.

    Returns:
        torch.Tensor: The adversarial example generated using FGSM attack.
    """
    # Calculate the sign of the gradient (the direction to maximize the loss)
    sign_data_grad = data_grad.sign()

    # Add the perturbation to the data sample using the gradient direction
    perturbed_data_sample = data_sample + epsilon * sign_data_grad

    # Clip the perturbed data to ensure it stays within the valid data range
    # (e.g., pixel values between 0 and 1 for images)
    if doClip:
        perturbed_data_sample = torch.clamp(perturbed_data_sample, 0, 1)

    return perturbed_data_sample

####### Grad align start ########
# https://github.com/tml-epfl/understanding-fast-adv-training/blob/master/

def getGradAlignReg(model, edges, nodes, high, adjmat, mask, y, opt, eps=0.007, half_prec=False, grad_align_lambda = 0.2):
    grad1 = get_input_grad(model, edges, nodes, high, adjmat, mask, y, opt, eps, half_prec, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, edges, nodes, high, adjmat, mask, y, opt, eps, half_prec, delta_init='random_uniform', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
    reg = grad_align_lambda * (1.0 - cos.mean())

    return reg

def get_input_grad(model, edges, nodes, high, adjmat, mask, y, opt, eps, half_prec, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(nodes, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(nodes.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(nodes.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(edges, nodes + delta, high, adjmat, mask)
    # loss = F.cross_entropy(output, y)
    loss = torch.mean(model.loss_fn(output, y))
    if half_prec:
        log.warning("half prec is not implemented. Set half_prec to false to silence warning")
    # if half_prec:
    #     with amp.scale_loss(loss, opt) as scaled_loss:
    #         grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if backprop else False)[0]
    #         grad /= scaled_loss / loss
    #else:
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad

def get_uniform_delta(shape, eps, requires_grad=True):
    if torch.cuda.is_available():
        delta = torch.zeros(shape).cuda()
    else:
        delta = torch.zeros(shape)
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

####### Grad align end ########