# selective_weight_decay.py - Franck Rothen

from typing import Any, Callable, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from UniversalInheritor import PlWrapper
import numpy as np
import time


from torchsummary import summary



def main():
    torch.manual_seed(0)

    model = SimpleNN()
    model = SWDWrapper(model)
    model.target_rate = 1.0
    model.init_mask()
    model.remove_pruned_weights()
    # model.target_rate = 1.0
    model.init_mask()
    
    # Generate some random data
    x_train = torch.rand(20000, 1)* 6 * torch.pi
    y_train = torch.sin(x_train)

    # Create a DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, train_loader)

    # Generate some test data for plotting
    x_test = torch.linspace(0, 6*torch.pi, 500000).unsqueeze(1)
    y_test = torch.sin(x_test)

    # model.remove_pruned_weights()
    summary(model, input_size=(1,))

    # Record the starting time
    start_time = time.time()

    # Your line of code
    y_pruned_pred = model(x_test)

    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Time taken: {elapsed_time} seconds")
    with torch.no_grad():
        test_loss = nn.MSELoss()(y_pruned_pred, y_test)
    
    print(test_loss)
    

    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='True')
    plt.plot(x_test.numpy(), y_pruned_pred.detach().numpy(), label='Predicted')
    plt.show()
    plt.close()

# Define the model


class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers
        self.input_layer = nn.Linear(1, 20)

        self.hidden_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):

        return self.model(x.to(self.device))
           
    def training_step(self, batch, batch_idx):
        # Forward pass
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=0.001)
    
class SWDWrapper(PlWrapper):
    def __init__(self, parent, target_rate: float = 0.1, decay_factor_min: float = 0.1, decay_factor_max: float = 10E4, s_max: int = 8):
        super().__init__(parent=parent)
        self.target_rate = target_rate
        self.decay_factor = torch.tensor(decay_factor_min)
        self.decay_factor_min = torch.tensor(decay_factor_min)
        self.decay_factor_max = torch.tensor(decay_factor_max)
        self.s_max = s_max-1

    def training_step(self, sample: tuple, batch_idx: int):
        loss = super().training_step(sample, batch_idx)
        for layer_name, layer in self.parent.model.named_children():
            if isinstance(layer, nn.Linear):
                for name, param in layer.named_parameters():
                    loss +=  self.decay_factor * torch.sum(torch.pow(param.to(self.device)* self.decay_masks[f'{layer_name}.{name}'].to(self.device), 2))
                    break
            # for name, param in layer.named_parameters():
            #     if 'weight' in name:
            #         loss +=  self.decay_factor * torch.sum(torch.pow(param.to(self.device)* self.decay_masks[name[7:]].to(self.device), 2))
        # for name, param in self.named_parameters():
        #     if 'weight' or 'bias' in name:
        #         loss +=  self.decay_factor * torch.sum(torch.pow(param.to(self.device)* self.decay_masks[name[7:]].to(self.device), 2))
        return loss
    
    def get_weight_matrix_L2_norm(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(torch.pow(weight_matrix, 2)))

    def get_prubability_axis(self, layer):
        return torch.zeros(2)
    
    def _init_layer_mask(self, layer):
        return torch.zeros(2)
    
    def init_mask(self):
        self.decay_masks = {}
        # Loop over all nn.Linear layers
        for name, module in self.named_modules():  
            if isinstance(module, nn.Linear):
                print(f"Found Linear layer: {name}")
        # for layer_name, layer in self.parent.named_children():
        #     if isinstance(layer, nn.Linear):
                # continue
            # elif isinstance(layer, nn.Sequential):
                # for sublayer_name, sublayer in layer.named_children():
                    # if isinstance(sublayer, nn.Linear):
            #             self.decay_masks[f'{layer_name}.{sublayer_name}_prunability_id'] = torch.zeros(2)
            #             for name, param in sublayer.named_parameters():
            #                 self.decay_masks[f'{layer_name}.{sublayer_name}.{name}'] = torch.zeros(sublayer.weight.shape)
            #                 self.decay_masks[f'{layer_name}.{sublayer_name}.{name}'] = torch.zeros(sublayer.bias.shape)
            # self.decay_masks[f'{layer_name}_prunability_id'] = torch.zeros(2)
            # for name, param in layer.named_parameters():
            #     self.decay_masks[f'{layer_name}.{name}'] = torch.zeros(layer.weight.shape)
            #     self.decay_masks[f'{layer_name}.{name}'] = torch.zeros(layer.bias.shape)
            

        return

    # def init_mask(self):
    #     layer_counts = 0
    #     for name, param in self.parent.named_parameters():
    #         if 'weight' in name:
    #             layer_counts += 1

    #     self.decay_masks = {}
    #     weight_layer_id = 0
    #     for name, param in self.parent.named_parameters():
    #         if 'weight' in name:
    #             self.decay_masks[name] = torch.zeros(param.shape)
    #             self.decay_masks[name.replace('weight', 'bias')] = torch.zeros(param.shape[0])
    #             if weight_layer_id != 0: 
    #                 self.decay_masks[name][:,:int(1-np.sqrt(self.target_rate)*len(param[0,:]))] = 1
    #             if weight_layer_id != layer_counts-1:
    #                 self.decay_masks[name][:int(1-np.sqrt(self.target_rate)*len(param[:,0])),:] = 1
    #                 self.decay_masks[name.replace('weight', 'bias')][:int(1-np.sqrt(self.target_rate)*len(param[:,0]))] = 1



    #             weight_layer_id += 1
            
    #     return
    
    def remove_pruned_weights(self):
        for layer_name, layer in self.parent.model.named_children():
            setattr(self.parent.model, layer_name, self._prune_layer(layer))
    
    def _prune_layer(self, layer):
        if isinstance(layer, nn.Linear):
            for name, param in self.named_parameters():
                if param is layer.weight:
                    full_name = name[7:] #TODO: fix name[7:]
                    break
            
            pruned_layer = nn.Linear(torch.sum(self.decay_masks[full_name][-1,:]==0).item(), torch.sum(self.decay_masks[full_name][:,-1]==0).item())
            if layer.bias is not None:
                pruned_layer.bias.data = layer.bias.data[~self.decay_masks[full_name.replace('weight', 'bias')].bool()]
            pruned_layer.weight.data = layer.weight.data[~self.decay_masks[full_name].bool()].reshape(pruned_layer.weight.shape)
            return pruned_layer
        else: 
            return layer
    
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        # super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        # for group in optimizer.param_groups:
        #     for name, param in self.named_parameters():
        #         if 'weight' in name and param.grad is not None:
        #             # lr_tensor = torch.tensor(group['lr'], device=param.grad.device)
        #             decay_mask = self.decay_masks[name[7:]].to(param.grad.device) #TODO: fix name[7:]
        #             # param.grad.data.add_(decay_mask * self.decay_factor*(1/lr_tensor * param.data - param.grad.data)) 
        #             param.data.add_(-decay_mask * self.decay_factor * param.data)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def on_train_epoch_start(self) -> None:
        self.decay_factor = self.decay_factor_min*torch.pow(self.decay_factor_max/self.decay_factor_min,np.min([self.current_epoch/self.s_max, 1.0]))
        return super().on_train_epoch_start()
    
    def on_train_end(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                break
                # print(name)
                # print(param.data)
        return super().on_train_end()



if __name__ == "__main__":
    main()