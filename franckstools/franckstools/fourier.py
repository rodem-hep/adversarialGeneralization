import torch
import numpy as np
def getDFTMatrix(n):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n))
    omega = torch.exp(-2.0j * torch.pi * np.mod(i*j,n) / n)
    return omega / torch.sqrt(torch.tensor(n, dtype=torch.float32))

def getInvDFTMatrix(n):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n))
    omega = torch.exp(2.0j * torch.pi * np.mod(i*j,n) / n)
    return omega / torch.sqrt(torch.tensor(n, dtype=torch.float32))

import torch

def sort_rows_by_average(tensor):
    # Calculate the average of each row
    row_averages = tensor.mean(dim=1)

    # Get the indices that would sort the row averages
    sorted_indices = torch.argsort(row_averages)

    center_sorted_indices = sorted_indices.clone() 
    center_id = len(sorted_indices)//2
    for i in range(len(sorted_indices)):
        if i % 2 == 0:
            center_sorted_indices[center_id-int(i/2)] = sorted_indices[len(sorted_indices)-1-i]
        else:
            center_sorted_indices[center_id+int((i+1)/2)] = sorted_indices[len(sorted_indices)-1-i]

    # Use fancy indexing to rearrange the rows
    sorted_tensor = tensor[center_sorted_indices]

    return sorted_tensor

def replace_fraction_smallest_across_tensors(tensor_list, fraction, replacement_value = 0):
    # Flatten each tensor to a 1D array and concatenate them
    flat_tensors = torch.cat([tensor.flatten() for tensor in tensor_list])

    # Calculate the number of values to replace based on the specified percentage
    num_values_to_replace = int(fraction * flat_tensors.size(0))

    # Get the indices of the num_values_to_replace smallest values
    _, indices = torch.topk(torch.abs(flat_tensors), k=num_values_to_replace, largest=False)

    # Replace the smallest values with the specified replacement value
    flat_tensors[indices] = replacement_value

    # Split the modified flat tensor back into individual tensors
    result_tensors = []
    start_idx = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        result_tensors.append(flat_tensors[start_idx:start_idx+numel].reshape(tensor.shape))
        start_idx += numel

    return result_tensors

def replace_fraction_smallest(tensor, fraction, replacement_value = 0.0):
    # Flatten the tensor to a 1D array
    flat_tensor = tensor.flatten()

    # Calculate the number of values to replace based on the specified percentage
    num_values_to_replace = int(fraction * flat_tensor.size(0))

    # Get the indices of the num_values_to_replace smallest values
    _, indices = torch.topk(torch.abs(flat_tensor), k=num_values_to_replace, largest=False)

    # Replace the smallest values with the specified replacement value
    flat_tensor[indices] = replacement_value

    # Reshape the modified flat tensor back into its original shape
    result_tensor = flat_tensor.reshape(tensor.shape)

    return result_tensor

def remove_n_smallest(tensor, n):
    # Flatten the tensor to a 1D array
    flat_tensor = tensor.flatten()

    # Get the indices of the n smallest values
    _, indices = torch.topk(torch.abs(flat_tensor), k=n, largest=False)

    # Remove the smallest values by setting them to a large number (e.g., infinity)
    flat_tensor[indices] = 0.0

    # Reshape the tensor back to its original shape
    result_tensor = flat_tensor.reshape(tensor.shape)

    return result_tensor

def main():

    import pytorch_lightning as pl
    import torch
    import torch.fft as fft

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import matplotlib.pyplot as plt
    import numpy as np

    import copy
    
    torch.manual_seed(0)

    torch.manual_seed(0)
    np.random.seed(0)

    
    # Generate some random data
    x_train = torch.rand(1000, 1) * 10
    y_train = torch.sin(x_train)
    # y_train = x_train**2+x_train
    # y_train = torch.sin(x_train)**2 + x_train



    import torch.nn.functional as F

    def complexRelu(tensor):
        return F.relu(tensor.real) + 1.0j*F.relu(tensor.imag)
    
    class FourierRelu(nn.Module):
        def __init__(self, mode = 'classic'):
            super().__init__()
            self.mode = mode
            
        def forward_classic(self, x):
            if x.dtype == torch.float32:
                return F.relu(x)
            return F.relu(x.real) + 1.0j*F.relu(x.imag)
        
        def forward_fourier(self, x):
            n = x.shape[1]
            DFT = getDFTMatrix(n)
            # DFT = torch.fft.fftn(torch.eye(n))
            return complexRelu(x)@DFT

        def forward(self, x):
            if self.mode == 'classic':
                return self.forward_classic(x)
            return self.forward_fourier(x)
        
        def switch_mode(self, mode = 'fourier'):
            self.mode = mode

        
    # Define the model
    class SimpleNN(pl.LightningModule):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 500, bias=True),
                FourierRelu(),
                nn.Linear(500, 50, bias=True),
                FourierRelu(),
                nn.Linear(50,50),
                FourierRelu(),
                nn.Linear(50,50),
                FourierRelu(),
                nn.Linear(50,50),
                FourierRelu(),
                nn.Linear(50,50),
                FourierRelu(),
                nn.Linear(50, 1, bias=True)
            )

        def forward(self, x):
            # x = torch.concat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
            # for name, param in self.model.named_parameters():
            #     print(name, param.data)
            #     a = x.to(torch.complex64)@getDFTMatrix(x.shape[1])
            #     x@param.data.transpose(0,1)
            #     paramF = param.data.clone().detach().to(torch.complex64)@getInvDFTMatrix(param.data.shape[1])
            #     a@paramF.transpose(0,1)

            # self.model[0].forward(x)
            return self.model(x)
        
        def forward_fourier(self, x):
            # x = torch.concat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
            return self.model(x.to(torch.complex64)@getDFTMatrix(x.shape[1])).real
                
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
        
        def prune_model(self, fraction):
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    param.data = replace_fraction_smallest(param.data, fraction=fraction)


        def transform_to_fourier(self):
            # Apply IDFT to weights matrix and activate the FourierRelu fourier mode
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    param_shape = param.data.shape
                    param.data = param.data.clone().detach().to(torch.complex64)@getInvDFTMatrix(param.data.shape[1]) #Careful, pytorch does Y = XW.T() and not X = XW => For non symmetric matrices (DFT is symmetric) we need to transpose the transformation matrix 

            for module in self.model.modules():
                if isinstance(module, FourierRelu):
                    module.switch_mode(mode='fourier')

            self.model = self.model.to(torch.complex64)
            self.forward = self.forward_fourier            
            

    # Create a DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the Lightning model
    model = SimpleNN()
    # print(model)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, train_loader)

    # Generate some test data for plotting
    x_test = torch.linspace(1, 10, 100).unsqueeze(1)
    y_test = torch.sin(x_test)
    # y_pred = model(x_test)

    # with torch.no_grad():
    #     test_loss = nn.MSELoss()(y_pred, y_test)

    # print(test_loss)

    # # model.transform_to_fourier()

    # y_pred_fourier = model(x_test)
    # with torch.no_grad():
    #     test_loss_fourier = nn.MSELoss()(y_pred_fourier, y_test)

    # print(test_loss_fourier)

    # # Plot the results
    # plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='True')
    # plt.plot(x_test.numpy(), y_pred.detach().numpy(), label='Predicted')
    # plt.plot(x_test.numpy(), y_pred_fourier.detach().numpy(), label='Predicted Fourier')
    # # plt.plot(x_test.numpy(), y_pred_pruned.detach().numpy(), label='Predicted Pruned')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'Predicted vs True, Test Loss: {test_loss.item():.4f}')
    # # output_dir = root / "plots"
    # # plt.savefig(f'{output_dir}/test.png')
    # # plt.show()
    # plt.close()

    losses = []
    fractions = np.linspace(0,0.5,300)
    for fraction in fractions:
        pruned_model = copy.deepcopy(model)
        pruned_model.prune_model(fraction=fraction)  
        y_pred_pruned = pruned_model(x_test)

        with torch.no_grad():
            test_loss = nn.MSELoss()(y_pred_pruned, y_test)
        losses.append(test_loss)  

    losses_fourier = []
    model.transform_to_fourier()

    for fraction in fractions:
        pruned_model = copy.deepcopy(model)
        pruned_model.prune_model(fraction=fraction)  
        y_pred_pruned = pruned_model(x_test)

        with torch.no_grad():
            test_loss = nn.MSELoss()(y_pred_pruned, y_test)
        losses_fourier.append(test_loss)
    # Prune the model
    # model.prune_model(min_amplitude=min_amp)  # You can adjust the sparsity level
    # Evaluate sparsity after pruning
    # sparsity_after = evaluate_sparsity(model)

    # y_pred_pruned = model(x_test)

    # with torch.no_grad():
    #     test_loss = nn.MSELoss()(y_pred_pruned, y_test)

    plt.plot(fractions, losses, label='classic pruning')
    plt.plot(fractions, losses_fourier, label='Franck pruning')
    plt.legend()
    plt.xlabel('pruning fraction')
    plt.ylabel('loss')
    # output_dir = root / "plots"
    plt.show()
    # plt.savefig(f'{output_dir}/test.png')
    plt.close()
    return 0
    return 0

if __name__ == '__main__':
    main()
