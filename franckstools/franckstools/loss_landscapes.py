# loss_landscapes.py - Franck Rothen

import torch
import matplotlib.pyplot as plt
import logging
import numpy as np

import torch.optim as optim

log = logging.getLogger(__name__)


def loss_vs_weight_variation(
    model, trainer, dataloader, weight_range, num_samples=250, num_distances=10
):
    original_params = [param.data.clone() for param in model.parameters()]
    distances = torch.linspace(0, weight_range, num_distances)
    loss_samples = []

    with torch.no_grad():
        for distance in distances:
            log.info(f"distance = {distance}")
            sample_losses = []
            for _ in range(num_samples):
                for param, original_param in zip(model.parameters(), original_params):
                    if param.requires_grad:
                        param.data = (
                            original_param
                            + torch.randn(original_param.size()) * distance
                        )

                loss = compute_loss(model, dataloader)

                sample_losses.append(loss.item())

            for param, original_param in zip(model.parameters(), original_params):
                if param.requires_grad:
                    param.data = original_param

            loss_samples.append(sample_losses)

    losses_tensor = torch.tensor(loss_samples)
    return losses_tensor, distances


def compute_loss(model, dataloader, doMean=True, returnStdDev=False):
    losses = []

    for batch in dataloader:
        batch = [b.to(batch[0].device) for b in batch]
        if hasattr(model, "parent"):
            result = model.parent._shared_step(batch, 0)[0]
        else:
            result = model._shared_step(batch, 0)[0]

        losses.append(result)

    losses = torch.stack(losses)

    if doMean:
        loss = torch.mean(losses)
    else:
        loss = losses

    if returnStdDev:
        std_dev = torch.std(losses)
        return loss, std_dev
    else:
        return loss


def gradient_ascent_weight_tracing(model, dataloader, num_steps=50, step_size=0.01):
    loss_values = []
    std_values = []
    opt = optim.SGD(model.parameters(), lr=step_size)
    # Perform gradient ascent steps
    for step in range(num_steps + 1):
        opt.zero_grad()
        log.info(f"progress: {step/num_steps*100:.1f}%")
        # Compute gradients
        loss, std_dev = compute_loss(model, dataloader, returnStdDev=True)
        loss.backward()

        # Update model weights to maximize the loss
        for param in model.parameters():
            param.data += step_size * param.grad.data / param.grad.data.norm()

        # Record loss value
        loss_values.append(loss.item())
        std_values.append(std_dev.item())

    return loss_values, std_values


def gradient_ascent_input_tracing(model, dataloader, num_steps=50, step_size=0.001):
    # Load a batch from the dataloader

    # input_batch, target_batch = next(iter(dataloader))
    # input_batch = input_batch.to(input_batch.device)
    # target_batch = target_batch.to(target_batch.device)

    # input_batch.requires_grad_(True)

    loss_values = []

    for batch in dataloader:
        batch_loss_values = []

        batch = [b.to(batch[0].device) for b in batch]

        edges, nodes, high, adjmat, mask, label = batch

        for element in [nodes]:
            element.requires_grad = True

        for step in range(num_steps + 1):
            log.info(f"progress: {step/num_steps*100:.1f}%")

            # Compute the loss with the current input data
            loss = compute_loss(
                model,
                [(edges, nodes, high, adjmat, mask, label)],
                doMean=True,
                returnStdDev=False,
            )

            # Compute gradients of the loss with respect to the input data
            loss.backward()

            # Update the input data to maximize the loss
            with torch.no_grad():
                for element in [nodes]:
                    element += step_size * element.grad.data / element.grad.data.norm()
                    element.grad.zero_()

            # Record loss value
            batch_loss_values.append(loss.item())

        loss_values.append(batch_loss_values)

    std_values = [np.std(batch_loss_values) for batch_loss_values in zip(*loss_values)]
    loss_values = [
        np.mean(batch_loss_values) for batch_loss_values in zip(*loss_values)
    ]

    return torch.tensor(loss_values), torch.tensor(std_values)


def compute_gauss_newton_approximation(model, dataloader):
    # Initialize an empty Hessian matrix
    num_params = sum(p.numel() for p in model.parameters())
    hessian_approx = torch.zeros((num_params, num_params), device=model.device)

    for edges, nodes, high, adjmat, mask, label in dataloader:
        # Clear previous gradients
        model.zero_grad()

        # Move data to the same device as the model
        for element in [edges, nodes, high, adjmat, mask, label]:
            element = element.to(model.device)

        # Forward pass
        outputs = model(edges, nodes, high, adjmat, mask)
        loss = model.loss_fn(outputs, label)
        loss = torch.mean(loss)

        # Compute gradients
        loss.backward()

        # Flatten gradients into a single vector
        gradients = torch.cat([p.grad.flatten() for p in model.parameters()])

        # Update Hessian approximation
        hessian_approx += torch.outer(gradients, gradients)

    # Divide by the number of data points to get the average
    hessian_approx /= len(dataloader.dataset)

    return hessian_approx


# def visualize_hessian_eigenvalues(hessian, output_path):
#     eigenvalues, _ = torch.eig(hessian)
#     eigenvalues = eigenvalues[:, 0]
#     sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)

#     # Visualize the eigenvalues
#     import matplotlib.pyplot as plt
#     plt.plot(sorted_eigenvalues)
#     plt.xlabel("Eigenvalue Index")
#     plt.ylabel("Eigenvalue Value")
#     plt.title("Eigenvalues of Hessian Matrix")
#     plt.show()
