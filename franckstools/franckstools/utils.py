# utils - Franck Rothen

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def AdMat2Tuple(batch_adjacency_tensor):
    # Convert the batch adjacency tensor to edge index
    edge_indices = []
    for batch_idx, adjacency_matrix in enumerate(batch_adjacency_tensor):
        # Find non-zero indices in the adjacency matrix
        non_zero_indices = torch.nonzero(adjacency_matrix, as_tuple=False)

        # Extract source and target node indices
        src_nodes = non_zero_indices[:, 0]
        tgt_nodes = non_zero_indices[:, 1]

        # Calculate the offset for source nodes due to batch index
        src_offset = batch_idx * adjacency_matrix.shape[0]

        # Adjust source node indices
        src_nodes_with_batch = src_nodes + src_offset

        # Create edge indices for the current batch
        batch_edge_indices = torch.stack((src_nodes_with_batch, tgt_nodes))

        # Append to the list of edge indices
        edge_indices.append(batch_edge_indices)

    # Concatenate the edge indices from each batch
    edge_indices = torch.cat(edge_indices, dim=1)

    return edge_indices


def power_iteration(A, num_iterations=1000):
    # largest_eigenvalue, eigenvector = power_iteration(hessian_matrix)
    device = A.device
    b = torch.randn(A.shape[0], 1, device=device)
    for _ in range(num_iterations):
        b = A @ b
        b_norm = torch.norm(b)
        b = b / b_norm
    eigenvalue = torch.squeeze(b.t() @ (A @ b))
    return eigenvalue.item(), b


# Function which returns a normalized discrete gaussian distribution
# with n points, such that the sum of the distribution is 1
def gaussian_dist(mass_bins, mu, sig):
    x = mass_bins
    y = (
        1
        / (np.sqrt(2 * np.pi * np.power(sig, 2.0)))
        * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
    )
    y = y / np.sum(y)
    return y


def uniform_proposal(x, delta=2.0):
    return np.random.uniform(x - delta, x + delta)


def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 1  # start somewhere

    for i in range(nsamples):
        trial = proposal(x)  # random neighbour from the proposal distribution
        acceptance = p(trial) / p(x)

        # accept the move conditionally
        if np.random.uniform() < acceptance:
            x = trial

        yield x


def optimal_transport_1D(source_samples, target_samples, x_bins):
    source = np.histogram(source_samples, bins=x_bins)[0]
    target = np.histogram(target_samples, bins=x_bins)[0]

    cdf_source = np.cumsum(source) / np.sum(source)
    cdf_target = np.cumsum(target) / np.sum(target)

    linear_interp_source = interp1d(
        x_bins[:-1] + np.diff(x_bins) / 2,
        cdf_source,
        fill_value=(0, 1),
        bounds_error=False,
    )
    inverse_linear_interp_target = interp1d(
        cdf_target,
        x_bins[:-1] + np.diff(x_bins) / 2,
        fill_value=(0, 1),
        bounds_error=False,
    )
    transformed_samples = inverse_linear_interp_target(
        linear_interp_source(source_samples)
    )
    return transformed_samples


def gaussian(x, mu, sig):
    return (
        1
        / (np.sqrt(2 * np.pi * np.power(sig, 2.0)))
        * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
    )


def pdf1(x):
    return (gaussian(x, 0, 1) + gaussian(x, 5, 1)) / 2


def pdf2(x):
    return gaussian(x, -4, 1)


def main():
    # # Generate a random double gaussian 1D dist

    # x = np.linspace(-10, 10, 100)
    # y1 = pdf1(x)
    # y2 = pdf2(x)

    # x_bins = np.linspace(-10, 10, 100)

    # # gaussian_dist1 = np.random.rand(0, 1, 10000)
    # # gaussian_dist2 = np.random.rand(5, 3, 10000)

    # dist1 = list(metropolis_sampler(pdf1, 10000))
    # dist2 = list(metropolis_sampler(pdf2, 10000))

    # # Transform dist1 to dist2
    # dist3 = optimal_transport_1D(dist2, dist1, x_bins)

    # #Get bin heights
    # hist1, edges1 = np.histogram(dist1, bins=x_bins)
    # hist2, edges2 = np.histogram(dist2, bins=x_bins)
    # hist3, edges3 = np.histogram(dist3, bins=x_bins)

    # # plt.plot(x, y1)
    # # plt.plot(x, y2)

    # plt.bar(edges1[:-1], hist1, width=np.diff(edges1), edgecolor="black", alpha=0.5, label="Original")
    # plt.bar(edges2[:-1], hist2, width=np.diff(edges2), edgecolor="black", alpha=0.5, label="Target")
    # plt.bar(edges3[:-1], hist3, width=np.diff(edges3), edgecolor="black", alpha=0.5, label="Transformed")
    # plt.legend()

    # plt.show()
    # plt.close()

    dist = gaussian_dist(10, 0, 10)
    print(dist)
    print(dist.sum())


if __name__ == "__main__":
    main()

