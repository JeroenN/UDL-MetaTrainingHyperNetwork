import torch
import torch.nn.functional as F


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise squared Euclidean distances between two sets of vectors.

    :param x: Tensor of shape (N, D)
    :param y: Tensor of shape (M, D)
    """
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1).unsqueeze(0)
    return x_norm + y_norm - 2.0 * x @ y.T


def compute_geometry_consistency_loss(
    mu_batch: torch.Tensor, target_output: torch.Tensor, mu_max_dist=None
) -> torch.Tensor:
    """
    Computes the geometry consistency loss between the latent representations (mu_batch) and the target outputs (target_output).

    :param mu_batch: Tensor representing the latent representations.
    :param target_output: Tensor representing the target outputs.
    :param mu_max_dist: Optional maximum distance for normalization.

    :return: Scalar tensor representing the geometry consistency loss.
    """
    sim_target = torch.mm(target_output, target_output.T)
    dist_target = 1.0 - sim_target

    mu_norm = F.normalize(mu_batch, p=2, dim=1)
    sim_mu = torch.mm(mu_norm, mu_norm.T)
    dist_mu = 1.0 - sim_mu

    loss = F.mse_loss(dist_target, dist_mu)

    return loss
