import torch
import torch.nn.functional as F


def pairwise_squared_distances(x, y):
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1).unsqueeze(0)
    return x_norm + y_norm - 2.0 * x @ y.T


def compute_geometry_consistency_loss(mu_batch, target_output, mu_max_dist=None):
    sim_target = torch.mm(target_output, target_output.T)
    dist_target = 1.0 - sim_target

    mu_norm = F.normalize(mu_batch, p=2, dim=1)
    sim_mu = torch.mm(mu_norm, mu_norm.T)
    dist_mu = 1.0 - sim_mu

    loss = F.mse_loss(dist_target, dist_mu)

    return loss
