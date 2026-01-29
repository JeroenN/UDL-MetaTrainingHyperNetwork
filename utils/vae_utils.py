from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    reduction: str = "mean",
):
    """
    Compute Structural Similarity Index (SSIM) loss between two images.

    :param img1: First image tensor.
    :param img2: Second image tensor.
    :param window_size: Size of the sliding window.
    :param reduction: Reduction method ('mean' or 'none').
    :return: SSIM loss value.
    """
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=0)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=0)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=0) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return 1 - ssim_map.mean() if reduction == "mean" else 1 - ssim_map


def loss_function(
    recon_x, x, mu, logvar, epoch, total_epochs, beta_start=0.0, beta_end=1.0
):
    """Compute VAE loss which includes MSE, SSIM, and KL divergence with beta annealing."""

    MSE = F.mse_loss(recon_x, x, reduction="sum")

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ssim = ssim_loss(recon_x, x)

    # beta annealing (linear)
    t = epoch / total_epochs
    beta = beta_start + (beta_end - beta_start) * t

    return MSE + ssim + beta * KLD


def loss_mse(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def loss_mse_only(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")
    return MSE


def evaluate_vae(
    vae,
    test_loader,
    epoch,
    total_epochs,
    beta_start,
    beta_end,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Evaluate VAE and optionally save visualizations.

    :param vae: VAE model to evaluate
    :param test_loader: DataLoader for the test dataset
    :param epoch: Current epoch number
    :param total_epochs: Total number of epochs
    :param beta_start: Starting beta value for KL annealing
    :param beta_end: Ending beta value for KL annealing
    :param output_dir: Directory to save visualizations during evaluation.
    """
    vae.eval()

    running_loss = 0.0
    list_mu = []
    list_logvar = []
    max_visualize = 5
    device = next(vae.parameters()).device

    if output_dir is not None:
        output_dir = Path(output_dir)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device, dtype=torch.float32)
            reconstruct, mu, logvar = vae(x)
            list_mu.append(mu.cpu())
            list_logvar.append(logvar.cpu())
            loss = loss_function(
                reconstruct,
                x,
                mu,
                logvar,
                epoch=epoch,
                total_epochs=1,
                beta_start=0.0,
                beta_end=1.0,
            )

            # Normalize loss per pixel
            batch_size = x.size(0)
            pixels = x[0].numel()
            running_loss += loss.item() / (batch_size * pixels)

            if output_dir is not None and batch_idx < max_visualize:
                img, ax = plt.subplots(2, 2)
                ax[0, 0].imshow(x[0].squeeze().to("cpu"))
                ax[0, 1].imshow(reconstruct[0].squeeze().detach().cpu())
                file_path = (
                    vis_dir / f"evaluation_batch_idx_{batch_idx}_epoch_{epoch}.png"
                )
                img.savefig(file_path)
                plt.close(img)

    print(f"VAE TEST LOSS PER PIXEL: {running_loss / (len(test_loader))}")
    return list_mu, list_logvar


def get_gaussian_from_vae(
    vae, x, idx, visualize: bool = False, output_dir: Optional[Union[str, Path]] = None
):
    """
    Get Gaussian parameters from VAE encoding.

    :param vae: VAE model
    :param x: Input tensor
    :param idx: Index for visualization file naming
    :param visualize: Whether to save visualization of reconstructions
    :param output_dir: Directory to save visualizations
    :return: mu and logvar tensors
    """
    vae.eval()
    device = next(vae.parameters()).device
    x = x.to(device, dtype=torch.float32)

    with torch.no_grad():
        reconstruct, mu, logvar = vae(x)

    if visualize and output_dir is not None:
        output_dir = Path(output_dir)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        img, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(x[0].squeeze().to("cpu"))
        ax[0, 1].imshow(reconstruct[0].squeeze().detach().cpu())
        file_path = vis_dir / f"results_{idx}.png"
        img.savefig(file_path)
        plt.close(img)

    return mu, logvar


def train_vae(
    vae,
    train_loader,
    test_loader,
    name,
    lr,
    epochs,
    log_interval,
    vae_description,
    models_folder,
    beta_start=0.0,
    beta_end=1.0,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Train VAE model, evaluate it, save the model, and also return KL divergence history.

    :param vae: VAE model to train
    :param train_loader: DataLoader for training data
    :param test_loader: DataLoader for testing data
    :param name: Name prefix for saving the model
    :param lr: Learning rate
    :param epochs: Number of training epochs
    :param log_interval: Interval for logging and evaluation
    :param vae_description: Description suffix for saving the model
    :param models_folder: Folder to save trained models
    :param beta_start: Starting beta value for KL annealing
    :param beta_end: Ending beta value for KL annealing
    :param output_dir: Directory to save visualizations during evaluation.
    :return: Trained VAE model and KL divergence history.
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    mu = []
    logvar = []
    kl_history = []

    for epoch in range(1, epochs + 1):
        vae.train()
        running_loss = 0.0
        device = next(vae.parameters()).device

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            reconstruct, mu, logvar = vae(x)
            loss = loss_function(
                reconstruct,
                x,
                mu,
                logvar,
                epoch=epoch,
                total_epochs=epochs,
                beta_start=beta_start,
                beta_end=beta_end,
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kl_term = (
                    0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
                )
                kl_history.append(kl_term.item())

            # Normalize loss per pixel
            batch_size = x.size(0)
            pixels = x[0].numel()
            running_loss += loss.item() / (batch_size * pixels)

        print(f"VAE: Epoch {epoch} loss_per_pixel={running_loss/len(train_loader):.4f}")
        if epoch % log_interval == 0:
            evaluate_vae(
                vae,
                test_loader,
                epoch,
                total_epochs=epochs,
                beta_start=beta_start,
                beta_end=beta_end,
                output_dir=output_dir,
            )
            file_name = name + vae_description + ".pth"
            # save_dir = models_folder / f"{epochs}_epochs"
            # save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"hyper_state_dict": vae.state_dict()}, models_folder / file_name
            )
    return vae, kl_history
