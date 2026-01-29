from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dataset import Dataset
from .vae_utils import train_vae


def plot_kl(kl_history: list[float], save_path=None):
    """
    Plot KL divergence history during VAE training.

    :param kl_history: List of KL divergence values.
    :param save_path: Optional path to save the plot.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(kl_history, label="KL per batch (mean per sample)")
    plt.xlabel("Training step")
    plt.ylabel("KL")
    plt.title("VAE KL during training")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def flatten_params(
    params: list[tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> list[torch.Tensor]:
    flat = []
    for W, b in params:
        flat.append(W)
        if b is not None:
            flat.append(b)
    return flat


def unflatten_params(
    flat_params: list[torch.Tensor],
    template_params: list[tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> list[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    new_params = []
    i = 0
    for W, b in template_params:
        W_new = flat_params[i]
        i += 1
        if b is not None:
            b_new = flat_params[i]
            i += 1
        else:
            b_new = None
        new_params.append((W_new, b_new))
    return new_params


def train_vae_for_dataset(
    vae: torch.nn.Module,
    dataset_name: str,
    dataset: dict[str, Dataset],
    batch_size_vae: int,
    num_workers: int,
    pin_memory: bool,
    lr_vae: float,
    epochs_vae: int,
    log_interval: int,
    vae_description: str,
    models_folder: Union[str, Path],
    beta_start: float,
    beta_end: float,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Train VAE on a specific dataset subset

    :param vae: VAE model to train
    :param dataset_name: Name of the dataset
    :param dataset: Dataset dictionary with 'train' and 'test' splits
    :param batch_size_vae: Batch size for VAE training
    :param num_workers: Number of workers for data loading
    :param pin_memory: Whether to pin memory in DataLoader
    :param lr_vae: Learning rate for VAE optimizer
    :param epochs_vae: Number of epochs to train the VAE
    :param log_interval: Interval for logging training progress
    :param vae_description: Description for the VAE model
    :param models_folder: Folder to save trained models
    :param beta_start: Starting beta value for KL annealing
    :param beta_end: Ending beta value for KL annealing
    :param output_dir: Directory to save visualizations during evaluation.
    """
    print(f"Training VAE for {dataset_name}...")

    train_loader = DataLoader(
        Dataset(dataset["train"]),
        batch_size=batch_size_vae,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    _, kl_history = train_vae(
        vae,
        train_loader,
        train_loader,
        dataset_name,
        lr_vae,
        epochs_vae,
        log_interval,
        vae_description,
        models_folder,
        beta_start,
        beta_end,
        output_dir=output_dir,
    )
    return vae, kl_history


def train_shared_vae(
    vae,
    train_datasets,
    batch_size_vae,
    num_workers,
    pin_memory,
    lr_vae,
    epochs_vae,
    log_interval,
    vae_description,
    models_folder,
    beta_start,
    beta_end,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Train a shared VAE on multiple datasets.

    :param vae: VAE model to train
    :param train_datasets: List of dataset subsets for training
    (rest are same as the previous function)
    """
    print(f"Training a shared VAE on {len(train_datasets)} training subsets")

    all_train_data = {}
    current_idx = 0

    for dataset_subset in train_datasets:
        train_split = dataset_subset["train"]

        for i in range(len(train_split)):
            ex = train_split[i]
            all_train_data[current_idx] = {"x": ex["x"], "y": ex["y"]}
            current_idx += 1

    combined_dataset = Dataset(all_train_data)

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size_vae,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    _, kl_history = train_vae(
        vae,
        combined_loader,
        combined_loader,
        "shared_vae",
        lr_vae,
        epochs_vae,
        log_interval,
        vae_description,
        models_folder,
        beta_start,
        beta_end,
        output_dir=output_dir,
    )

    vae.eval()
    return vae, kl_history
