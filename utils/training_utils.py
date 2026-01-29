from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .dataset import Dataset

from .vae_utils import train_vae


def plot_kl(kl_history, save_path=None):
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


def flatten_params(params):
    flat = []
    for W, b in params:
        flat.append(W)
        if b is not None:
            flat.append(b)
    return flat


def unflatten_params(flat_params, template_params):
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
    vae,
    dataset_name,
    dataset,
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
):
    """Train VAE on a specific dataset subset"""
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
):
    print(f"Training a shared VAE on {len(train_datasets)} training subsets")

    # Combine ALL training data from ALL subsets
    all_train_data = {}
    current_idx = 0

    for dataset_subset in train_datasets:
        # Get the actual data from the subset
        train_split = dataset_subset["train"]

        # Convert to the format Dataset class expects
        for i in range(len(train_split)):
            ex = train_split[i]
            all_train_data[current_idx] = {"x": ex["x"], "y": ex["y"]}
            current_idx += 1

    # Create a combined dataset with correct format
    combined_dataset = Dataset(all_train_data)

    # Create DataLoader for the combined dataset
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size_vae,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Train VAE
    _, kl_history = train_vae(
        vae,
        combined_loader,
        combined_loader,  # Using same for validation for simplicity
        "shared_vae",
        lr_vae,
        epochs_vae,
        log_interval,
        vae_description,
        models_folder,
        beta_start,
        beta_end,
    )

    vae.eval()
    return vae, kl_history
