"""Main entry point for meta-learning hypernetwork experiments."""
import shutil
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import yaml

from utils import (
    HyperNetwork,
    TargetNet,
    ResourceManager,
    meta_training,
    plot_kl_histories,
)

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

# Dataset configuration
TRAIN_DATASET_NAMES = ["kmnist", "fashion_mnist"]
TEST_DATASET_NAME = ["mnist"]


def load_config(path: Union[str, Path]) -> dict:
    """Load YAML configuration file."""
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict, base_dir: Path) -> dict:
    """Setup experiment directory structure.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for the project
        
    Returns:
        Dictionary with paths to experiment directories
    """
    experiment_name = config.get("experiment", {}).get("name", "default")
    
    if experiment_name == "auto":
        from datetime import datetime
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiments_folder = base_dir / "experiments" / experiment_name
    paths = {
        "experiment": experiments_folder,
        "models": experiments_folder / "models",
        "plots": experiments_folder / "plots",
        "visualizations": experiments_folder / "visualizations",
        "embeddings": experiments_folder / "embeddings",
    }
    
    # Ensure directories exist
    for folder in paths.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config used for this experiment
    config_source = base_dir / "config.yaml"
    config_dest = experiments_folder / "config.yaml"
    shutil.copy2(config_source, config_dest)
    
    return paths


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def main():
    """Main training entry point."""
    base_dir = Path(__file__).parent
    config = load_config(base_dir / "config.yaml")
    device = get_device()
    
    # Setup experiment directories
    paths = setup_experiment(config, base_dir)
    
    # Extract commonly used config values for printing
    vae_head_dim = config["vae"]["vae_head_dim"]
    hidden_layers = config["target_net"]["hidden_layers"]
    output_head = config["target_net"]["output_head"]
    cluster_using_gaussians = config["model"]["cluster_using_guassians"]
    image_width_height = config["data"]["image_width_height"]
    use_combined_loader = config["data"].get("use_combined_loader", False)
    ablation_mode = config["ablation"]["mode"]
    save_embeddings_interval = config.get("experiment", {}).get("save_embeddings_interval", 10)
    shared_vae = config["vae"]["shared_vae"]
    print_grads = config["troubleshoot"]["print_grads"]
    
    condition_dim = vae_head_dim * 2
    input_dim = vae_head_dim * 2 if cluster_using_gaussians else image_width_height**2
    target_layer_sizes = [input_dim, *hidden_layers, output_head]
    
    print("Loaded config:", base_dir / "config.yaml")
    print("Experiment directory:", paths["experiment"])
    print("target_layer_sizes =", target_layer_sizes)
    print("condition_dim =", condition_dim)
    print("use_combined_loader =", use_combined_loader)
    print("ablation_mode =", ablation_mode)
    print("save_embeddings_interval =", save_embeddings_interval)
    print(f"Working on device {device}")

    # Initialize ResourceManager
    resources = ResourceManager(
        train_dataset_names=TRAIN_DATASET_NAMES,
        test_dataset_names=TEST_DATASET_NAME,
        model_folder=paths["models"],
        config=config,
        device=device,
        share_vae=shared_vae,
        output_dir=paths["experiment"],
    )

    # Plot KL histories from VAE training
    if shared_vae:
        plot_kl_histories(resources.kl_history_shared, paths["plots"])
    else:
        plot_kl_histories(resources.kl_history_by_dataset, paths["plots"])

    # Initialize networks
    hyper = HyperNetwork(
        layer_sizes=target_layer_sizes,
        condition_dim=condition_dim,
        head_hidden=config["hypernet"]["head_hidden"],
        use_bias=config["hypernet"]["use_bias"],
    ).to(device)

    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)

    # Get dataset names for training
    test_subset_names = list(resources.test_datasets.keys())
    test_dataset_for_eval = test_subset_names[0] if test_subset_names else TEST_DATASET_NAME
    train_subset_names = list(resources.train_datasets.keys())

    # Run meta-training
    meta_training(
        hyper=hyper,
        target=target,
        train_dataset_names=train_subset_names,
        test_dataset_name=test_dataset_for_eval,
        resources=resources,
        config=config,
        device=device,
        print_grads=print_grads,
        output_dir=paths["experiment"],
    )


if __name__ == "__main__":
    main()
