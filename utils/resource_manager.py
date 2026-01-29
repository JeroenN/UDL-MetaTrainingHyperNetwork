"""Resource manager for handling datasets, dataloaders, and VAEs."""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Dataset, get_dataset
from .vae_net import VAE
from .vae_utils import get_gaussian_from_vae
from .training_utils import train_shared_vae, train_vae_for_dataset


class CombinedDataset(torch.utils.data.Dataset):
    """Dataset that combines multiple datasets and tracks which dataset each sample came from."""
    
    def __init__(self, datasets_with_names: List[Tuple[torch.utils.data.Dataset, str]]):
        self.datasets = []
        self.dataset_names = []
        self.cumulative_sizes = [0]
        
        for dataset, name in datasets_with_names:
            self.datasets.append(dataset)
            self.dataset_names.append(name)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])):
            if start <= idx < end:
                local_idx = idx - start
                X, y = self.datasets[i][local_idx]
                return X, y, self.dataset_names[i]
        raise IndexError(f"Index {idx} out of range")


class ResourceManager:
    """Manages datasets, dataloaders, and VAE models for meta-learning experiments."""
    
    def __init__(
        self,
        train_dataset_names: List[str],
        test_dataset_names: List[str],
        model_folder: Union[str, Path],
        config: dict,
        device: torch.device,
        share_vae: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize ResourceManager.
        
        Args:
            train_dataset_names: List of training dataset names
            test_dataset_names: List of test dataset names
            model_folder: Path to save/load model checkpoints
            config: Configuration dictionary with all hyperparameters
            device: Torch device to use
            share_vae: Whether to use a shared VAE across datasets
            output_dir: Directory for saving visualizations
        """
        self.loaders = {}
        self.vaes = {}
        self.iters = {}
        self.model_folder = Path(model_folder)
        self.output_dir = Path(output_dir) if output_dir else None
        self.vae_distributions = {}
        self.train_datasets = {}
        self.test_datasets = {}
        self.config = config
        self.device = device

        self.share_vae = share_vae
        self.shared_vae = None

        self.kl_history_shared = None
        self.kl_history_by_dataset = {}

        # Extract config values
        self.num_classes = config["data"]["num_classes"]
        self.batch_size_innerloop = config["training"]["batch_size_innerloop"]
        self.num_workers = 2 if device.type == "cuda" else 0
        self.pin_memory = device.type == "cuda"
        self.image_width_height = config["data"]["image_width_height"]
        self.vae_head_dim = config["vae"]["vae_head_dim"]
        self.vae_description = config["vae"]["vae_description"]
        self.batch_size_vae = config["training"]["batch_size_vae"]
        self.lr_vae = float(config["training"]["lr_vae"])
        self.epochs_vae = config["training"]["epochs_vae"]
        self.log_interval = config["training"]["log_interval"]
        self.beta_start = config["vae"]["beta_start"]
        self.beta_end = config["vae"]["beta_end"]
        self.ablation_mode = config["ablation"]["mode"]
        self.ablation_noise_std = float(config.get("ablation", {}).get("noise_std", 1.0))
        self.use_combined_loader = config["data"].get("use_combined_loader", False)

        for name in tqdm(train_dataset_names, desc="Loading train datasets"):
            self.process_dataset(name, is_train=True)

        for name in tqdm(test_dataset_names, desc="Loading test datasets"):
            self.process_dataset(name, is_train=False)

        if share_vae:
            self.setup_shared_vae()
        else:
            self.setup_seperate_vaes()

        for dataset_name in list(self.train_datasets.keys()) + list(
            self.test_datasets.keys()
        ):
            self.set_vae_data_distributions(dataset_name)

        if self.use_combined_loader:
            self._setup_combined_loader()

    def process_dataset(self, dataset_name, is_train=True):
        """Helper to process a dataset (training or test) and create loaders"""
        subsets = get_dataset(
            name=dataset_name,
            preprocess=True,
            to_tensor=True,
            flatten=False,
            class_limit=self.num_classes,
        )

        for i, subset in enumerate(subsets):
            if is_train:
                subset_name = f"{dataset_name}_{i+1}"
                self.train_datasets[subset_name] = subset
            else:
                subset_name = f"{dataset_name}_test_{i+1}"  
                self.test_datasets[subset_name] = subset

            self._create_loader(subset_name, subset, shuffle=is_train)

    def _create_loader(self, dataset_name, dataset_subset, shuffle=True):
        """Helper to create dataloader for a dataset subset"""
        dataset = Dataset(dataset_subset["train"])
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_innerloop,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loaders[dataset_name] = loader
        self.iters[dataset_name] = iter(loader)

    def setup_shared_vae(self):
        self.shared_vae = VAE(
            w=self.image_width_height, h=self.image_width_height, ls_dim=self.vae_head_dim
        ).to(self.device)
        file_name = "shared_vae" + self.vae_description + ".pth"
        path = self.model_folder / file_name

        if path.exists():
            self.shared_vae.load_state_dict(
                torch.load(path, map_location=self.device)["hyper_state_dict"]
            )
            print(f"Loaded shared VAE from {path}")
            self.kl_history_shared = None
        else:
            print(f"Shared VAE not found at {path}. Starting training...")
            all_train_subsets = []
            for subset_name, subset in self.train_datasets.items():
                all_train_subsets.append(subset)
            _, kl_history = train_shared_vae(
                vae=self.shared_vae,
                train_datasets=all_train_subsets,
                batch_size_vae=self.batch_size_vae,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                lr_vae=self.lr_vae,
                epochs_vae=self.epochs_vae,
                log_interval=self.log_interval,
                vae_description=self.vae_description,
                models_folder=self.model_folder,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                output_dir=self.output_dir,
            )
            self.kl_history_shared = kl_history

        self.shared_vae.to(self.device).eval()

        for subset_name in self.train_datasets:
            self.vaes[subset_name] = self.shared_vae
        for subset_name in self.test_datasets:
            self.vaes[subset_name] = self.shared_vae

    def setup_seperate_vaes(self):
        """Setup separate VAE for each dataset subset"""
        for subset_name in tqdm(
            self.train_datasets.keys(), desc="Training dataset VAEs"
        ):
            self._train_or_load_vae(subset_name, self.train_datasets[subset_name])

        for test_name in self.test_datasets:
            self._train_or_load_vae(test_name, self.test_datasets[test_name])

    def _train_or_load_vae(self, dataset_name, dataset_subset):
        """Helper to train or load VAE for a specific dataset subset"""
        vae = VAE(w=self.image_width_height, h=self.image_width_height, ls_dim=self.vae_head_dim)
        file_name = dataset_name + self.vae_description + ".pth"
        path = self.model_folder / file_name
        kl_history = None

        if path.exists():
            vae.load_state_dict(
                torch.load(path, map_location=self.device)["hyper_state_dict"]
            )
            self.kl_history_by_dataset[dataset_name] = None
        else:
            print(f"VAE for {dataset_name} not found. Starting training...")
            _, kl_history = train_vae_for_dataset(
                vae=vae,
                dataset_name=dataset_name,
                dataset=dataset_subset,
                batch_size_vae=self.batch_size_vae,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                lr_vae=self.lr_vae,
                epochs_vae=self.epochs_vae,
                log_interval=self.log_interval,
                vae_description=self.vae_description,
                models_folder=self.model_folder,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                output_dir=self.output_dir,
            )

        vae.to(self.device).eval()
        self.vaes[dataset_name] = vae
        self.kl_history_by_dataset[dataset_name] = kl_history

    def get_batch(self, dataset_name, idx=None):
        """Get a batch from the specified dataset."""
        try:
            X, y = next(self.iters[dataset_name])
        except StopIteration:
            self.iters[dataset_name] = iter(self.loaders[dataset_name])
            X, y = next(self.iters[dataset_name])

        X = X.to(self.device)
        y = y.to(self.device)

        unique_labels = torch.unique(y)
        label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
        y = torch.tensor([label_map[int(lab)] for lab in y], device=y.device)

        vae = self.get_vae(dataset_name)

        with torch.no_grad():
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)

        mu = mu.to(self.device)
        logvar = logvar.to(self.device)
        return X, y, mu, logvar

    def get_loader(self, dataset_name):
        return self.loaders[dataset_name]

    def get_vae(self, dataset_name):
        return self.vaes[dataset_name]

    def get_vae_data_distribution(self, dataset_name):
        return self.vae_distributions[dataset_name]

    def set_vae_data_distributions(self, dataset_name):
        vae = self.get_vae(dataset_name)
        loader = self.get_loader(dataset_name)

        mus = []
        logvars = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                
                if self.ablation_mode == "pixel_noise":
                    X = torch.randn_like(X) * self.ablation_noise_std
                    X = torch.clamp(X, 0.0, 1.0)

                mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
                mus.append(mu)
                logvars.append(logvar)

        mu = torch.concat(mus, dim=0)
        logvar = torch.concat(logvars, dim=0)
        distribution = torch.concat((mu, logvar), dim=1)
        self.vae_distributions[dataset_name] = distribution

    def _setup_combined_loader(self):
        """Setup combined loader from all training datasets."""
        all_train_datasets = []
        for subset_name, subset in self.train_datasets.items():
            dataset = Dataset(subset["train"])
            all_train_datasets.append((dataset, subset_name))
        
        if all_train_datasets:
            combined_dataset = CombinedDataset(all_train_datasets)
            self.combined_loader = DataLoader(
                combined_dataset,
                batch_size=self.batch_size_innerloop,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            self.combined_iter = iter(self.combined_loader)
            print(f"Created combined loader with {len(combined_dataset)} samples from {len(all_train_datasets)} datasets")
            
            self._create_combined_vae_distribution()

    def _create_combined_vae_distribution(self):
        """Create a combined VAE distribution from all training datasets."""
        all_distributions = []
        for name in self.train_datasets.keys():
            if name in self.vae_distributions:
                all_distributions.append(self.vae_distributions[name])
        
        if all_distributions:
            self.combined_vae_distribution = torch.cat(all_distributions, dim=0)
            print(f"Created combined VAE distribution with {self.combined_vae_distribution.size(0)} samples")
        else:
            self.combined_vae_distribution = None

    def get_combined_batch(self):
        """Get a batch from the combined shuffled dataset."""
        try:
            X, y, dataset_names = next(self.combined_iter)
        except StopIteration:
            self.combined_iter = iter(self.combined_loader)
            X, y, dataset_names = next(self.combined_iter)

        X = X.to(self.device)
        y = y.to(self.device)

        unique_labels = torch.unique(y)
        
        # If more unique labels than num_classes, filter to keep only num_classes labels
        if len(unique_labels) > self.num_classes:
            valid_labels = unique_labels[:self.num_classes]
            mask = torch.isin(y, valid_labels)
            X = X[mask]
            y = y[mask]
            dataset_names = [n for n, m in zip(dataset_names, mask.cpu().tolist()) if m]
            unique_labels = valid_labels
        
        label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
        y = torch.tensor([label_map[int(lab)] for lab in y], device=y.device)

        # Encode each sample with its corresponding VAE
        mus = []
        logvars = []
        with torch.no_grad():
            for i, name in enumerate(dataset_names):
                vae = self.get_vae(name)
                sample = X[i:i+1]
                mu, logvar = get_gaussian_from_vae(vae, sample, 0, visualize=False)
                mus.append(mu)
                logvars.append(logvar)
        
        mu = torch.cat(mus, dim=0).to(self.device)
        logvar = torch.cat(logvars, dim=0).to(self.device)
        
        return X, y, mu, logvar

    def get_combined_vae_distribution(self):
        """Get the combined VAE distribution for conditioning."""
        return self.combined_vae_distribution
