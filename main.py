import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, grad
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from utils import (
    VAE,
    Dataset,
    HyperNetwork,
    TargetNet,
    compute_geometry_consistency_loss,
    get_dataset,
    get_gaussian_from_vae,
    plot_kl,
    train_shared_vae,
    train_vae_for_dataset,
)

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

#TRAIN_DATASET_NAMES = ["kmnist", "hebrew_chars", "fashion_mnist", "math_shapes"]
TRAIN_DATASET_NAMES = ["kmnist", "fashion_mnist"]
TEST_DATASET_NAME = ["mnist"]

models_folder = Path(__file__).parent / "models"
visualization_folder = Path(__file__).parent / "visualization"
# Ensure directories exist
models_folder.mkdir(parents=True, exist_ok=True)
visualization_folder.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

num_workers = 2 if device.type == "cuda" else 0
pin_memory = device.type == "cuda"


def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


CFG = load_config(Path(__file__).parent / "config.yaml")

batch_size_outerloop = CFG["training"]["batch_size_outerloop"]
batch_size_innerloop = CFG["training"]["batch_size_innerloop"]
batch_size_vae = CFG["training"]["batch_size_vae"]
epochs_hyper = CFG["training"]["epochs_hyper"]
epochs_vae = CFG["training"]["epochs_vae"]
lr_outer = float(CFG["training"]["lr_outer"])
lr_vae = float(CFG["training"]["lr_vae"])
lr_inner = float(CFG["training"]["lr_inner"])
weight_inner_loss = CFG["training"]["weight_innerloss"]
log_interval = CFG["training"]["log_interval"]
save_path = CFG["training"]["save_path"]
steps_innerloop = CFG["meta"]["steps_innerloop"]
steps_outerloop = CFG["meta"]["steps_outerloop"]
image_width_height = CFG["data"]["image_width_height"]
num_classes = CFG["data"]["num_classes"]
vae_head_dim = CFG["vae"]["vae_head_dim"]
n_samples_conditioning = CFG["vae"]["n_samples_conditioning"]
vae_description = CFG["vae"]["vae_description"]
shared_vae = CFG["vae"]["shared_vae"]
cluster_using_guassians = CFG["model"]["cluster_using_guassians"]
use_contrastive_loss = CFG["model"]["use_contrastive_loss"]
contrastive_temp = float(CFG["model"]["contrastive_temp"])
head_hidden = CFG["hypernet"]["head_hidden"]
use_bias = CFG["hypernet"]["use_bias"]
hidden_layers = CFG["target_net"]["hidden_layers"]
output_head = CFG["target_net"]["output_head"]
beta_start = CFG["vae"]["beta_start"]
beta_end = CFG["vae"]["beta_end"]
print_grads = CFG["troubleshoot"]["print_grads"]

ablation_mode = CFG["ablation"]["mode"]
ablation_noise_std = float(CFG.get("ablation", {}).get("noise_std", 1.0))
use_combined_loader = CFG["data"].get("use_combined_loader", False)

condition_dim = vae_head_dim * 2
input_dim = vae_head_dim * 2 if cluster_using_guassians else image_width_height**2
target_layer_sizes = [input_dim, *hidden_layers, output_head]

print("Loaded config:", Path(__file__).parent / "config.yaml")
print("target_layer_sizes =", target_layer_sizes)
print("condition_dim =", condition_dim)
print("use_combined_loader =", use_combined_loader)
print("ablation_mode =", ablation_mode)

n_samples_conditioning = batch_size_innerloop


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
    def __init__(
        self,
        train_dataset_names,
        test_dataset_names,
        model_folder,
        share_vae: bool = False,
    ):
        self.loaders = {}
        self.vaes = {}
        self.iters = {}
        self.model_folder = model_folder
        self.vae_distributions = {}
        self.train_datasets = {}
        self.test_datasets = {}

        self.share_vae = share_vae
        self.shared_vae = None

        self.kl_history_shared = None
        self.kl_history_by_dataset = {}  # if you also train separate VAEs

        # Process training datasets
        for name in tqdm(train_dataset_names, desc="Loading train datasets"):
            self.process_dataset(name, is_train=True)

        # Process test datasets
        for name in tqdm(test_dataset_names, desc="Loading test datasets"):
            self.process_dataset(name, is_train=False)

        # VAE initialization
        if share_vae:
            self.setup_shared_vae()
        else:
            self.setup_seperate_vaes()

        # Set VAE distributions for all datasets
        for dataset_name in list(self.train_datasets.keys()) + list(
            self.test_datasets.keys()
        ):
            self.set_vae_data_distributions(dataset_name)

        # Setup combined loader if enabled
        if use_combined_loader:
            self._setup_combined_loader()

    def process_dataset(self, dataset_name, is_train=True):
        """Helper to process a dataset (training or test) and create loaders"""
        subsets = get_dataset(
            name=dataset_name,
            preprocess=True,
            to_tensor=True,
            flatten=False,
            class_limit=num_classes,
        )

        for i, subset in enumerate(subsets):
            if is_train:
                subset_name = f"{dataset_name}_{i+1}"
                self.train_datasets[subset_name] = subset
            else:
                subset_name = f"{dataset_name}_test_{i+1}"  # Adds "test" to avoid naming conflicts
                self.test_datasets[subset_name] = subset

            # Create loader with appropriate shuffle setting
            self._create_loader(subset_name, subset, shuffle=is_train)

    def _create_loader(self, dataset_name, dataset_subset, shuffle=True):
        """Helper to create dataloader for a dataset subset"""
        dataset = Dataset(dataset_subset["train"])
        loader = DataLoader(
            dataset,
            batch_size=batch_size_innerloop,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loaders[dataset_name] = loader
        self.iters[dataset_name] = iter(loader)

    def setup_shared_vae(self):
        """Setup shared VAE for all datasets - trained only on training subsets"""
        self.shared_vae = VAE(
            w=image_width_height, h=image_width_height, ls_dim=vae_head_dim
        )
        file_name = "shared_vae" + vae_description + ".pth"
        path = self.model_folder / file_name

        if path.exists():
            self.shared_vae.load_state_dict(
                torch.load(path, map_location=device)["hyper_state_dict"]
            )
            print(f"Loaded shared VAE from {path}")
            self.kl_history_shared = None
        else:
            print(f"Shared VAE not found at {path}. Starting training...")
            # Collect only training subsets for VAE training (not test!)
            all_train_subsets = []
            for subset_name, subset in self.train_datasets.items():
                all_train_subsets.append(subset)
            _, kl_history = train_shared_vae(
                vae=self.shared_vae,
                train_datasets=all_train_subsets,
                batch_size_vae=batch_size_vae,
                num_workers=num_workers,
                pin_memory=pin_memory,
                lr_vae=lr_vae,
                epochs_vae=epochs_vae,
                log_interval=log_interval,
                vae_description=vae_description,
                models_folder=models_folder,
                beta_start=beta_start,
                beta_end=beta_end,
            )

        self.shared_vae.to(device).eval()
        self.kl_history_shared = kl_history

        # Assign shared VAE to all datasets (both train and test)
        for subset_name in self.train_datasets:
            self.vaes[subset_name] = self.shared_vae
        for subset_name in self.test_datasets:
            self.vaes[subset_name] = (
                self.shared_vae
            )  # Test uses shared VAE for inference only

    def setup_seperate_vaes(self):
        """Setup separate VAE for each dataset subset"""
        # Train separate VAE for each training subset
        for subset_name in tqdm(
            self.train_datasets.keys(), desc="Training dataset VAEs"
        ):
            self._train_or_load_vae(subset_name, self.train_datasets[subset_name])

        # Train separate VAE for each test subset
        for test_name in self.test_datasets:
            self._train_or_load_vae(test_name, self.test_datasets[test_name])

    def _train_or_load_vae(self, dataset_name, dataset_subset):
        """Helper to train or load VAE for a specific dataset subset"""
        vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
        file_name = dataset_name + vae_description + ".pth"
        path = self.model_folder / file_name

        if path.exists():
            vae.load_state_dict(
                torch.load(path, map_location=device)["hyper_state_dict"]
            )
            self.kl_history_by_dataset[dataset_name] = None
        else:
            print(f"VAE for {dataset_name} not found. Starting training...")
            _, kl_history = train_vae_for_dataset(
                vae=vae,
                dataset_name=dataset_name,
                dataset=dataset_subset,
                batch_size_vae=batch_size_vae,
                num_workers=num_workers,
                pin_memory=pin_memory,
                lr_vae=lr_vae,
                epochs_vae=epochs_vae,
                log_interval=log_interval,
                vae_description=vae_description,
                models_folder=models_folder,
                beta_start=beta_start,
                beta_end=beta_end,
            )

        vae.to(device).eval()
        self.vaes[dataset_name] = vae
        self.kl_history_by_dataset[dataset_name] = kl_history

    def get_batch(self, dataset_name, idx=None):
        # For compatibility, idx is ignored for single loader per dataset
        try:
            X, y = next(self.iters[dataset_name])
        except StopIteration:
            self.iters[dataset_name] = iter(self.loaders[dataset_name])
            X, y = next(self.iters[dataset_name])

        X = X.to(device)
        y = y.to(device)

        unique_labels = torch.unique(y)
        label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
        y = torch.tensor([label_map[int(lab)] for lab in y], device=y.device)

        vae = self.get_vae(dataset_name)

        with torch.no_grad():
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)

        mu = mu.to(device)
        logvar = logvar.to(device)
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
                X = X.to(device)
                
                if ablation_mode == "pixel_noise":
                    X = torch.randn_like(X) * ablation_noise_std
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
                batch_size=batch_size_innerloop,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            self.combined_iter = iter(self.combined_loader)
            print(f"Created combined loader with {len(combined_dataset)} samples from {len(all_train_datasets)} datasets")
            
            # Create combined VAE distribution
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

        X = X.to(device)
        y = y.to(device)

        # Remap labels to 0..num_unique-1
        unique_labels = torch.unique(y)
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
        
        mu = torch.cat(mus, dim=0).to(device)
        logvar = torch.cat(logvars, dim=0).to(device)
        
        return X, y, mu, logvar

    def get_combined_vae_distribution(self):
        """Get the combined VAE distribution for conditioning."""
        return self.combined_vae_distribution


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


def plot_losses_and_accuracies(inner_losses_dict, outer_losses_dict, average_acc_diff):
    Path(Path(__file__).parent / "visualization" / "plots").mkdir(
        parents=True, exist_ok=True
    )
    plt.figure()
    for dataset_name in inner_losses_dict:
        plt.plot(inner_losses_dict[dataset_name], label=f"{dataset_name} - inner loss")
        plt.plot(
            outer_losses_dict[dataset_name],
            linestyle="--",
            label=f"{dataset_name} - outer loss",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Losses per Dataset")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(__file__).parent / "visualization" / "plots" / "loss.png")
    plt.close()

    if len(average_acc_diff) > 0:
        plt.figure()
        plt.plot(average_acc_diff, label="Accuracy difference train vs no_train")

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracies averaged")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(__file__).parent / "visualization" / "plots" / "accuracy.png")
        plt.close()


# Basically clustering algorithm, find the neurons that correspond to the classes
def get_optimal_neuron_permutation(logits, targets, num_classes):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    confusion = np.zeros((num_classes, num_classes))
    for p, t in zip(preds, targets_np):
        confusion[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion)

    class_to_neuron = {c: r for r, c in zip(row_ind, col_ind)}

    used_neurons = set(class_to_neuron.values())
    all_neurons = set(range(num_classes))
    remaining_neurons = list(all_neurons - used_neurons)

    perm_indices = []
    for c in range(num_classes):
        if c in class_to_neuron:
            perm_indices.append(class_to_neuron[c])
        else:
            perm_indices.append(remaining_neurons.pop())

    return torch.tensor(perm_indices, device=logits.device)


def evaluate_classification(
    hyper,
    target: TargetNet,
    dataset_name,
    resources: ResourceManager,
    distributions_targets,
    mu_max_dist,
):

    vae_distribution = resources.get_vae_data_distribution(dataset_name)
    dist_inner_pool = vae_distribution
    dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]

    params_base = hyper(dist_inner)

    correct_base = 0
    total_base = 0

    with torch.no_grad():
        for mu, logvar, y in distributions_targets:
            y = y.to(device)
            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_base)

            perm = get_optimal_neuron_permutation(logits, y, num_classes)
            aligned_logits = logits[:, perm]

            preds = torch.argmax(aligned_logits, dim=1)
            correct_base += (preds == y).sum().item()
            total_base += y.size(0)

    acc_no_training = correct_base / total_base if total_base > 0 else 0.0

    buffers_hyper = dict(hyper.named_buffers())

    correct_adapted = 0
    total_adapted = 0

    for mu, logvar, y in distributions_targets:

        current_params_hyper = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in hyper.named_parameters()
        }

        for _ in range(steps_innerloop):

            dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]

            params_target = functional_call(
                hyper, (current_params_hyper, buffers_hyper), (dist_inner,)
            )

            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_target)
            target_output = F.normalize(logits, p=2, dim=1)

            loss = compute_geometry_consistency_loss(mu, target_output)
            loss = loss * weight_inner_loss

            grads = torch.autograd.grad(
                loss, current_params_hyper.values(), create_graph=False
            )

            current_params_hyper = {
                name: (p - g * lr_inner).detach().requires_grad_(True)
                for (name, p), g in zip(current_params_hyper.items(), grads)
            }

        with torch.no_grad():
            y = y.to(device)
            dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]
            params_target_adapted = functional_call(
                hyper, (current_params_hyper, buffers_hyper), (dist_inner,)
            )

            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_target_adapted)

            perm = get_optimal_neuron_permutation(logits, y, num_classes)
            aligned_logits = logits[:, perm]

            preds = torch.argmax(aligned_logits, dim=1)
            correct_adapted += (preds == y).sum().item()
            total_adapted += y.size(0)

    acc_training = correct_adapted / total_adapted if total_adapted > 0 else 0.0

    return acc_no_training, acc_training


def meta_training(
    hyper: HyperNetwork,
    target: TargetNet,
    train_dataset_names,
    test_dataset_name,
    resources: ResourceManager,
    print_grads: bool = False,
):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_outer)

    distributions_targets = []
    loader = resources.get_loader(test_dataset_name)
    vae = resources.get_vae(test_dataset_name)
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
            distributions_targets.append((mu, logvar, y))

            if batch_idx == steps_innerloop:
                break

    inner_losses = defaultdict(list)
    outer_losses = defaultdict(list)
    acc_training_list = []
    acc_no_training_list = []
    average_acc_diff = []
    mu_max_dist = None

    for epoch in tqdm(range(epochs_hyper), desc="Epochs"):
        params_hyper = dict(hyper.named_parameters())
        buffers_hyper = dict(hyper.named_buffers())

        if use_combined_loader:
            vae_dist_inner = resources.get_combined_vae_distribution()
            dataset_name = "combined"
        else:
            # Select a random training subset
            train_subset_names = list(resources.train_datasets.keys())
            dataset_name = random.choice(train_subset_names)
            vae_dist_inner = resources.get_vae_data_distribution(dataset_name)

        adapted_params_hyper = {k: v.clone() for k, v in params_hyper.items()}
        # Innerloop
        for _ in range(steps_innerloop):
            if use_combined_loader:
                inner_X, _, inner_mu, inner_logvar = resources.get_combined_batch()
            else:
                inner_X, _, inner_mu, inner_logvar = resources.get_batch(dataset_name)
            dist_inner = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]
            inner_distribution_input = torch.concat((inner_mu, inner_logvar), dim=1)

            params_target_inner = functional_call(
                hyper, (adapted_params_hyper, buffers_hyper), (dist_inner,)
            )

            target_logits = target.forward(
                inner_distribution_input, params_target_inner
            )

            target_output = F.normalize(target_logits, p=2, dim=1)

            inner_loss = compute_geometry_consistency_loss(
                inner_mu, target_output, mu_max_dist
            )
            inner_loss = inner_loss * weight_inner_loss
            grads = torch.autograd.grad(
                inner_loss,
                adapted_params_hyper.values(),
                create_graph=True,
                allow_unused=False,
            )

            adapted_params_hyper = {
                name: p - g * lr_inner
                for (name, p), g in zip(adapted_params_hyper.items(), grads)
            }

        # Outerloop
        dist_outer = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]
        if use_combined_loader:
            X_outer, y_outer, mu_outer, logvar_outer = resources.get_combined_batch()
        else:
            X_outer, y_outer, mu_outer, logvar_outer = resources.get_batch(dataset_name)
        outer_distribution_input = torch.concat((mu_outer, logvar_outer), dim=1)

        params_target_outer = functional_call(
            hyper, (adapted_params_hyper, buffers_hyper), (dist_outer,)
        )

        logits_outer = target.forward(outer_distribution_input, params_target_outer)

        perm = get_optimal_neuron_permutation(logits_outer, y_outer, num_classes)

        aligned_logits_outer = logits_outer[:, perm]

        outer_loss = F.cross_entropy(aligned_logits_outer, y_outer)

        optimizer.zero_grad()
        outer_loss.backward()

        ## Print gradients
        if print_grads:
            for name, p in hyper.named_parameters():
                if p.grad is not None:
                    print(
                        f"{name}: grad mean={p.grad.mean():.4e}, max={p.grad.abs().max():.4e}"
                    )

        optimizer.step()

        acc_no_training, acc_training = evaluate_classification(
            hyper,
            target,
            test_dataset_name,
            resources,
            distributions_targets,
            mu_max_dist,
        )
        inner_losses[dataset_name].append(inner_loss.detach().item())
        outer_losses[dataset_name].append(outer_loss.detach().item())
        acc_no_training_list.append(acc_no_training)
        acc_training_list.append(acc_training)
        if epoch > 50:
            acc_diff = [
                a_train - a_no_train
                for a_train, a_no_train in zip(acc_training_list, acc_no_training_list)
            ]
            average_acc_diff.append(sum(acc_diff[-50:]) / 50)
        if epoch % log_interval == 0:
            print(f"accuracy no training {acc_no_training:.4f}")
            print(f"accuracy training {acc_training: .4f}")
            if epoch > 100:
                print(
                    f"average acc training last 100 {sum(acc_training_list[-100:])/100:.4f}"
                )
                print(
                    f"average acc no training last 100 {sum(acc_no_training_list[-100:])/100:.4f}"
                )
            print(f"outer_loss: {outer_loss: .4f}")
            print(f"inner_loss: {inner_loss: .4f}")
            plot_losses_and_accuracies(inner_losses, outer_losses, average_acc_diff)


def plot_kl_histories(histories: dict | list, out_dir: Union[str, Path]):
    """
    Function to plot the KL divergence histories.

    :param histories: Dictionary of dataset names to KL histories OR a single KL history list.
    :param out_dir: Directory to save the plots.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # In case of per-dataset VAEs
    if isinstance(histories, dict):
        plotted = False
        for name, hist in histories.items():
            if hist is None or len(hist) == 0:
                continue

            plot_kl(
                hist,
                save_path=out_dir / f"{name}_kl.png",
            )
            plotted = True

        if not plotted:
            print("[KL] No per-dataset KL histories to plot.")
        return

    # In case of single history - shared VAE
    if histories is None or len(histories) == 0:
        print("[KL] No shared KL history to plot.")
        return

    plot_kl(
        histories,
        save_path=out_dir / "shared_vae_kl.png",
    )


def main():

    print(f"Working on device {device}")

    resources = ResourceManager(
        train_dataset_names=TRAIN_DATASET_NAMES,
        test_dataset_names=TEST_DATASET_NAME,
        model_folder=models_folder,
        share_vae=shared_vae,
    )

    if shared_vae:
        plot_kl_histories(
            resources.kl_history_shared, visualization_folder / "kl_plots"
        )
    else:
        plot_kl_histories(
            resources.kl_history_by_dataset, visualization_folder / "kl_plots"
        )

    hyper = HyperNetwork(
        layer_sizes=target_layer_sizes,
        condition_dim=condition_dim,
        head_hidden=head_hidden,
        use_bias=use_bias,
    ).to(device)

    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)

    # Get list of test subset names for evaluation
    test_subset_names = list(resources.test_datasets.keys())
    # Use first test subset for evaluation (you might want to modify this)
    test_dataset_for_eval = (
        test_subset_names[0] if test_subset_names else TEST_DATASET_NAME
    )

    # Get list of training subset names for meta-training
    train_subset_names = list(resources.train_datasets.keys())

    meta_training(
        hyper, target, train_subset_names, test_dataset_for_eval, resources, print_grads
    )


if __name__ == "__main__":
    main()
