import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, grad
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset_loading import Dataset, get_dataset
from utils import VAE, HyperNetwork, TargetNet, get_gaussian_from_vae, train_vae

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

train_dataset_names = ["kmnist", "fashion_mnist"]  # , "math_shapes", "hebrew_chars"]
test_dataset_name = "mnist"
models_folder = Path(__file__).parent / "models" / "vaes"
timestamp = int(time.time())
experiment_folder = (
    Path(__file__).parent / "experiments" / f"meta_contrastive_experiment_{timestamp}"
)
experiment_folder.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

num_workers = 2 if device.type == "cuda" else 0
pin_memory = device.type == "cuda"
import torch

print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
print("device:", torch.cuda.get_device_name(0))
print("total mem (GiB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("free/total:", [x / 1024**3 for x in torch.cuda.mem_get_info()])


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
save_path = experiment_folder / "hyper_model.pth"
steps_innerloop = CFG["meta"]["steps_innerloop"]
steps_outerloop = CFG["meta"]["steps_outerloop"]
image_width_height = CFG["data"]["image_width_height"]
num_classes = CFG["data"]["num_classes"]
vae_head_dim = CFG["vae"]["vae_head_dim"]
n_samples_conditioning = CFG["vae"]["n_samples_conditioning"]
retrain_vae = CFG["vae"]["retrain_vae"]
vae_description = CFG["vae"]["vae_description"]
cluster_using_guassians = CFG["model"]["cluster_using_guassians"]
use_contrastive_loss = CFG["model"]["use_contrastive_loss"]
contrastive_temp = float(CFG["model"]["contrastive_temp"])
head_hidden = CFG["hypernet"]["head_hidden"]
use_bias = CFG["hypernet"]["use_bias"]
hidden_layers = CFG["target_net"]["hidden_layers"]
output_head = CFG["target_net"]["output_head"]

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
    def __init__(self, dataset_names, test_name, model_folder):
        self.loaders = {}

        self.vaes = {}
        self.iters = {}
        self.model_folder = model_folder

        self.vae_distributions = {}
        self.train_dataset_names = dataset_names

        all_names = list(set(dataset_names + [test_name]))

        # Collect all training datasets for combined loader
        all_train_datasets = []

        for name in tqdm(all_names, desc="Loading datasets"):
            vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
            file_name = name + vae_description + ".pth"
            path = self.model_folder / file_name
            if path.exists():
                vae.load_state_dict(
                    torch.load(path, map_location=device)["hyper_state_dict"]
                )
            else:
                print(f"vae not found at {path}. Starting VAE training")
                train_vaes(name)

            vae.to(device)
            vae.eval()
            self.vaes[name] = vae

            bs = batch_size_innerloop

            datasets = get_dataset(
                name=name,
                preprocess=True,
                to_tensor=True,
                flatten=False,
                class_limit=num_classes,
            )
            loaders_per_dataset = []
            iters_per_dataset = []
            for idx, ds in enumerate(datasets):
                split = "train"
                dataset = Dataset(ds[split])

                loader = DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                loaders_per_dataset.append(loader)
                iters_per_dataset.append(iter(loader))
                self.loaders[name] = loaders_per_dataset
                self.iters[name] = iters_per_dataset

                # Collect training datasets (not test)
                if name in dataset_names:
                    all_train_datasets.append((dataset, name))

            self.set_vae_data_distributions(name)

        # Create combined loader from all training datasets
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

    def get_batch(self, dataset_name, idx):
        try:
            X, y = next(self.iters[dataset_name][idx])
        except StopIteration:
            self.iters[dataset_name][idx] = iter(self.loaders[dataset_name][idx])
            X, y = next(self.iters[dataset_name][idx])

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

    def get_loader(self, dataset_name, idx: Optional[int] = None):
        if idx is None:
            return self.loaders[dataset_name]
        return self.loaders[dataset_name][idx]

    def get_vae(self, dataset_name):
        return self.vaes[dataset_name]

    def get_vae_data_distribution(self, dataset_name, idx):
        return self.vae_distributions[dataset_name][idx]

    def set_vae_data_distributions(self, dataset_name):
        vae = self.get_vae(dataset_name)
        loaders = self.get_loader(dataset_name)
        distribution_per_dataset = []
        for loader in loaders:
            mus = []
            logvars = []
            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(loader):
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
            distribution_per_dataset.append(distribution)

        self.vae_distributions[dataset_name] = distribution_per_dataset

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

    def _create_combined_vae_distribution(self):
        """Create a combined VAE distribution from all training datasets."""
        all_distributions = []
        for name in self.train_dataset_names:
            if name in self.vae_distributions:
                for dist in self.vae_distributions[name]:
                    all_distributions.append(dist)
        
        if all_distributions:
            self.combined_vae_distribution = torch.cat(all_distributions, dim=0)
            print(f"Created combined VAE distribution with {self.combined_vae_distribution.size(0)} samples")
        else:
            self.combined_vae_distribution = None

    def get_combined_vae_distribution(self):
        """Get the combined VAE distribution for conditioning."""
        return self.combined_vae_distribution


def pairwise_squared_distances(x, y):
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1).unsqueeze(0)
    return x_norm + y_norm - 2.0 * x @ y.T


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
    plt.savefig(experiment_folder / "loss.png")
    plt.close()

    if len(average_acc_diff) > 0:
        plt.figure()
        plt.plot(average_acc_diff, label="Accuracy difference train vs no_train")

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracies averaged")
        plt.legend()
        plt.grid(True)
        plt.savefig(experiment_folder / "accuracy_difference.png")
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
    resources: ResourceManager,
    distributions_targets,
    mu_max_dist,
):

    vae_distribution = resources.get_vae_data_distribution(test_dataset_name, 0)
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


def compute_geometry_consistency_loss(mu_batch, target_output, mu_max_dist=None):
    sim_target = torch.mm(target_output, target_output.T)
    dist_target = 1.0 - sim_target

    mu_norm = F.normalize(mu_batch, p=2, dim=1)
    sim_mu = torch.mm(mu_norm, mu_norm.T)
    dist_mu = 1.0 - sim_mu

    loss = F.mse_loss(dist_target, dist_mu)

    return loss


def meta_training(hyper: HyperNetwork, target: TargetNet, resources: ResourceManager):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_outer)

    distributions_targets = []
    loader = resources.get_loader(test_dataset_name, 0)
    vae = resources.get_vae(test_dataset_name)
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
            distributions_targets.append((mu, logvar, y))

            if batch_idx == steps_innerloop:
                break
    ref_mu = torch.cat([mu for mu, _, _ in distributions_targets], dim=0).to(device)
    ref_logvar = torch.cat(
        [logvar for _, logvar, _ in distributions_targets], dim=0
    ).to(device)
    ref_labels = torch.cat([y for _, _, y in distributions_targets], dim=0).cpu()
    inner_losses = defaultdict(list)
    outer_losses = defaultdict(list)
    acc_training_list = []
    acc_no_training_list = []
    average_acc_diff = []
    mu_max_dist = None
    log_every = 10
    log_embedding_epochs = set(range(0, epochs_hyper, log_every))
    target_embeddings_over_time = {}

    for epoch in tqdm(range(epochs_hyper), desc="Epochs"):
        params_hyper = dict(hyper.named_parameters())
        buffers_hyper = dict(hyper.named_buffers())

        if use_combined_loader:
            vae_dist_inner = resources.get_combined_vae_distribution()
            dataset_name = "combined"
        else:
            dataset_name = random.choice(train_dataset_names)
            idx = np.random.randint(0, len(resources.get_loader(dataset_name)))
            vae_dist_inner = resources.get_vae_data_distribution(dataset_name, idx)

        adapted_params_hyper = {k: v.clone() for k, v in params_hyper.items()}
        # Innerloop
        for _ in range(steps_innerloop):
            if use_combined_loader:
                inner_X, _, inner_mu, inner_logvar = resources.get_combined_batch()
            else:
                inner_X, _, inner_mu, inner_logvar = resources.get_batch(dataset_name, idx)
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
            X_outer, y_outer, mu_outer, logvar_outer = resources.get_batch(
                dataset_name, idx
            )
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
        optimizer.step()

        if epoch in log_embedding_epochs:
            if use_combined_loader:
                vae_dist_inner = resources.get_combined_vae_distribution()
            else:
                vae_dist_inner = resources.get_vae_data_distribution(dataset_name, idx)
            dist_inner = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]

            with torch.no_grad():
                params_target = functional_call(
                    hyper, (adapted_params_hyper, buffers_hyper), (dist_inner,)
                )

                z = torch.cat((ref_mu, ref_logvar), dim=1)
                logits = target.forward(z, params_target)
                emb = F.normalize(logits, p=2, dim=1).cpu()

            target_embeddings_over_time[epoch] = {
                "embeddings": emb,
                "labels": ref_labels.clone(),
            }
            torch.save(
                target_embeddings_over_time,
                experiment_folder / f"{epoch}_embeddings.pth",
            )

        acc_no_training, acc_training = evaluate_classification(
            hyper, target, resources, distributions_targets, mu_max_dist
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


def train_vaes(dataset_name):
    print(f"Training VAE for {dataset_name}...")
    datasets = get_dataset(
        name=dataset_name, preprocess=True, to_tensor=True, flatten=False
    )
    data = datasets[0]
    train_loader = DataLoader(
        Dataset(data["train"]),
        batch_size=batch_size_vae,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim).to(
        device
    )
    train_vae(
        vae,
        train_loader,
        train_loader,
        dataset_name,
        lr_vae,
        epochs_vae,
        log_interval,
        vae_description,
        models_folder,
    )


def main():
    if retrain_vae:
        train_vaes()

    resources = ResourceManager(
        dataset_names=train_dataset_names,
        test_name=test_dataset_name,
        model_folder=models_folder,
    )

    hyper = HyperNetwork(
        layer_sizes=target_layer_sizes,
        condition_dim=condition_dim,
        head_hidden=head_hidden,
        use_bias=use_bias,
    ).to(device)

    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)

    meta_training(hyper, target, resources)


if __name__ == "__main__":
    main()
