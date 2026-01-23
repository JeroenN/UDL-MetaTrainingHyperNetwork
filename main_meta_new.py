# hypernet_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
import yaml
import re
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from utils import VAE, TargetNet, HyperNetwork, get_gaussian_from_vae, train_vae
from dataset_loading import get_dataset, Dataset

import torch.nn.functional as F
from torch.func import functional_call, grad


# Disable nnpack to avoid potential issues on some systems
try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

train_dataset_names = ["kmnist", "hebrew_chars", "fashion_mnist"]#, "kmnist",]# "math_shapes"]
test_dataset_name = "mnist"
models_folder = Path(__file__).parent / "models"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

num_workers = 2 if device.type == "cuda" else 0
pin_memory = device.type == "cuda"

def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

CFG = load_config(Path(__file__).parent / "config.yaml")

# Pull values out
batch_size_outerloop = CFG["training"]["batch_size_outerloop"]
batch_size_innerloop = CFG["training"]["batch_size_innerloop"]
batch_size_vae = CFG["training"]["batch_size_vae"]
epochs_hyper = CFG["training"]["epochs_hyper"]
epochs_vae = CFG["training"]["epochs_vae"]
lr_hyper = float(CFG["training"]["lr_hyper"])
lr_vae = float(CFG["training"]["lr_vae"])
lr_target = float(CFG["training"]["lr_target"])
log_interval = CFG["training"]["log_interval"]
save_path = CFG["training"]["save_path"]
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

condition_dim = vae_head_dim * 2
input_dim = vae_head_dim * 2 if cluster_using_guassians else image_width_height**2
target_layer_sizes = [input_dim, *hidden_layers, output_head]

print("Loaded config:", Path(__file__).parent / "config.yaml")
print("target_layer_sizes =", target_layer_sizes)
print("condition_dim =", condition_dim)

n_samples_conditioning = batch_size_innerloop

class ResourceManager:
    def __init__(self, dataset_names, test_name, model_folder):
        self.loaders = {}

        self.vaes = {}
        self.iters = {}
        self.model_folder = model_folder

        self.vae_distributions = {}
        
        all_names = list(set(dataset_names + [test_name]))
        
        for name in tqdm(all_names, desc = "Loading datasets"):
            vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
            file_name = name + vae_description + ".pth"
            path = self.model_folder / file_name
            if path.exists():
                vae.load_state_dict(torch.load(path, map_location=device)["hyper_state_dict"])
            else:
                print(f"vae not found at {path}. Starting VAE training")
                train_vaes(name)

            vae.to(device)
            vae.eval()
            self.vaes[name] = vae

            bs = batch_size_innerloop

            datasets = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False, class_limit = num_classes)
            loaders_per_dataset = []
            iters_per_dataset = []
            for idx, ds in enumerate(datasets):
                # Use 'test' split for the test set, 'train' for others
                split = "train" #"test" if name == test_name and name not in inner_names + outer_names else "train"
                dataset = Dataset(ds[split])
    
                loader = DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
                loaders_per_dataset.append(loader)
                iters_per_dataset.append(iter(loader)) # Create infinite iterator
                self.loaders[name] = loaders_per_dataset
                self.iters[name] = iters_per_dataset

            self.set_vae_data_distributions(name)

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
        
        # No grad for VAE part
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
                    mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize= False)
                    mus.append(mu)
                    logvars.append(logvar)

            mu = torch.concat(mus, dim =0)
            logvar = torch.concat(logvars, dim=0)

            distribution = torch.concat((mu, logvar), dim =1)
            distribution_per_dataset.append(distribution)
            
        self.vae_distributions[dataset_name] = distribution_per_dataset


def pairwise_squared_distances(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
    return x_norm + y_norm - 2.0 * x @ y.T


def compute_soft_dist(embeddings, centroids, temperature=1.0):
    dists = pairwise_squared_distances(embeddings, centroids)#torch.cdist(embeddings, centroids, p=2)**2
    assignments = F.softmax(-dists / temperature, dim=1)

    return assignments
    

def compute_soft_centroids(embeddings, K = num_classes, iterations=1, temperature=1.0):
    indices = torch.randperm(embeddings.size(0))[:K]
    initial_centroids = embeddings[indices].detach()
    centroids = initial_centroids
    
    for _ in range(iterations):
        dists = pairwise_squared_distances(embeddings, centroids)#torch.cdist(embeddings, centroids, p=2)**2
        
        assignments = F.softmax(-dists / temperature, dim=1) 
        
        weight_sums = assignments.sum(dim=0, keepdim=True).T 
        
        weighted_embeddings_sum = torch.matmul(assignments.T, embeddings)
        
        centroids = weighted_embeddings_sum / (weight_sums + 1e-8)
        
    return centroids, assignments

def soft_clustering_loss(embeddings, centroids, temperature=0.5, alpha=1.0):
    sim = torch.matmul(embeddings, centroids.T)
    
    assignments = F.softmax(sim / temperature, dim=1) # (N, K)
    
    weighted_sim = torch.sum(assignments * sim, dim=1)
    clustering_loss = torch.mean(1.0 - weighted_sim)
    
    p_bar = assignments.mean(dim=0) # (K,)
    entropy_reg = -torch.sum(p_bar * torch.log(p_bar + 1e-8))
    
    total_loss = clustering_loss - (alpha * entropy_reg)
    
    return total_loss


def supervised_hungarian_accuracy(soft_assignments: torch.Tensor, true_labels: torch.Tensor) -> float:
    device = soft_assignments.device
    n_samples, n_clusters = soft_assignments.shape

    cost_matrix = torch.zeros((n_clusters, n_clusters), device=device)
    probs = soft_assignments.detach()
    
    for i in range(n_clusters):
        cluster_mask = probs[:, i]
        for j in range(n_clusters):
            label_mask = (true_labels == j).float()
            cost_matrix[i, j] = torch.sum(cluster_mask * label_mask)
    
    row_ind, col_ind = linear_sum_assignment(-cost_matrix.cpu().numpy())
    mapping = torch.zeros(n_clusters, dtype=torch.long, device=device)
    for cluster_idx, label_idx in zip(row_ind, col_ind):
        mapping[cluster_idx] = label_idx

    predicted_clusters = torch.argmax(soft_assignments, dim=1)
    mapped_preds = mapping[predicted_clusters]

    correct = (mapped_preds == true_labels).sum().item()
    accuracy = correct / n_samples
    return accuracy


def supervised_hungarian_loss(soft_assignments, true_labels):
    device = soft_assignments.device
    n_samples, n_clusters = soft_assignments.shape
    
    cost_matrix = torch.zeros((n_clusters, n_clusters), device=device)

    probs = soft_assignments.detach() 
    
    for i in range(n_clusters):
        cluster_mask = probs[:, i] 
        for j in range(n_clusters):
            label_mask = (true_labels == j).float()
            cost_matrix[i, j] = torch.sum(cluster_mask * label_mask)

    row_ind, col_ind = linear_sum_assignment(-cost_matrix.cpu().numpy())
    
    mapping = torch.zeros(n_clusters, dtype=torch.long, device=device)
    for cluster_idx, label_idx in zip(row_ind, col_ind):
        mapping[cluster_idx] = label_idx

    idx = torch.argsort(mapping)
    reordered_assignments = soft_assignments[:, idx]

    loss = F.nll_loss(torch.log(reordered_assignments + 1e-8), true_labels)
    
    return loss

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

def get_fixed_centroids(num_classes, dim, device):
    C = torch.eye(num_classes, dim)
    C = F.normalize(C, p=2, dim=1)
    return C.to(device)


def plot_losses_and_accuracies(inner_losses_dict,
                               outer_losses_dict,
                               acc_no_training_list,
                               acc_training_list):
    Path(Path(__file__).parent / "visualization" / "plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    for dataset_name in inner_losses_dict:
        plt.plot(
            inner_losses_dict[dataset_name],
            label=f"{dataset_name} - inner loss"
        )
        plt.plot(
            outer_losses_dict[dataset_name],
            linestyle="--",
            label=f"{dataset_name} - outer loss"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Losses per Dataset")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(__file__).parent / "visualization" / "plots" / "loss.png")


    plt.figure()
    plt.plot(acc_no_training_list, label="Accuracy (no training)")
    plt.plot(acc_training_list, label="Accuracy (with training)")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracies")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(__file__).parent / "visualization" / "plots" / "accuracy.png")

def compute_inner_loss(hyper, target, buffers_hyper, vae_dist, centroids, current_params_hyper, data_input):
    params_target = functional_call(hyper, (current_params_hyper, buffers_hyper), (vae_dist,))
    
    X, y, mu, logvar = data_input
    distribution = torch.concat((mu, logvar), dim=1)
    
    logits = target.forward(distribution, params_target)
    logits = F.normalize(logits, p=2, dim=1)
    
    return soft_clustering_loss(logits, centroids, temperature=0.1, alpha=0.01)



def evaluate_clustering(hyper, target: TargetNet, resources: ResourceManager, centroids, distributions_targets):
    
    vae_distribution = resources.get_vae_data_distribution(test_dataset_name, 0)
    params_old = hyper(vae_distribution)


    all_assignments = []
    all_y = []
    with torch.no_grad():
        for batch_idx, (distribution, y) in enumerate(distributions_targets):
            logits = target.forward(distribution, params_old)
            logits = F.normalize(logits, p=2, dim=1)
            #assignments = compute_soft_dist(embeddings=logits, centroids = centroids, temperature=1.0)
            sim = torch.matmul(logits, centroids.T)
            assignments = F.softmax(sim / 0.01, dim=1)
            all_assignments.append(assignments.cpu())
            all_y.append(y.cpu())

            if batch_idx == steps_innerloop:
                break
    
    all_assignments = torch.concat(all_assignments, dim = 0)
    all_y = torch.concat(all_y, dim =0)
    acc_no_training = supervised_hungarian_accuracy(all_assignments, all_y)
    current_params_hyper = {
        k: v.clone().detach().requires_grad_(True) 
        for k, v in hyper.named_parameters()
    }
    buffers_hyper = dict(hyper.named_buffers())

    dist_inner = vae_distribution[torch.randperm(vae_distribution.size(0))[:1024]]
    for batch_idx, (distribution, y) in enumerate(distributions_targets):
        params_target = functional_call(hyper, (current_params_hyper, buffers_hyper), (dist_inner,))

        logits = target.forward(distribution, params_target)
        logits = F.normalize(logits, p=2, dim=1)

        loss = soft_clustering_loss(logits, centroids, temperature=0.5, alpha=0.01)

        grads = torch.autograd.grad(
            loss,
            current_params_hyper.values(),
            create_graph=False 
        )

        current_params_hyper = {
            name: (p - g * lr_hyper).detach().requires_grad_(True)
            for (name, p), g in zip(current_params_hyper.items(), grads)
        }
    

    all_assignments = []
    with torch.no_grad():
        for batch_idx, (distribution, y) in enumerate(distributions_targets):
            logits = target.forward(distribution, params_target)
            logits = F.normalize(logits, p=2, dim=1)
            sim = torch.matmul(logits, centroids.T)
            assignments = F.softmax(sim / 0.01, dim=1)
            #assignments = compute_soft_dist(embeddings=logits, centroids = centroids, temperature=1.0)
            all_assignments.append(assignments.cpu())


    all_assignments = torch.concat(all_assignments, dim = 0)
    acc_training = supervised_hungarian_accuracy(all_assignments, all_y)
    return acc_no_training, acc_training

def meta_training(hyper: HyperNetwork, target: TargetNet, resources: ResourceManager):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_hyper)
    
    centroids = get_fixed_centroids(num_classes, output_head, device)
    
    distributions_targets = []
    loader = resources.get_loader(test_dataset_name, 0)
    vae = resources.get_vae(test_dataset_name)
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
            distribution = torch.concat((mu, logvar), dim=1)
            distributions_targets.append((distribution, y)) 

            if batch_idx == steps_innerloop:
                break
    
    inner_losses = defaultdict(list)
    outer_losses = defaultdict(list)
    acc_training_list = []
    acc_no_training_list = []

    for epoch in tqdm(range(epochs_hyper), desc="Epochs"):
        params_hyper = dict(hyper.named_parameters())
        buffers_hyper = dict(hyper.named_buffers())

        dataset_name = random.choice(train_dataset_names)
        idx = np.random.randint(0, len(resources.get_loader(dataset_name)))

        vae_dist_outer = resources.get_vae_data_distribution(dataset_name, idx)
        vae_dist_inner = resources.get_vae_data_distribution(dataset_name, idx)

        
        adapted_params_hyper = {k: v.clone() for k, v in params_hyper.items()}
        
        
        for _ in range(steps_innerloop):
            inner_batch = resources.get_batch(dataset_name, idx)
            dist_inner = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]
            inner_loss = compute_inner_loss(hyper, target, buffers_hyper, dist_inner, centroids, adapted_params_hyper, inner_batch)
            grads = torch.autograd.grad(
                inner_loss,
                adapted_params_hyper.values(),
                create_graph=True 
            )

            adapted_params_hyper = {
                name: p - g * lr_target
                for (name, p), g in zip(adapted_params_hyper.items(), grads)
            }

        #Outer Loop
        dist_outer = vae_dist_outer[torch.randperm(vae_dist_outer.size(0))[:1024]]
        params_target_outer = functional_call(hyper, (adapted_params_hyper, buffers_hyper), (dist_outer,))
        
        X, y, mu, logvar = resources.get_batch(dataset_name, idx)
        distribution = torch.concat((mu, logvar), dim=1)
        
        logits = target.forward(distribution, params_target_outer)
        logits = F.normalize(logits, p=2, dim=1)
        
        sim = torch.matmul(logits, centroids.T)
        assignments = F.softmax(sim / 0.1, dim=1)
        
        # This loss has a path back to 'params_hyper' through 'adapted_params_hyper'
        outer_loss = supervised_hungarian_loss(assignments, y)

        optimizer.zero_grad()
        outer_loss.backward() # Backprops through the inner loop steps
        optimizer.step()

        acc_no_training, acc_training = evaluate_clustering(hyper, target, resources, centroids, distributions_targets)
        inner_losses[dataset_name].append(inner_loss.detach().item())
        outer_losses[dataset_name].append(outer_loss.detach().item())
        acc_no_training_list.append(acc_no_training)
        acc_training_list.append(acc_training)
        if epoch % log_interval == 0:
            print(f"accuracy no training {acc_no_training:.4f}")
            print(f"accuracy training {acc_training: .4f}")
            if epoch > 100:
                print(f"average acc training last 100 {sum(acc_training_list[-100:])/100:.4f}")
                print(f"average acc no training last 100 {sum(acc_no_training_list[-100:])/100:.4f}")
            print(f"outer_loss: {outer_loss: .4f}")
            print(f"inner_loss: {inner_loss: .4f}")
            plot_losses_and_accuracies(inner_losses, outer_losses, acc_no_training_list, acc_training_list)    


def train_vaes(dataset_name):
    print(f"Training VAE for {dataset_name}...")
    datasets = get_dataset(name=dataset_name, preprocess=True, to_tensor=True, flatten=False)
    data = datasets[0]
    train_loader = DataLoader(Dataset(data["train"]), batch_size=batch_size_vae, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    #test_loader = DataLoader(Dataset(data["test"]), batch_size=batch_size_vae, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim).to(device)
    train_vae(vae, train_loader, train_loader, dataset_name, lr_vae, epochs_vae, log_interval, vae_description, models_folder)

def main():
    if retrain_vae:
        train_vaes()

    resources = ResourceManager(
        dataset_names = train_dataset_names,
        test_name = test_dataset_name,
        model_folder = models_folder
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