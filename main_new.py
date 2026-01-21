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

from utils import VAE, TargetNet, HyperNetwork, get_gaussian_from_vae, train_vae
from dataset_loading import get_dataset, Dataset


# Disable nnpack to avoid potential issues on some systems
try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

outloop_dataset_names = ["fashion_mnist", "kmnist"]
innerloop_dataset_names = ["math_shapes"]
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
log_interval = CFG["training"]["log_interval"]
save_path = CFG["training"]["save_path"]
steps_innerloop = CFG["meta"]["steps_innerloop"]
steps_outerloop = CFG["meta"]["steps_outerloop"]
image_width_height = CFG["data"]["image_width_height"]
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
num_classes = CFG["target_net"]["num_classes"]

condition_dim = vae_head_dim * 2
input_dim = vae_head_dim * 2 if cluster_using_guassians else image_width_height**2
target_layer_sizes = [input_dim, *hidden_layers, num_classes]

print("Loaded config:", Path(__file__).parent / "config.yaml")
print("target_layer_sizes =", target_layer_sizes)
print("condition_dim =", condition_dim)

n_samples_conditioning = batch_size_innerloop

class ResourceManager:
    def __init__(self, inner_names, outer_names, test_name, model_folder):
        self.loaders = {}
        self.vaes = {}
        self.iters = {}
        self.model_folder = model_folder
        
        all_names = list(set(inner_names + outer_names + [test_name]))
        
        print("Loading VAEs and Datasets")
        for name in all_names:
            vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
            file_name = name + vae_description + ".pth"
            path = self.model_folder / file_name
            if path.exists():
                vae.load_state_dict(torch.load(path, map_location=device)["hyper_state_dict"])
            else:
                print(f"vae not found at {path}. Starting VAE training")
                train_vaes()

            vae.to(device)
            vae.eval()
            self.vaes[name] = vae

            is_inner = name in inner_names
            bs = batch_size_innerloop if is_inner else batch_size_outerloop
            if name == test_name:
                bs = batch_size_innerloop # Use inner size for testing or separate config

            datasets = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)
            data_dict = datasets[0] 
            
            # Use 'test' split for the test set, 'train' for others
            split = "test" if name == test_name and name not in inner_names + outer_names else "train"
            dataset = Dataset(data_dict[split])
            
            loader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            self.loaders[name] = loader
            self.iters[name] = iter(loader) # Create infinite iterator

    def get_batch(self, dataset_name):
        """Returns (X, y, mu, logvar) for a single batch."""
        try:
            X, y = next(self.iters[dataset_name])
        except StopIteration:
            self.iters[dataset_name] = iter(self.loaders[dataset_name])
            X, y = next(self.iters[dataset_name])
            
        X = X.to(device)
        y = y.to(device)
        vae = self.vaes[dataset_name]
        
        # No grad for VAE part
        with torch.no_grad():
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
            
        return X, y, mu, logvar
    
    def get_test_loader(self, dataset_name):
        return self.loaders[dataset_name]
    
    def get_vae(self, dataset_name):
        return self.vaes[dataset_name]


def contrastive_loss_vectorized(X, y, temp):
    X = F.normalize(X, p=2, dim=1)
    cos_sim_matrix = torch.matmul(X, X.T) / temp
    
    exp_scores = torch.exp(cos_sim_matrix)
    
    y = y.view(-1, 1)
    mask_self = torch.eye(X.shape[0], device=X.device).bool()
    
    mask_positive = (y == y.T) & (~mask_self)

    exp_scores_no_self = exp_scores * (~mask_self).float()
    denominator = exp_scores_no_self.sum(dim=1)

    numerator = (exp_scores * mask_positive.float()).sum(dim=1)
    
    epsilon = 1e-8
    log_prob = torch.log(numerator + epsilon) - torch.log(denominator + epsilon)
    
    return -log_prob.mean()


def inner_loop(hyper, target, optimizer, criterion, resources):
    dataset_name = random.choice(innerloop_dataset_names)
    
    hyper.train()

    for _ in range(steps_innerloop):
        X, y, mu, logvar = resources.get_batch(dataset_name)
        
        # Prepare inputs
        conditioning_vector = torch.cat(
            (mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), 1
        )
        
        params = hyper(conditioning_vector)

        if cluster_using_guassians:
             target_input = torch.cat((mu, logvar), 1)
        else:
             target_input = X.view(X.shape[0], -1)

        logits = target.forward(target_input, params)

        if use_contrastive_loss:
            loss = contrastive_loss_vectorized(logits, y, contrastive_temp)
        else:
            loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if use_contrastive_loss:
        print(f"Inner loop ({dataset_name}) loss: {loss.item():.4f}")

    return hyper


def outer_loop(hyper, target, optimizer, criterion, resources):
    hyper.train()
    
    losses = []
    optimizer.zero_grad()

    for name in outloop_dataset_names:
        X, y, mu, logvar = resources.get_batch(name)

        conditioning_vector = torch.cat(
            (mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), 1
        )
        
        params = hyper(conditioning_vector)

        if cluster_using_guassians:
             target_input = torch.cat((mu, logvar), 1)
        else:
             target_input = X.view(X.shape[0], -1)

        logits = target.forward(target_input, params)

        if use_contrastive_loss:
            loss = contrastive_loss_vectorized(logits, y, contrastive_temp)
        else:
            loss = criterion(logits, y)
        
        losses.append(loss)

    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
    
    print(f"Outer loop loss: {total_loss.item():.4f}")
    return hyper

def kmeans_accuracy(x: torch.Tensor, y: torch.Tensor, n_clusters: int = 10):
    x = F.normalize(x, p=2, dim=1)
    x_np = x.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy().flatten()

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred = kmeans.fit_predict(x_np)

    D = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(n_clusters):
        for j in range(n_clusters):
            D[i, j] = np.sum((pred == i) & (y_np == j))

    row_ind, col_ind = linear_sum_assignment(-D)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping[p] for p in pred])

    accuracy = np.mean(mapped_pred == y_np)
    return accuracy

def get_clusters_cross_entropy(logits, y, return_all_ties=True):
    preds = torch.argmax(logits, dim=1).cpu().long()
    labels = y.cpu().long()
    
    if preds.numel() == 0: return {}

    n_pred = int(preds.max().item()) + 1
    n_label = int(labels.max().item()) + 1

    pair_index = preds * n_label + labels
    counts_1d = torch.bincount(pair_index, minlength=n_pred * n_label)
    counts = counts_1d.view(n_pred, n_label)

    result = {}
    for p in range(n_pred):
        row = counts[p]
        total_for_pred = int(row.sum().item())
        if total_for_pred == 0:
            result[p] = (None, 0)
            continue
        top_count = int(row.max().item())
        best_labels = (row == top_count).nonzero(as_tuple=False).flatten().tolist()
        if not return_all_ties and len(best_labels) > 1:
            best_labels = [best_labels[0]]
        result[p] = (best_labels, top_count, total_for_pred)
    return result

def evaluate_clustering(hyper, target, resources):
    hyper.eval()
    
    loader = resources.get_test_loader(test_dataset_name)
    vae = resources.get_vae(test_dataset_name)
    
    all_logits = []
    all_y = []
    
    params = None

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            mu, logvar = get_gaussian_from_vae(vae, X, batch_idx, visualize=False)
            
            if params is None:
                conditioning_vector = torch.cat(
                    (mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), dim=1
                )
                params = hyper(conditioning_vector)
            
            if cluster_using_guassians:
                inp = torch.cat((mu, logvar), 1)
            else:
                inp = X.view(X.shape[0], -1)

            logits = target.forward(inp, params)
            all_logits.append(logits)
            all_y.append(y)

    all_logits = torch.cat(all_logits, 0)
    all_y = torch.cat(all_y, 0)

    if use_contrastive_loss:
        acc = kmeans_accuracy(all_logits, all_y)
        print(f"TEST SET ACCURACY ({test_dataset_name}): {acc:.4f}")
    else:
        print(get_clusters_cross_entropy(all_logits, all_y))


def meta_train_hyper(hyper, target, resources):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_hyper)
    criterion = nn.CrossEntropyLoss().to(device)

    print(f"Starting Meta-Training for {epochs_hyper} epochs...")
    
    for epoch in range(epochs_hyper):
        hyper = inner_loop(hyper, target, optimizer, criterion, resources)
        hyper = outer_loop(hyper, target, optimizer, criterion, resources)
        
        print(f"EPOCH: {epoch}")
        if epoch != 0 and epoch % log_interval == 0:
            evaluate_clustering(hyper, target, resources)

def train_vaes():
    dataset_names = ["math_shapes"]
    for name in dataset_names:
        print(f"Training VAE for {name}...")
        datasets = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)
        data = datasets[0]
        train_loader = DataLoader(Dataset(data["train"]), batch_size=batch_size_vae, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(Dataset(data["test"]), batch_size=batch_size_vae, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim).to(device)
        train_vae(vae, train_loader, test_loader, name, lr_vae, epochs_vae, log_interval, vae_description, models_folder)

def main():
    if retrain_vae:
        train_vaes()

    resources = ResourceManager(
        inner_names=innerloop_dataset_names, 
        outer_names=outloop_dataset_names, 
        test_name=test_dataset_name,
        model_folder=models_folder
    )

    hyper = HyperNetwork(
        layer_sizes=target_layer_sizes,
        condition_dim=condition_dim,
        head_hidden=head_hidden,
        use_bias=use_bias,
    ).to(device)
    
    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)
    
    meta_train_hyper(hyper, target, resources)

if __name__ == "__main__":
    main()