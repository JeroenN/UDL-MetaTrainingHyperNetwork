# hypernet_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from vae import VAE
from vae import loss_mse, loss_function
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_loading import get_dataset, Dataset, DATASET_NAMES
import random
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np


outloop_dataset_names = ['fashion_mnist', 'kmnist']
innerloop_dataset_names = ['math_shapes']
test_dataset_name = "mnist"
models_folder = Path(__file__).parent / "models"

device = torch.device("cuda")
batch_size_outerloop = 256
batch_size_innerloop = 128
epochs_hyper = 1000
epochs_vae = 100
lr_hyper = 1e-4
lr_vae = 1e-4
log_interval = 2
save_path = "hypernet_checkpoint.pth"
vae_head_dim = 10
n_samples_conditioning = 100
steps_innerloop = 1
steps_outerloop = 10

image_width_height = 28

use_contrastive_loss = True
contrastive_temp = 0.07

cluster_using_guassians = True

# Target network
if cluster_using_guassians:
    target_layer_sizes = [vae_head_dim*2, 400, 200, 10]
else:
    target_layer_sizes = [image_width_height**2, 400, 200, 10]



# Hypernetwork internal sizes
embed_dim = 0           
head_hidden = 256        
use_bias = True

#vae
condition_dim = vae_head_dim * 2 * n_samples_conditioning #*2, because there are two output enconder heads in VAE
retrain_vae = False
vae_description = "_head_10"
batch_size_vae = 256

class TargetNet:

    def __init__(self, layer_sizes, activation=F.relu):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes) - 1

    def forward(self, x, params):
        assert len(params) == self.num_layers
        out = x.view(x.shape[0], -1)
        for i, (W, b) in enumerate(params):
            out = F.linear(out, W, b)
            if i != self.num_layers - 1:
                out = self.activation(out)
        return out

class HyperNetwork(nn.Module):
    def __init__(self, layer_sizes, embed_dim=32, condition_dim=1000, head_hidden=256, use_bias=True):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.use_bias = use_bias
        
        self.layer_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim) * 0.1)
            for _ in range(self.num_layers)
        ])

        self.heads = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            n_params = out_dim * in_dim + (out_dim if use_bias else 0)
            head = nn.Sequential(
                nn.Linear(embed_dim+condition_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, n_params)
            )
            # small init
            nn.init.normal_(head[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(head[-1].bias, 0.0)
            self.heads.append(head)

    def forward(self, conditioning):
        params = []
        for j in range(self.num_layers):
            if embed_dim > 0:
                z = self.layer_embeddings[j] 
                z_cond = torch.cat([z, conditioning], dim=0)
            else:
                z_cond = conditioning

            head_input = z_cond.unsqueeze(0)  

            flat = self.heads[j](head_input).squeeze(0)

            out_dim = self.layer_sizes[j+1]
            in_dim = self.layer_sizes[j]
            w_n = out_dim * in_dim
            W_flat = flat[:w_n]
            W = W_flat.view(out_dim, in_dim)
            if self.use_bias:
                b = flat[w_n:].view(out_dim)
            else:
                b = None
            params.append((W, b))
        return params


def evaluate_vae(vae, test_loader, epoch):
    vae.eval()
    running_loss = 0
    list_mu = []
    list_logvar = []
    max_visualize = 5
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(device)
        reconstruct, mu, logvar = vae(x)
        list_mu.append(mu)
        list_logvar.append(logvar)
        running_loss += loss_mse(reconstruct, x, mu, logvar)

        if batch_idx < max_visualize:
            img, ax = plt.subplots(2,2)
            ax[0,0].imshow(x[0].squeeze().to('cpu'))
            ax[0,1].imshow(reconstruct[0].squeeze().detach().cpu())
            file_path = Path(__file__).parent / "visualization" / f"evaluation_batch_idx_{batch_idx}_epoch_{epoch}.png"
            img.savefig(file_path)
    
    print(f"VAE TEST LOSS: {running_loss / (len(test_loader))}")
    return list_mu, list_logvar


def get_gaussian_from_vae(vae, x, idx, visualize: bool = False):
    vae.eval()
    reconstruct, mu, logvar = vae(x)

    if visualize:
        img, ax = plt.subplots(2,2)
        ax[0,0].imshow(x[0].squeeze().to('cpu'))
        ax[0,1].imshow(reconstruct[0].squeeze().detach().cpu())
        file_path = Path(__file__).parent / "visualization" / f"results_{idx}.png"
        img.savefig(file_path)
        
    return mu, logvar



def train_vae(vae, train_loader, test_loader, name):
    optimizer = torch.optim.Adam(vae.parameters(), lr = lr_vae)
    mu = []
    logvar = []
    for epoch in range(1, epochs_vae + 1):
        vae.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)

            optimizer.zero_grad()
            reconstruct, mu, logvar = vae(x)
            loss = loss_function(reconstruct, x, mu, logvar)

            loss.backward()
            running_loss+= loss
            optimizer.step()

        print(f"VAE: Epoch {epoch} loss={running_loss/len(train_loader):.4f}")
        running_loss = 0.0  
        if epoch % log_interval == 0:
            evaluate_vae(vae, test_loader, epoch)
            file_name = name + vae_description + ".pth"
            torch.save({'hyper_state_dict': vae.state_dict()}, models_folder / file_name)
    return vae

def create_batches(loader, number_of_batches, vae):
    data_x = []
    data_y = []
    data_mu = []
    data_logvar = []
    for batch_idx, (X, y) in enumerate(loader):
        if batch_idx == number_of_batches:
            break

        mu, logvar = get_gaussian_from_vae(vae, X.to(device), 0, visualize=False)

        data_mu.append(mu.unsqueeze(0)) #Add extra dimension before batch dimension, so that the batches can be looped over
        data_logvar.append(logvar.unsqueeze(0))

        data_y.append(y.unsqueeze(0))
        if cluster_using_guassians:
            data_x.append(torch.cat((mu, logvar), 1).unsqueeze(0))
        else:
            data_x.append(X.unsqueeze(0))

    data_x = torch.cat(data_x, 0)
    data_y = torch.cat(data_y, 0)
    data_mu = torch.cat(data_mu, 0)
    data_logvar = torch.cat(data_logvar, 0)
    return data_x, data_y, data_mu, data_logvar

# Loads the batches for one specific task into memory, ready to be used for meta learning
def create_batches_innerloop(dataset_name , number_of_batches):
    data = get_dataset(name=dataset_name, preprocess=True, to_tensor=True, flatten=False)
    train = Dataset(data['train'])
    loader = DataLoader(train, batch_size=batch_size_innerloop, shuffle=True, num_workers=2, pin_memory=True)
    
    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
    file_name = dataset_name + vae_description + ".pth"
    vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
    vae.to(device)

    data_x, data_y, data_mu, data_logvar = create_batches(loader, number_of_batches, vae)
    return data_x, data_y, data_mu, data_logvar

# Loads the batches into memory for all tasks, ready to be used for meta learning
def create_batches_outerloop():
    loaders = []
    for name in outloop_dataset_names:
        data = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)
        train = Dataset(data['train'])
        loaders.append((name, DataLoader(train, batch_size=batch_size_outerloop, shuffle=True, num_workers=2, pin_memory=True)))
        
    all_data_x = []
    all_data_y = []
    all_data_mu = []
    all_data_logvar = []



    for (name, loader) in loaders:
        vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
        file_name = name + vae_description + ".pth"
        vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
        vae.to(device)

        data_x, data_y, data_mu, data_logvar = create_batches(loader, 1, vae)
        all_data_x.append(data_x)
        all_data_y.append(data_y)
        all_data_mu.append(data_mu)
        all_data_logvar.append(data_logvar)


            
    data_x = torch.cat(all_data_x, 0)
    data_y = torch.cat(all_data_y, 0)
    data_mu = torch.cat(all_data_mu, 0)
    data_logvar = torch.cat(all_data_logvar, 0)

    return data_x, data_y, data_mu, data_logvar
      
def inner_loop(hyper, target, optimizer, criterion):
    dataset_id = random.randint(0, len(innerloop_dataset_names)-1)
    dataset_name = innerloop_dataset_names[dataset_id]
    data_x, data_y, data_mu, data_logvar = create_batches_innerloop(dataset_name, steps_innerloop)

    for batch_idx in range(data_x.shape[0]):
        X = data_x[batch_idx].to(device)
        y = data_y[batch_idx].to(device)
        mu = data_mu[batch_idx].to(device)
        logvar = data_logvar[batch_idx].to(device)
        conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), 0).view(-1)
        X = X.view(X.shape[0], -1)
        
        #optimizer.zero_grad()

        params = hyper(conditioning_vector)
        params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

        logits = target.forward(X, params)
        if use_contrastive_loss:
            loss = contrastive_loss_new(logits, y)
            print(f"Inner loop loss: {loss.item()}")
        else:
            loss = criterion(logits, y)

        loss.backward()
        #torch.autograd.grad(loss, hyper.parameters(), create_graph=True, allow_unused=True)
        optimizer.step()
    
    return hyper

def outer_loop(hyper, target, optimizer, criterion):
    hyper.train()
    data_x, data_y, data_mu, data_logvar = create_batches_outerloop()
    optimizer.zero_grad()
    losses = []
    for batch_idx in range(data_x.shape[0]):
        X = data_x[batch_idx].to(device)
        y = data_y[batch_idx].to(device)
        mu = data_mu[batch_idx].to(device)
        logvar = data_logvar[batch_idx].to(device)
        conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), 0).view(-1)
        X = X.view(X.shape[0], -1)
        
        params = hyper(conditioning_vector)
        params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

        logits = target.forward(X, params)

        if use_contrastive_loss:
            losses.append(contrastive_loss_new(logits, y))
        else:
            losses.append(criterion(logits, y))

        #losses.append(criterion(logits, y))


    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
    print(f"Outerloop loss: {total_loss.item()}")
    return hyper


def get_clusters_cross_entropy(logits: torch.Tensor, y: torch.Tensor,
                 return_all_ties: bool = True
                ) -> Dict[int, Tuple[Optional[List[int]], int]]:
    """
    For each predicted class (argmax over logits) find which true label
    (argmax over y) occurs most often with that prediction and how often.

    Args:
        logits: tensor of shape (N, C_pred) or similar. Predictions = argmax(logits, dim=1).
        y: tensor of shape (N, C_label) (one-hot or logits). Labels = argmax(y, dim=1).
        return_all_ties: if True, return list of tied best labels (if any);
                         if False, pick the first tied label only.

    Returns:
        dict where key = predicted class (int) and value = (best_label(s), count)
          - best_label(s) is a list of ints if return_all_ties=True (or when multiple ties),
            otherwise a single-int list or None if there are no examples for that prediction.
          - count is the number of occurrences of that best label (0 if prediction never occurs).
    """
    preds = torch.argmax(logits, dim=1).cpu().to(torch.long)
    labels = y.cpu().to(torch.long)

    if preds.numel() == 0:
        return {}

    n_pred = int(preds.max().item()) + 1
    n_label = int(labels.max().item()) + 1

    # Vectorized count: flatten pair index and bincount
    pair_index = preds * n_label + labels
    counts_1d = torch.bincount(pair_index, minlength=n_pred * n_label)
    counts = counts_1d.view(n_pred, n_label)

    result: Dict[int, Tuple[Optional[List[int]], int]] = {}
    for p in range(n_pred):
        row = counts[p]
        total_for_pred = int(row.sum().item())
        if total_for_pred == 0:
            # prediction p never occurred in the batch
            result[p] = (None, 0)
            continue

        top_count = int(row.max().item())
        best_labels = (row == top_count).nonzero(as_tuple=False).flatten().tolist()

        if not return_all_ties and len(best_labels) > 1:
            best_labels = [best_labels[0]]

        result[p] = (best_labels, top_count, total_for_pred)

    return result

def kmeans_accuracy(x: torch.Tensor, y: torch.Tensor, n_clusters: int = 10):
    """
    Perform KMeans clustering on x and compute accuracy against y.

    Args:
        x (torch.Tensor): Input features [n, 10]
        y (torch.Tensor): True labels [n, 1]
        n_clusters (int): Number of clusters

    Returns:
        float: clustering accuracy
    """
    x = F.normalize(x, p=2, dim=1)
    x_np = x.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy().flatten()
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred = kmeans.fit_predict(x_np)
    
    # Compute best mapping between cluster labels and true labels
    D = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(n_clusters):
        for j in range(n_clusters):
            D[i, j] = np.sum((pred == i) & (y_np == j))
    
    row_ind, col_ind = linear_sum_assignment(-D)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping[p] for p in pred])
    
    accuracy = np.mean(mapped_pred == y_np)
    return accuracy

#def contrastive_loss(x: torch.Tensor, y: torch.Tensor):
#    x = F.normalize(x, p=2, dim = 1)
#    torch.matmul(x, x.T)

def evaluate_clustering(hyper, target):
    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
    file_name = test_dataset_name + vae_description + ".pth"
    vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
    vae.to(device)

    data = get_dataset(name=test_dataset_name, preprocess=True, to_tensor=True, flatten=False)
    test = Dataset(data['test'])
    test_loader = DataLoader(test, batch_size=batch_size_innerloop, shuffle=False, num_workers=2, pin_memory=True)
    params = torch.Tensor()
    all_logits = torch.Tensor()
    all_y = torch.Tensor()
    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.to(device)
        mu, logvar = get_gaussian_from_vae(vae,X, batch_idx, visualize=False)
        conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning])).view(-1)


        if batch_idx == 0:
            params = hyper(conditioning_vector)
            params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]
        
        if cluster_using_guassians:
            input = torch.cat((mu, logvar), 1)
        else:
            input = X.view(X.shape[0], -1)
            
        logits = target.forward(input, params)
        if batch_idx == 0:
            all_logits = logits
            all_y = y
            continue

        all_logits = torch.concat((all_logits, logits), 0)
        all_y = torch.concat((all_y, y), 0)
    
    if use_contrastive_loss:
        print(f"ACCURACY: {kmeans_accuracy(all_logits, all_y)}")
    else:
        print(get_clusters_cross_entropy(all_logits, all_y))


def contrastive_loss_new(X, y):
    X = F.normalize(X, p=2, dim=1)

    cos_sim_matrix = torch.matmul(X, X.T)
    y = y.unsqueeze(0)
    # torch.eye gives the identity matrix, with ~ we are saying exclude this. So it
    # excludes the pairs where the indices are the same
    mask_positive = (y == y.T) & (~torch.eye(X.shape[0], dtype=bool, device=device))  
    mask_negative = ~mask_positive

    cos_sim_matrix = cos_sim_matrix / contrastive_temp

    loss = 0
    for i in range(X.shape[0]):
        pos_cos_sim = cos_sim_matrix[i][mask_positive[i]]
        neg_cos_sim = cos_sim_matrix[i][mask_negative[i]]
        numerator = torch.exp(pos_cos_sim).sum()
        denominator = torch.exp(neg_cos_sim).sum() + numerator
        loss += torch.log(numerator / denominator)

    return loss / X.shape[0]


def meta_train_hyper(hyper, target):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_hyper)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    for epoch in range(epochs_hyper):
        hyper = inner_loop(hyper, target, optimizer, criterion)
        hyper = outer_loop(hyper, target, optimizer, criterion)
        print(f"EPOCH: {epoch}")
        if epoch != 0 and epoch % log_interval == 0:
            evaluate_clustering(hyper, target)


def train_vaes():
    dataset_names = ['math_shapes']#['fashion_mnist', 'kmnist', 'mnist']
    for name in dataset_names:
        data = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)

        train = Dataset(data['train'])
        test = Dataset(data['test'])
        train_loader = DataLoader(train, batch_size=batch_size_vae, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test, batch_size=batch_size_vae, shuffle=False, num_workers=2, pin_memory=True)
        vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim).to(device)
        vae = train_vae(vae, train_loader, test_loader, name)


def evaluate_vae_clustering():
    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
    file_name = test_dataset_name + vae_description + ".pth"
    vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
    vae.to(device)

    data = get_dataset(name=test_dataset_name, preprocess=True, to_tensor=True, flatten=False)
    test = Dataset(data['test'])
    test_loader = DataLoader(test, batch_size=batch_size_vae, shuffle=False, num_workers=2, pin_memory=True)
    all_mu = []
    all_y = []
    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.to(device)
        mu, logvar = get_gaussian_from_vae(vae,X, batch_idx, visualize=False)

        all_mu.append(mu)
        all_y.append(y)

    all_mu = torch.cat(all_mu,0)
    all_y = torch.cat(all_y,0)
    print("CLUSTERING ACCURACY VAE: ")
    print(kmeans_accuracy(all_mu, all_y))

def main():
    if retrain_vae:
        train_vaes()
    
    #evaluate_vae_clustering()
    hyper = HyperNetwork(layer_sizes=target_layer_sizes, embed_dim=embed_dim, condition_dim=condition_dim, head_hidden=head_hidden, use_bias=use_bias).to(device)
    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)
    hyper.to(device)
    meta_train_hyper(hyper, target)


    

if __name__ == "__main__":
    main()
