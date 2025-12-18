# hypernet_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from vae import VAE
from vae import loss_mse
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_loading import get_dataset, Dataset, DATASET_NAMES
import random
from typing import Dict, List, Optional, Tuple

train_dataset_names = ['fashion_mnist', 'kmnist']
test_dataset_name = "mnist"
models_folder = Path(__file__).parent / "models"

device = torch.device("cuda")
batch_size = 256
epochs_hyper = 100
epochs_vae = 5
lr_hyper = 1e-3
lr_vae = 1e-4
log_interval = 10
save_path = "hypernet_checkpoint.pth"
vae_head_dim = 5
n_samples_conditioning = 100
steps_innerloop = 1
steps_outerloop = 10

image_width_height = 28

target_layer_sizes = [784, 400, 200, 10]

# Hypernetwork internal sizes
embed_dim = 0           
head_hidden = 256        
use_bias = True

#vae
condition_dim = vae_head_dim * 2 * n_samples_conditioning #*2, because there are two output enconder heads in VAE

class TargetNet:

    def __init__(self, layer_sizes, activation=F.relu):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes) - 1

    def forward(self, x, params):
        assert len(params) == self.num_layers
        out = x.view(x.shape[0], -1)
        for i, (W, b) in enumerate(params):
            # W expected shape (out_features, in_features)
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


def evaluate_vae(vae, test_loader):
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
            file_path = Path(__file__).parent / "visualization" / f"evaluation_batch_idx_{batch_idx}.png"
            img.savefig(file_path)
    
    print(f"VAE TEST LOSS: {running_loss / (len(test_loader))}")
    return list_mu, list_logvar


def get_gaussian_from_vae(vae, x, idx, visualize: bool = True):
    vae.eval()
    reconstruct, mu, logvar = vae(x)

    if visualize:
        img, ax = plt.subplots(2,2)
        ax[0,0].imshow(x[0].squeeze().to('cpu'))
        ax[0,1].imshow(reconstruct[0].squeeze().detach().cpu())
        file_path = Path(__file__).parent / "visualization" / f"results_{idx}.png"
        img.savefig(file_path)
        
    return mu, logvar



def train_vae(vae, train_loader, test_loader):
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
            loss = loss_mse(reconstruct, x, mu, logvar)

            loss.backward()
            running_loss+= loss
            optimizer.step()

            if batch_idx % log_interval == 0:
                avg = running_loss / log_interval
                print(f"VAE: Epoch {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)}]  loss={avg:.4f}")
                running_loss = 0.0
    
    evaluate_vae(vae, test_loader)
    return vae

def evaluate_hyper(model_hyper, target_net, vae, dataloader, device):
    model_hyper.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            mu, logvar = get_gaussian_from_vae(vae,X, 0, visualize=False)
            conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning])).view(-1)
            params = model_hyper(conditioning_vector)
            params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

            logits = target_net.forward(X, params)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.shape[0]

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += X.shape[0]

    return total_loss / total, correct / total

# Loads the batches for one specific task into memory, ready to be used for meta learning
def create_batches_innerloop(dataset_name , number_of_batches):
    data = get_dataset(name=dataset_name, preprocess=True, to_tensor=True, flatten=False)
    train = Dataset(data['train'])
    dataset_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    data_x = torch.Tensor() 
    data_y = torch.Tensor()
    data_mu = torch.Tensor()
    data_logvar = torch.Tensor()

    for batch_idx, (X, y) in enumerate(dataset_loader):
        if batch_idx == number_of_batches:
            break

        if batch_idx == 0:
            data_x = X.unsqueeze(0) 
            data_y = y.unsqueeze(0)

            vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
            file_name = test_dataset_name + ".pth"
            vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
            vae.to(device)

            mu, logvar = get_gaussian_from_vae(vae, X.to(device), 0, visualize=False)
            data_mu = mu.unsqueeze(0)
            data_logvar = logvar.unsqueeze(0)
            continue
        
        data_x = torch.cat((data_x, X.unsqueeze(0)), 0)
        data_y = torch.cat((data_y, y.unsqueeze(0)), 0)
        data_mu = torch.cat((data_mu, mu.unsqueeze(0)), 0)
        data_logvar = torch.cat((data_logvar, logvar.unsqueeze(0)), 0)

    return data_x, data_y, data_mu, data_logvar

# Loads the batches into memory for all tasks, ready to be used for meta learning
def create_batches_outerloop():
    loaders = []
    for name in train_dataset_names:
        data = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)
        train = Dataset(data['test'])
        loaders.append((name, DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)))
        
    data_x = torch.Tensor() 
    data_y = torch.Tensor()
    data_mu = torch.Tensor()
    data_logvar = torch.Tensor()
    for (name, data_loader) in loaders:
        for batch_idx, (X, y) in enumerate(data_loader):
            if batch_idx == 1:
                break
            
            if batch_idx == 0:
                data_x = X.unsqueeze(0) 
                data_y = y.unsqueeze(0)
                file_name = name + ".pth"

                vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
                file_name = test_dataset_name + ".pth"
                vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
                vae.to(device)

                mu, logvar = get_gaussian_from_vae(vae, X.to(device), 0, visualize=False)
                data_mu = mu.unsqueeze(0)
                data_logvar = logvar.unsqueeze(0)
                continue
            
            data_x = torch.cat((data_x, X.unsqueeze(0)), 0)
            data_y = torch.cat((data_y, y.unsqueeze(0)), 0)
            data_mu = torch.cat((data_mu, mu.unsqueeze(0)), 0)
            data_logvar = torch.cat((data_logvar, logvar.unsqueeze(0)), 0)

    return data_x, data_y, data_mu, data_logvar
      
def inner_loop(hyper, target, optimizer, criterion):
    dataset_id = random.randint(0, len(train_dataset_names)-1)
    dataset_name = train_dataset_names[dataset_id]
    data_x, data_y, data_mu, data_logvar = create_batches_innerloop(dataset_name, steps_innerloop)

    for batch_idx in range(data_x.shape[0]):
        X = data_x[batch_idx].to(device)
        y = data_y[batch_idx].to(device)
        mu = data_mu[batch_idx].to(device)
        logvar = data_logvar[batch_idx].to(device)
        conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning]), 0).view(-1)
        X = X.view(X.shape[0], -1)
        
        optimizer.zero_grad()

        params = hyper(conditioning_vector)
        params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

        logits = target.forward(X, params)
        loss = criterion(logits, y)

        loss.backward()
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
        losses.append(criterion(logits, y))


    total_loss = 0
    for loss in losses:
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    return hyper

def get_clusters(logits: torch.Tensor, y: torch.Tensor,
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


def evaluate_clustering(hyper, target):
    vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim)
    file_name = test_dataset_name + ".pth"
    vae.load_state_dict(torch.load(models_folder / file_name)['hyper_state_dict'])
    vae.to(device)

    data = get_dataset(name=test_dataset_name, preprocess=True, to_tensor=True, flatten=False)
    test = Dataset(data['test'])
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    params = torch.Tensor()
    all_logits = torch.Tensor()
    all_y = torch.Tensor()
    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.to(device)
        mu, logvar = get_gaussian_from_vae(vae,X, batch_idx, visualize=batch_idx==1)
        conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning])).view(-1)
        X = X.view(X.shape[0], -1)

        if batch_idx == 0:
            params = hyper(conditioning_vector)
            params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]
        
        logits = target.forward(X, params)
        if batch_idx == 0:
            all_logits = logits
            all_y = y
            continue

        all_logits = torch.concat((all_logits, logits), 0)
        all_y = torch.concat((all_y, y), 0)
    
    print(get_clusters(all_logits, all_y))
        
        
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

def train_hyper_old(hyper, target, vae, train_loader, test_loader):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_hyper)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs_hyper + 1):
        hyper.train()
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader, start=1):
            X = X.to(device)
            y = y.to(device)
            mu, logvar = get_gaussian_from_vae(vae,X, batch_idx, visualize=batch_idx==1)
            conditioning_vector = torch.concatenate((mu[:n_samples_conditioning], logvar[:n_samples_conditioning])).view(-1)
            X = X.view(X.shape[0], -1)

            optimizer.zero_grad()

            params = hyper(conditioning_vector)
            params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

            logits = target.forward(X, params)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % log_interval == 0:
                avg = running_loss / log_interval
                print(f"HYPER: Epoch {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)}]  loss={avg:.4f}")
                running_loss = 0.0

        val_loss, val_acc = evaluate_hyper(hyper, target, test_loader, device)
        print(f"Epoch {epoch} completed — test loss: {val_loss:.4f}, test acc: {val_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'hyper_state_dict': hyper.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

    print("Training finished. Checkpoint saved to", save_path)
    return hyper

def train_vaes():
    dataset_names = ['fashion_mnist', 'kmnist', 'mnist']
    for name in dataset_names:
        data = get_dataset(name=name, preprocess=True, to_tensor=True, flatten=False)

        train = Dataset(data['train'])
        test = Dataset(data['test'])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        vae = VAE(w=image_width_height, h=image_width_height, ls_dim=vae_head_dim).to(device)
        vae = train_vae(vae, train_loader, test_loader)
        file_name = name + ".pth"
        torch.save({'hyper_state_dict': vae.state_dict()}, models_folder / file_name)

def main():
    #train_vaes()
    hyper = HyperNetwork(layer_sizes=target_layer_sizes, embed_dim=embed_dim, condition_dim=condition_dim, head_hidden=head_hidden, use_bias=use_bias).to(device)
    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)
    hyper.to(device)
    meta_train_hyper(hyper, target)


    

if __name__ == "__main__":
    main()
