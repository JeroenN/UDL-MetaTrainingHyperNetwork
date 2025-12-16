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
from dataset_loading import get_dataset, Dataset

device = torch.device("cuda")
batch_size = 256
epochs_hyper = 10
epochs_vae = 3
lr_hyper = 1e-4
lr_vae = 1e-4
log_interval = 10
save_path = "hypernet_checkpoint.pth"

target_layer_sizes = [784, 400, 200, 10]

# Hypernetwork internal sizes
embed_dim = 32           
head_hidden = 256        
use_bias = True


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
    def __init__(self, layer_sizes, embed_dim=32, head_hidden=256, use_bias=True):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.embed_dim = embed_dim
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
                nn.Linear(embed_dim, head_hidden),
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
            z = self.layer_embeddings[j] 

            z_cond = torch.cat([z, conditioning], dim=0)
            head_input = z_cond.unsqueeze(0)  # (1, embed_dim + cond_dim)
            head_input = z.unsqueeze(0)

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
    running_loss = 0
    list_mu, list_logvar = []
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(device)
        reconstruct, mu, logvar = vae(x)
        list_mu.append(mu)
        list_logvar.append(logvar)
        running_loss += loss_mse(reconstruct, x, mu, logvar)

        img, ax = plt.subplots(2,2)
        ax[0,0].imshow(x[0].squeeze().to('cpu'))
        ax[0,1].imshow(reconstruct[0].squeeze().detach().cpu())
        file_path = Path(__file__).parent / "visualization" / f"evaluation_batch_idx_{batch_idx}.png"
        img.savefig(file_path)
    
    print(f"VAE TEST LOSS: {running_loss / (len(test_loader))}")
    return list_mu, list_logvar


def get_guassian_from_vae(vae, data_loader):
    running_loss = 0
    list_mu, list_logvar = []
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.to(device)
        reconstruct, mu, logvar = vae(x)
        list_mu.append(mu)
        list_logvar.append(logvar)

        img, ax = plt.subplots(2,2)
        ax[0,0].imshow(x[0].squeeze().to('cpu'))
        ax[0,1].imshow(reconstruct[0].squeeze().detach().cpu())
        file_path = Path(__file__).parent / "visualization" / f"results_batch_idx_{batch_idx}.png"
        img.savefig(file_path)
    
    return list_mu, list_logvar



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
    
    list_mu, list_logvar =get_results_vae(vae, test_loader)
    return list_mu, list_logvar


def evaluate_hyper(model_hyper, target_net, dataloader, device):
    model_hyper.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            params = model_hyper()
            params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in params]

            logits = target_net.forward(X, params)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.shape[0]

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += X.shape[0]

    return total_loss / total, correct / total

def train_hyper(hyper, target, train_loader, test_loader):
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_hyper)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs_hyper + 1):
        hyper.train()
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader, start=1):
            X = X.to(device)
            y = y.to(device)
            X = X.view(X.shape[0], -1)

            optimizer.zero_grad()

            params = hyper()
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

    hyper.eval()
    sample_params = hyper()
    sample_params = [(W.to(device), b.to(device) if b is not None else None) for (W, b) in sample_params]
    X_sample, y_sample = next(iter(test_loader))
    X_sample = X_sample[:8].to(device).view(8, -1)
    with torch.no_grad():
        logits = target.forward(X_sample, sample_params)
        preds = logits.argmax(dim=-1)
    print("Sample predictions:", preds.cpu().tolist())
    print("Ground truth      :", y_sample[:8].tolist())

def main():
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #])
    #train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    #test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    data = get_dataset(name="mnist", preprocess=True, to_tensor=True, flatten=False)

    train = Dataset(data['train'])
    test = Dataset(data['test'])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)




    
    vae = VAE(w=28, h=28, ls_dim=5).to(device)
    mu, logvar = train_vae(vae, train_loader, test_loader)

    hyper = HyperNetwork(layer_sizes=target_layer_sizes, embed_dim=embed_dim, head_hidden=head_hidden, use_bias=use_bias).to(device)
    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)
    train_hyper(hyper, target, train_loader, test_loader)

    

if __name__ == "__main__":
    main()
