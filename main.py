# hypernet_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os

device = torch.device("cuda")
batch_size = 256
epochs = 10
lr = 1e-3
log_interval = 100
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

    def forward(self, conditioning=None):
        params = []
        for j in range(self.num_layers):
            z = self.layer_embeddings[j] 
            if conditioning is not None:
                z_cond = torch.cat([z, conditioning], dim=0)
                head_input = z_cond.unsqueeze(0)  # (1, embed_dim + cond_dim)
                head_input = z.unsqueeze(0)
            else:
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

def evaluate(model_hyper, target_net, dataloader, device):
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

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    hyper = HyperNetwork(layer_sizes=target_layer_sizes, embed_dim=embed_dim, head_hidden=head_hidden, use_bias=use_bias).to(device)
    target = TargetNet(layer_sizes=target_layer_sizes, activation=F.relu)

    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
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
                print(f"Epoch {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)}]  loss={avg:.4f}")
                running_loss = 0.0

        val_loss, val_acc = evaluate(hyper, target, test_loader, device)
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

if __name__ == "__main__":
    main()
