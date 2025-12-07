import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import struct
from vae import VAE
from torch.utils.data import Dataset
from tqdm import tqdm

DATA_DIR = Path("/home4/s6019595/UDL-MetaTrainingHyperNetwork/data/mnist")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
w, h = 28, 28
I_IN_CHANNELS = 1
IN_SIZE = (I_IN_CHANNELS, w, h)

def load_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def load_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:  # <<< fixed here
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
   
def ssim_loss(img1, img2, window_size=11, reduction='mean'):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=0)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=0)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=0) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean() if reduction == 'mean' else 1 - ssim_map 

def loss_function(recon_x, x, mu, logvar):

    MSE = F.mse_loss(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ssim = MSE*ssim_loss(recon_x,x)
    
    return MSE + ssim + KLD

def loss_mse(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def loss_mse_only(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    return MSE

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path: Path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_architecture": model.__class__.__name__,
        "model_args": {"w": model.w, "h": model.h, "ls_dim": model.ls_dim, "in_channels": model.in_channels},  # Save architecture parameters
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: Path, device):
    checkpoint = torch.load(str(path), weights_only=True)
    vae = VAE.from_checkpoint(checkpoint).to(device)
    return checkpoint["epoch"], vae
    
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # Normalize to [0, 1] and convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        # Add channel dimension: (28, 28) -> (1, 28, 28)
        image = image.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        # Return same image for both input and output (autoencoder task)
        return image, image

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for image_x, image_y in train_loader:
        image_x = image_x.to(device)
        image_y = image_y.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(image_x)
        loss = criterion(recon_batch, image_y, mu, logvar)
        
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), 20.0)
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image_x, image_y in val_loader:
            image_x = image_x.to(device)
            image_y = image_y.to(device)
            
            recon_batch, mu, logvar = model(image_x)
            loss = criterion(recon_batch, image_y, mu, logvar)
            val_loss += loss.item()
    
    return val_loss / len(val_loader.dataset)

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer,
    criterion,
    device,
    checkpoint_folder,
):
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Model: latent_dim={model.ls_dim}, image_size=({model.w}x{model.h})")
    
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          checkpoint_folder / "best_checkpoint.pth")
    
    # Save final checkpoint
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    parser.add_argument('--ls', type=int, default=32, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--experiment', type=str, default='mnist_vae', help='Experiment name')
    
    args = parser.parse_args()
    
    # Load MNIST data
    print("Loading MNIST data...")
    X_train = load_images(DATA_DIR / "train-images.idx3-ubyte")
    y_train = load_labels(DATA_DIR / "train-labels.idx1-ubyte")
    X_test = load_images(DATA_DIR / "t10k-images.idx3-ubyte")
    y_test = load_labels(DATA_DIR / "t10k-labels.idx1-ubyte")
    
    print(f"Train images: {X_train.shape}, Train labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}, Test labels: {y_test.shape}")
    
    # Create datasets
    train_dataset = MNISTDataset(X_train, y_train)
    val_dataset = MNISTDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize or load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        start_epoch, model = load_checkpoint(Path(args.checkpoint), device)
        checkpoint_folder = Path(args.checkpoint).parent
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Creating new model...")
        model = VAE(w, h, args.ls, in_channels=I_IN_CHANNELS).to(device)
        checkpoint_folder = Path("checkpoints") / args.experiment
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_function  # or loss_mse or loss_mse_only
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_folder=checkpoint_folder
    )

if __name__ == "__main__":
    main()
