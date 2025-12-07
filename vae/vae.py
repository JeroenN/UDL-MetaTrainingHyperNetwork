from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 3
KERNEL_SIZE = 4
DECODER_STRIDE = 2

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, padding: int, strides: tuple[int, int] = (1, 1), last_channels: int | None = None, non_residual: bool = False):
        super().__init__()

        lows = min(strides)
        maxs = max(strides)
        downsample = maxs > 1

        if downsample and (lows != 1) and (maxs != 2):
            raise ValueError("Can only handle strides (1, 1), (1, 2), or (2, 1)")

        last_channels = last_channels or out_channels 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True, stride=strides[0])
        self.conv2 = nn.Conv2d(out_channels, last_channels, kernel_size=kernel_size, padding=padding, bias=True, stride=strides[1])

        self.use_shortcut = (in_channels != last_channels) or downsample
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, last_channels, padding=0, kernel_size=1, stride=maxs, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.non_residual = non_residual

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)

        if not self.non_residual:
            residual = self.shortcut(residual)
            x += residual

        x = F.leaky_relu(x)
        return x



class ResidualLeakyRelu(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()

        self.residual = nn.Linear(in_dim, out_dim, bias=False)

        dims = np.linspace(in_dim, out_dim, hidden + 2, endpoint=True, dtype=int)
        block = []

        for left, right in zip(dims[:-1], dims[1:], strict=True):
            block.append(nn.Linear(left, right))
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)


    def forward(self, x):
        y = self.block(x) + self.residual(x)
        return y

def eiv(element_or_vector: list | float, index: int):
    try:
        return element_or_vector[index]
    except TypeError:
        return element_or_vector


CDim = tuple[int, int, int, int]

class VAE(nn.Module):

    def __init__(self, w: int, h: int, ls_dim: int, in_channels: int = 3):
        super(VAE, self).__init__()
        self.ls_dim = ls_dim
        self.w = w
        self.h = h
        self.in_channels = in_channels

        # Encoder: Convolutional layers
        # For MNIST (28x28): 28 -> 14 -> 7 -> 7 (with appropriate architecture)
        # For RGB (128x128): 128 -> 64 -> 32 -> 16 -> 8
        
        if w == 28 and h == 28 and in_channels == 1:
            # MNIST-specific architecture
            self.encoder = nn.Sequential(
                ResidualBlock(in_channels=1, out_channels=32, kernel_size=3, padding=1, strides=(1, 2)),  # 28 -> 14
                ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1, strides=(1, 2)),  # 14 -> 7
                ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, strides=(1, 1)),  # 7 -> 7
            )
            
            self.latent_space_img_dims: CDim = [128, 7, 7]
            last_ch, last_w, last_h = self.latent_space_img_dims
            
            self.decoder = nn.Sequential(
                ResidualBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1, strides=(1, 1)),
                nn.PixelShuffle(2),  # 7 -> 14, channels: 64 -> 16
                
                ResidualBlock(in_channels=16, out_channels=64, kernel_size=3, padding=1, strides=(1, 1)),
                nn.PixelShuffle(2),  # 14 -> 28, channels: 64 -> 16
                
                ResidualBlock(in_channels=16, out_channels=8, last_channels=4, kernel_size=3, padding=1, strides=(1, 1), non_residual=True),
                nn.PixelShuffle(2),  # 28 -> 56, channels: 4 -> 1
                
                # Need to downsample back to 28x28
                nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            )
        else:
            # Original RGB architecture
            self.encoder = nn.Sequential(
                ResidualBlock(in_channels=in_channels, out_channels=30, kernel_size=3, padding=1, strides=(1, 2)),
                ResidualBlock(in_channels=30, out_channels=90, kernel_size=3, padding=1, strides=(1, 2)),
                ResidualBlock(in_channels=90, out_channels=150, kernel_size=3, padding=1, strides=(1, 2)),
                ResidualBlock(in_channels=150, out_channels=200, last_channels=250, kernel_size=3, padding=1, strides=(1, 2)),
            )

            self.latent_space_img_dims: CDim = [250, 8, 8]
            last_ch, last_w, last_h  = self.latent_space_img_dims

            self.decoder = nn.Sequential(
                ResidualBlock(in_channels=250, out_channels=200, kernel_size=3, padding=1, strides=(1, 1)),
                nn.PixelShuffle(2),

                ResidualBlock(in_channels=50, out_channels=160, kernel_size=3, padding=1, strides=(1, 1)),
                nn.PixelShuffle(2),

                ResidualBlock(in_channels=40, out_channels=160, kernel_size=3, padding=1, strides=(1, 1)),
                nn.PixelShuffle(2),

                ResidualBlock(in_channels=40, out_channels=20, last_channels=12, kernel_size=1, padding=0, strides=(1, 1), non_residual=True),
                nn.PixelShuffle(2),
            )

        self.fc_dim = last_ch * last_w * last_h

        # Latent space representations
        self.fc_mu = nn.Linear(self.fc_dim, ls_dim)
        self.fc_logvar = nn.Linear(self.fc_dim, ls_dim)
        
        self.decode_block = ResidualLeakyRelu(ls_dim, self.fc_dim, hidden=1)

    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.fc_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5  * logvar).to("cuda")
        eps = torch.randn_like(std, device="cuda")  # Sample from standard normal
        return mu + eps * std

    def decode(self, z):
        h = self.decode_block(z)
        last_ch, last_w, last_h = self.latent_space_img_dims
        h = h.view(-1, last_ch, last_w, last_h)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


    @classmethod
    def from_checkpoint(cls, checkpoint) -> 'VAE':
        model: 'VAE' = cls(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def from_checkpoint_path(cls, path: Path) -> 'VAE':
        checkpoint = torch.load(str(path), weights_only=True)
        return cls.from_checkpoint(checkpoint)
    


def calc_cnn_transform_size(x: int, padding: int, kernel: int, stride: int, dilation: int = 1):
    return int((x + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1

def calc_padding_decoder(
        x: int,
        x_wanted: int,
        kernel: int,
        stride: int,
        out_padding: int = 0,
        kernel_step: int | None = None
    ) -> tuple[int, int]:
    # x_wanted = (x - 1) * stride - 2 * padding + kernel + out_padding
    # x_wanted - (x - 1) * stride - kernel - out_padding =  - 2 * padding
    # padding = (x - x_wanted + kernel)
    def calc(kernel: int):
        return ((x - 1) * stride - x_wanted + kernel + out_padding) / 2
    
    actual_kernel = kernel
    padding = calc(actual_kernel)
    if round(padding) != padding:
        if kernel_step is None:
            raise ValueError("Padding is not an integer and kernel step is not allowed")
        else:
            assert (kernel_step in {-1, 1}), "kernel step needs to be -1 or 1"
            actual_kernel += kernel_step
            padding = calc(actual_kernel)
            assert round(padding) == padding, "WHAT? WHY?"

    return int(padding), actual_kernel