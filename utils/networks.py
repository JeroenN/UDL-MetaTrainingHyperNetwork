import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        layer_sizes,
        condition_dim=10,
        head_hidden=256,
        use_bias=True,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.condition_dim = condition_dim
        self.use_bias = use_bias

        self.heads = nn.ModuleList()
        self.query = nn.Parameter(torch.randn(condition_dim))

        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            n_params = out_dim * in_dim + (out_dim if use_bias else 0)
            head = nn.Sequential(
                nn.Linear(condition_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, n_params),
            )
            # small init
            nn.init.normal_(head[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(head[-1].bias, 0.0)
            self.heads.append(head)

    def attention_pool(self, conditioning):
        scores = torch.matmul(conditioning, self.query)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(conditioning * weights.unsqueeze(-1), dim=0)
        return pooled

    def forward(self, conditioning):
        params = []
        for j in range(self.num_layers):
            z_cond = self.attention_pool(conditioning)

            head_input = z_cond.unsqueeze(0)

            flat = self.heads[j](head_input).squeeze(0)

            out_dim = self.layer_sizes[j + 1]
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
