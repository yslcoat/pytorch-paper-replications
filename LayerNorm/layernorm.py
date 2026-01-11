import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.n_features))
        self.beta = nn.Parameter(torch.zeros(self.n_features))

    def forward(self, x: torch.Tensor):
        activation_mean = torch.mean(x, dim=-1, keepdim=True)
        activation_var = torch.var(x, dim=-1, unbiased=False, keepdim=True)

        x_hat = (x - activation_mean) / torch.sqrt(activation_var + self.eps)
        scaled_and_shifted = self.gamma * x_hat + self.beta

        return scaled_and_shifted