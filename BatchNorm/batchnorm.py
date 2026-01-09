import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(self.n_features))
        self.beta = nn.Parameter(torch.zeros(self.n_features))

        # Used to calculate EMA instead of calculating mean and variance of the entire training set after training. 
        self.register_buffer('running_mean', torch.zeros(n_features))
        self.register_buffer('running_var', torch.ones(n_features))

    def forward(self, x: torch.Tensor):
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        scaled_and_shifted = self.gamma * x_hat + self.beta

        return scaled_and_shifted