import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, w_0: nn.Module, rank: int, input_dim: int, output_dim: int, alpha: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, input_dim))
        self.B = nn.Parameter(torch.zeros(output_dim, rank))

        self.w_0 = w_0
        for param in self.w_0.parameters():
            param.requires_grad = False

        self.scaling = alpha / rank


    def forward(self, x: torch.Tensor):
        frozen_output = self.w_0(x)
        low_rank_output = x@self.A.T@self.B.T

        return frozen_output + self.scaling*low_rank_output