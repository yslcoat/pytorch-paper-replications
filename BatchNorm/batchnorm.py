import torch
import torch.nn as nn


class BatchNorm(nn.module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError