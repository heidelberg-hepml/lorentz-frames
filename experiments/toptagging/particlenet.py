import torch
from torch import nn


class ParticleNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, edge_features, lframes):
        raise NotImplementedError
