# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import torch
from torch import nn

from experiments.baselines.cgenn.utils import unsqueeze_like

EPS = 1e-6


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        # 1-d (channels,) rather than the official (1, channels): a norm GAIN must fall
        # under the optimizer's ndim<=1 weight-decay exemption like every other norm
        # gain in the model family (the official 2-d shape silently got weight-decayed,
        # an undocumented regularization asymmetry hitting only the CGENN hybrids).
        # Broadcasting below is unchanged: we unsqueeze to (1, channels) before use.
        self.a = nn.Parameter(torch.ones(channels))

    def forward(self, input):
        norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a.unsqueeze(0), norm, dim=2)
        return a * input / norm
