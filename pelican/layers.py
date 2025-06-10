import torch
from torch import nn

from primitives import (
    bell_number,
    aggregate_0to2,
    aggregate_1to2,
    aggregate_2to0,
    aggregate_2to1,
    aggregate_2to2,
)

ACTIVATION = {
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
}


class GeneralAggregator(nn.Module):
    def __init__(
        self, in_rank, out_rank, in_channels, out_channels, factorize=True, aggr="mean"
    ):
        super().__init__()
        num_maps = bell_number(in_rank + out_rank)
        self.aggr = aggr
        self.aggregator = None

        if factorize:
            self.coeffs00 = nn.Parameter(torch.empty(in_channels, num_maps))
            self.coeffs01 = nn.Parameter(torch.empty(out_channels, num_maps))
            self.coeffs10 = nn.Parameter(torch.empty(in_channels, out_channels))
            self.coeffs11 = nn.Parameter(torch.empty(in_channels, out_channels))
            torch.nn.init.normal_(self.coeffs00, std=(1.0 / num_maps) ** 0.5)
            torch.nn.init.normal_(self.coeffs01, std=(1.0 / num_maps) ** 0.5)
            torch.nn.init.normal_(self.coeffs10, std=(2.0 / num_maps) ** 0.5)
            torch.nn.init.normal_(self.coeffs11, std=(2.0 / num_maps) ** 0.5)

            self.coeffs = self.coeffs00.unsqueeze(1) * self.coeffs10.unsqueeze(
                2
            ) + self.coeffs01.unsqueeze(0) * self.coeffs11.unsqueeze(2)
        else:
            self.coeffs = nn.Parameter(torch.empty(in_channels, out_channels, num_maps))
            torch.nn.init.normal_(
                self.coeffs, std=(4.0 / (in_channels * num_maps)) ** 0.5
            )

    def forward(self, x, edge_index, batch):
        x = self.aggregator(x, edge_index=edge_index, batch=batch, reduce=self.aggr)
        out = torch.einsum("ijk,bik->bj", self.coeffs, x)
        return out


class Aggregator2to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to2


class Aggregator2to1(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 1, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to1


class Aggregator2to0(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 0, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to0


class Aggregator1to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(1, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_1to2


class Aggregator0to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(0, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_0to2


class PELICANBlock(nn.Module):
    def __init__(
        self,
        channels_1,
        channels_2,
        activation="gelu",
        factorize=True,
        aggr="mean",
    ):
        super().__init__()
        linear_in = nn.Linear(channels_1, channels_2)
        norm = nn.LayerNorm(normalized_shape=channels_2)
        self.activation = ACTIVATION[activation]
        self.mlp = nn.ModuleList([linear_in, self.activation, norm])

        self.aggregator = Aggregator2to2(
            in_channels=channels_2,
            out_channels=channels_1,
            factorize=factorize,
            aggr=aggr,
        )

    def forward(self, x, edge_index, batch):
        for layer in self.mlp:
            x = layer(x)

        x = self.aggregator(x, edge_index=edge_index, batch=batch)
        x = self.activation(x)
        return x
