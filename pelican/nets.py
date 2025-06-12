from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import (
    Aggregator0to2,
    Aggregator1to2,
    Aggregator2to2,
    Aggregator2to1,
    Aggregator2to0,
    PELICANBlock,
)


class PELICAN(nn.Module):
    def __init__(
        self,
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_rank0=0,
        in_rank1=0,
        in_rank2=1,
        out_rank=0,
        out_channels=1,
        factorize=True,
        activation="gelu",
        aggr="mean",
        checkpoint_blocks=False,
    ):
        super().__init__()

        # embed inputs into edge features
        assert in_rank2 > 0
        self.in_aggregator2 = Aggregator2to2(
            in_channels=in_rank2,
            out_channels=hidden_channels,
            factorize=factorize,
            aggr=aggr,
        )
        if in_rank1 > 0:
            self.in_aggregator1 = Aggregator1to2(
                in_channels=in_rank1,
                out_channels=hidden_channels,
                factorize=factorize,
                aggr=aggr,
            )
        else:
            self.input_aggregator1 = None
        if in_rank0 > 0:
            self.in_aggregator0 = Aggregator0to2(
                in_channels=in_rank0,
                out_channels=hidden_channels,
                factorize=factorize,
                aggr=aggr,
            )
        else:
            self.input_aggregator0 = None

        # process edge features
        self._checkpoint_blocks = checkpoint_blocks
        self.blocks = nn.ModuleList(
            [
                PELICANBlock(
                    hidden_channels=hidden_channels,
                    increase_hidden_channels=increase_hidden_channels,
                    activation=activation,
                    factorize=factorize,
                    aggr=aggr,
                )
                for _ in range(num_blocks)
            ]
        )

        # extract outputs from edge features
        if out_rank == 0:
            self.out_aggregator = Aggregator2to0(
                in_channels=hidden_channels,
                out_channels=out_channels,
                factorize=factorize,
                aggr=aggr,
            )
        elif out_rank == 1:
            self.out_aggregator = Aggregator2to1(
                in_channels=hidden_channels,
                out_channels=out_channels,
                factorize=factorize,
                aggr=aggr,
            )
        elif out_rank == 2:
            self.out_aggregator = Aggregator2to2(
                in_channels=hidden_channels,
                out_channels=out_channels,
                factorize=factorize,
                aggr=aggr,
            )
        else:
            raise NotImplementedError

    def forward(self, in_rank2, edge_index, batch, in_rank1=None, in_rank0=None):
        # embed inputs into edge features
        x = self.in_aggregator2(in_rank2, edge_index, batch)
        if in_rank1 is not None:
            edges_fromrank1 = self.in_aggregator1(in_rank1, edge_index, batch)
            x += edges_fromrank1
        if in_rank0 is not None:
            edges_fromrank0 = self.in_aggregator0(in_rank0, edge_index, batch)
            x += edges_fromrank0

        # process edge features
        for block in self.blocks:
            if self._checkpoint_blocks:
                x = checkpoint(
                    block, x, edge_index=edge_index, batch=batch, use_reentrant=False
                )
            else:
                x = block(x, edge_index=edge_index, batch=batch)

        # extract outputs from edge features
        out = self.out_aggregator(x, edge_index=edge_index, batch=batch)
        return out
