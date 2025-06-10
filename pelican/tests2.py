import torch
from torch_scatter import scatter
from functools import lru_cache
from layers import *
from primitives import *
from nets import PELICAN


def get_batch_from_ptr(ptr):
    """Reconstruct batch indices (batch) from pointer (ptr).

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.

    Returns
    -------
    torch.Tensor
        A tensor where each element indicates the batch index for each item.
    """
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


def get_edge_index_from_ptr(ptr, remove_self_loops=True):
    """Construct edge index of fully connected graph from pointer (ptr).

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.
    remove_self_loops : bool, optional
        Whether to remove self-loops from the edge index, by default True.

    Returns
    -------
    torch.Tensor
        A tensor of shape (2, E) where E is the number of edges, representing the edge index.
    """
    row = torch.arange(ptr.max(), device=ptr.device)
    diff = ptr[1:] - ptr[:-1]
    repeats = (diff).repeat_interleave(diff)
    row = row.repeat_interleave(repeats)

    repeater = torch.stack(
        (-diff + 1, torch.ones_like(diff, device=ptr.device))
    ).T.reshape(-1)
    extras = repeater.repeat_interleave(repeater.abs())
    integ = torch.ones(row.shape[0], dtype=torch.long, device=ptr.device)
    mask = (row[1:] - row[:-1]).to(torch.bool)
    integ[0] = 0
    integ[1:][mask] = extras[:-1]
    col = torch.cumsum(integ, 0)

    edge_index = torch.stack((row, col))

    if remove_self_loops:
        row, col = edge_index
        edge_index = edge_index[:, row != col]

    return edge_index


# generate data
B = 5
C = 1
reduce = "mean"  # mul, mean, min, max
diff = torch.randint(low=2, high=15, size=(B,))
ptr = torch.cumsum(diff, dim=0)
ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
n_nodes = ptr[-1]
batch = get_batch_from_ptr(ptr)

# get edge_index
# note: sum_diag_part only works if self-loops are included
edge_index = get_edge_index_from_ptr(ptr, remove_self_loops=False)
n_edges = edge_index.shape[1]

x = torch.randn(n_edges, C)
print(n_nodes, n_edges, x.shape)

layer = Aggregator2to2(in_channels=C, out_channels=1, factorize=True)
out = layer(x, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)


layer = Aggregator2to1(in_channels=C, out_channels=1, factorize=True)
out = layer(x, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)


layer = Aggregator2to0(in_channels=C, out_channels=1, factorize=True)
out = layer(x, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)

nodes = torch.randn(n_nodes, C)
layer = Aggregator1to2(in_channels=C, out_channels=1, factorize=True)
out = layer(nodes, edge_index=edge_index, batch=batch)
print(nodes.shape, out.shape)

graph = torch.randn(B, C)
layer = Aggregator0to2(in_channels=C, out_channels=1, factorize=True)
out = layer(graph, edge_index=edge_index, batch=batch)
print(graph.shape, out.shape)

layer = PELICANBlock(channels_1=C, channels_2=1)
out = layer(x, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)

layer = PELICAN(num_blocks=2, channels_1=3, channels_2=8)
out = layer(x, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)

layer = PELICAN(
    num_blocks=2,
    channels_1=3,
    channels_2=8,
    in_rank0=1,
    in_rank1=1,
    in_rank2=1,
)
out = layer(x, in_rank1=nodes, in_rank0=graph, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)

layer = PELICAN(
    num_blocks=2,
    channels_1=3,
    channels_2=8,
    in_rank0=1,
    in_rank1=1,
    in_rank2=1,
    out_rank=1,
)
out = layer(x, in_rank1=nodes, in_rank0=graph, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)

layer = PELICAN(
    num_blocks=2,
    channels_1=3,
    channels_2=8,
    in_rank0=1,
    in_rank1=1,
    in_rank2=1,
    out_rank=2,
)
out = layer(x, in_rank1=nodes, in_rank0=graph, edge_index=edge_index, batch=batch)
print(x.shape, out.shape)
