import torch
from torch_scatter import scatter
from functools import lru_cache


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

# row and column sum
row_avg = scatter(x, edge_index[0], dim=0, reduce=reduce)
col_avg = scatter(x, edge_index[1], dim=0, reduce=reduce)
print(row_avg.shape, col_avg.shape)
row_avg_expanded = row_avg[edge_index[0]]
col_avg_expanded = col_avg[edge_index[1]]
print(row_avg_expanded.shape, col_avg_expanded.shape)

# graph sum
edge_batch = batch[edge_index[0]]
graph_avg = scatter(x, edge_batch, dim=0, reduce=reduce)
graph_avg_expanded = graph_avg[edge_batch]
print(graph_avg.shape, graph_avg_expanded.shape)

# diagonal sum
is_diag = edge_index[0] == edge_index[1]
diag_idx = batch[edge_index[0][is_diag]]
diag_avg = scatter(x[is_diag], diag_idx, dim=0, reduce=reduce)
diag_avg_expanded = torch.zeros_like(x)
diag_avg_expanded[is_diag] = diag_avg[diag_idx]
print(diag_avg.shape, diag_avg_expanded.shape)


@lru_cache()
def get_indexing(edge_index, batch):
    edge_batch = batch[edge_index[0]]
    is_diag = edge_index[0] == edge_index[1]
    diag_idx = batch[edge_index[0][is_diag]]

    N = int(edge_index.max()) + 1
    eid = torch.arange(edge_index.size(1))
    lin_id = edge_index[0] * N + edge_index[1]
    inv = torch.full((N * N,), -1)
    inv.scatter_(0, lin_id, eid)
    perm_transpose = inv[edge_index[1] * N + edge_index[0]]
    return edge_batch, is_diag, diag_idx, perm_transpose


def aggregate(x, indexing):
    # can share this across layers (if applicable)
    edge_batch, is_diag, diag_idx, _ = indexing
    diags = x[is_diag]

    row_agg = scatter(x, edge_index[0], dim=0, reduce=reduce)
    col_agg = scatter(x, edge_index[1], dim=0, reduce=reduce)
    graph_agg = scatter(x, edge_batch, dim=0, reduce=reduce)
    diag_agg = scatter(diags, diag_idx, dim=0, reduce=reduce)
    return diags, row_agg, col_agg, graph_agg, diag_agg


# pelican layer
def eops_2to2(x, edge_index, batch, reduce="sum"):
    indexing = get_indexing(edge_index, batch)
    edge_batch, is_diag, diag_idx, perm_transpose = indexing

    zeros = torch.zeros_like(x)
    diags, row_agg, col_agg, graph_agg, diag_agg = aggregate(x, indexing)
    is_diag = is_diag.unsqueeze(-1)

    # permutation-equivariant maps
    ops = [None] * 15
    ops[0] = x
    ops[1] = x[perm_transpose]
    ops[2] = torch.where(is_diag, x, zeros)
    ops[3] = diags[edge_index[0]]
    ops[4] = diags[edge_index[1]]
    ops[5] = torch.where(is_diag, col_agg[edge_index[0]], zeros)
    ops[6] = col_agg[edge_index[0]]
    ops[7] = col_agg[edge_index[1]]
    ops[8] = torch.where(is_diag, row_agg[edge_index[1]], zeros)
    ops[9] = row_agg[edge_index[0]]
    ops[10] = row_agg[edge_index[1]]
    ops[11] = diag_agg[edge_batch]
    ops[12] = torch.where(is_diag, diag_agg[edge_batch], zeros)
    ops[13] = graph_agg[edge_batch]
    ops[14] = torch.where(is_diag, graph_agg[edge_batch], zeros)
    return torch.stack(ops, dim=-1)


out = eops_2to2(x, edge_index, batch)
print(x.shape, out.shape)


def eops_2to0(x, edge_index, batch, reduce="sum"):
    indexing = get_indexing(edge_index, batch)
    edge_batch, is_diag, diag_idx, perm_transpose = indexing

    zeros = torch.zeros_like(x)
    diags, row_agg, col_agg, graph_agg, diag_agg = aggregate(x, indexing)
    is_diag = is_diag.unsqueeze(-1)

    # permutation-equivariant maps
    ops = [None] * 2
    ops[0] = graph_agg
    ops[1] = diag_agg
    return torch.stack(ops, dim=-1)


out = eops_2to0(x, edge_index, batch)
print(x.shape, out.shape)


def eops_2to1(x, edge_index, batch, reduce="sum"):
    indexing = get_indexing(edge_index, batch)
    edge_batch, is_diag, diag_idx, perm_transpose = indexing

    zeros = torch.zeros_like(x)
    diags, row_agg, col_agg, graph_agg, diag_agg = aggregate(x, indexing)
    is_diag = is_diag.unsqueeze(-1)

    # permutation-equivariant maps
    ops = [None] * 5
    ops[0] = diags
    ops[1] = row_agg
    ops[2] = col_agg
    ops[3] = graph_agg[batch]
    ops[4] = diag_agg[batch]
    return torch.stack(ops, dim=-1)


out = eops_2to1(x, edge_index, batch)
print(x.shape, out.shape)

x = torch.randn(batch.numel(), C)


def eops_1to1(x, edge_index, batch, reduce="sum"):
    # aggregate nodes
    node_agg = scatter(x, batch, dim=0, reduce=reduce)

    ops = [None] * 2
    ops[0] = x
    ops[1] = node_agg[batch]
    return torch.stack(ops, dim=-1)


out = eops_1to1(x, edge_index, batch)
print(x.shape, out.shape)


def eops_1to2(x, edge_index, batch, reduce="sum"):
    indexing = get_indexing(edge_index, batch)
    edge_batch, is_diag, diag_idx, perm_transpose = indexing
    is_diag = is_diag.unsqueeze(-1)
    zeros = torch.zeros(edge_index.size(1), x.shape[-1])

    # aggregate nodes
    node_agg = scatter(x, batch, dim=0, reduce=reduce)

    ops = [None] * 5
    ops[0] = torch.where(is_diag, x[edge_index[0]], zeros)
    ops[1] = x[edge_index[0]]
    ops[2] = x[edge_index[1]]
    ops[3] = torch.where(is_diag, node_agg[edge_batch], zeros)
    ops[4] = node_agg[edge_batch]

    return torch.stack(ops, dim=-1)


out = eops_1to2(x, edge_index, batch)
print(x.shape, out.shape)

"""
Notes
- PELICAN mindset: First aggregate, then expand in different ways
- Can allow for seperate graph, node and edge representations and mixing between them
- Could also do 0to2 and 0to1 maps, but these are not relevant in particle physics
"""
