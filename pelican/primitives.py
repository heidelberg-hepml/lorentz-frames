import torch
from torch_scatter import scatter
from functools import lru_cache


def bell_number(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 5
    elif n == 4:
        return 15
    else:
        raise NotImplementedError(
            f"Asking for bell_number(n={n}), but bell_number(n>4) not implemented yet."
        )


def aggregate_0to2(graph, edge_index, batch, reduce="mean"):
    edge_batch = batch[edge_index[0]]
    is_diag = edge_index[0] == edge_index[1]
    is_diag = is_diag.unsqueeze(-1)

    zeros = torch.zeros(
        edge_index.size(1), graph.shape[-1], device=graph.device, dtype=graph.dtype
    )

    ops = [None] * 2
    ops[0] = graph[edge_batch]
    ops[1] = torch.where(is_diag, graph[edge_batch], zeros)
    return torch.stack(ops, dim=-1)


def aggregate_1to2(nodes, edge_index, batch, reduce="mean"):
    edge_batch = batch[edge_index[0]]
    is_diag = edge_index[0] == edge_index[1]

    nodes_agg = scatter(nodes, batch, dim=0, reduce=reduce)
    zeros = torch.zeros(
        edge_index.size(1), nodes.shape[-1], device=nodes.device, dtype=nodes.dtype
    )
    is_diag = is_diag.unsqueeze(-1)

    ops = [None] * 5
    ops[0] = torch.where(is_diag, nodes[edge_index[0]], zeros)
    ops[1] = nodes[edge_index[0]]
    ops[2] = nodes[edge_index[1]]
    ops[3] = torch.where(is_diag, nodes_agg[edge_batch], zeros)
    ops[4] = nodes_agg[edge_batch]
    return torch.stack(ops, dim=-1)


def aggregate_2to0(edges, edge_index, batch, reduce="mean"):
    is_diag = edge_index[0] == edge_index[1]
    edge_batch = batch[edge_index[0]]

    graph_agg = scatter(edges, edge_batch, dim=0, reduce=reduce)
    diag_agg = scatter(edges[is_diag], edge_batch[is_diag], dim=0, reduce=reduce)

    ops = [None] * 2
    ops[0] = graph_agg
    ops[1] = diag_agg
    return torch.stack(ops, dim=-1)


def aggregate_2to1(edges, edge_index, batch, reduce="mean"):
    is_diag = edge_index[0] == edge_index[1]
    edge_batch = batch[edge_index[0]]
    diag_idx = batch[edge_index[0][is_diag]]

    diags = edges[is_diag]
    row_agg = scatter(edges, edge_index[0], dim=0, reduce=reduce)
    col_agg = scatter(edges, edge_index[1], dim=0, reduce=reduce)
    graph_agg = scatter(edges, edge_batch, dim=0, reduce=reduce)
    diag_agg = scatter(diags, diag_idx, dim=0, reduce=reduce)

    ops = [None] * 5
    ops[0] = diags
    ops[1] = row_agg
    ops[2] = col_agg
    ops[3] = graph_agg[batch]
    ops[4] = diag_agg[batch]
    return torch.stack(ops, dim=-1)


def get_transpose(row, col):
    key = (row << 32) | col
    rev_key = (col << 32) | row
    key_sorted, perm = key.sort()
    idx = torch.searchsorted(key_sorted, rev_key)
    return perm[idx]


def aggregate_2to2(edges, edge_index, batch, reduce="mean"):
    E, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    perm_T = get_transpose(row, col)
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    row_agg = scatter(edges, row, dim=0, dim_size=N, reduce=reduce)
    col_agg = scatter(edges, col, dim=0, dim_size=N, reduce=reduce)
    graph_agg = scatter(edges, edge_batch, dim=0, dim_size=N, reduce=reduce)
    diag_agg = scatter(diags, row, dim=0, dim_size=N, reduce=reduce)

    ops = edges.new_empty(15, E, C)
    ops[0] = edges
    ops[1] = edges[perm_T]
    ops[2] = diags
    ops[3] = diag_agg[row]
    ops[4] = diag_agg[col]
    ops[5] = col_agg[row] * diag_mask
    ops[6] = row_agg[col] * diag_mask
    ops[7] = col_agg[row]
    ops[8] = col_agg[col]
    ops[9] = row_agg[row]
    ops[10] = row_agg[col]
    ops[11] = diag_agg[edge_batch]
    ops[12] = diag_agg[edge_batch] * diag_mask
    ops[13] = graph_agg[edge_batch]
    ops[14] = graph_agg[edge_batch] * diag_mask
    return ops.permute(1, 2, 0)
