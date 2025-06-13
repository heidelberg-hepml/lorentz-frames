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
    _, C = graph.shape
    E = edge_index.size(1)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(graph)

    ops = graph.new_empty(2, E, C)
    ops[0] = graph[edge_batch]
    ops[1] = graph[edge_batch] * diag_mask
    return ops.permute(1, 2, 0)


def aggregate_1to2(nodes, edge_index, batch, reduce="mean"):
    _, C = nodes.shape
    E = edge_index.size(1)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(nodes)

    nodes_agg = scatter(nodes, batch, dim=0, reduce=reduce)
    is_diag = is_diag.unsqueeze(-1)

    ops = nodes.new_empty(5, E, C)
    ops[0] = nodes[row] * diag_mask
    ops[1] = nodes[row]
    ops[2] = nodes[col]
    ops[3] = nodes_agg[edge_batch] * diag_mask
    ops[4] = nodes_agg[edge_batch]
    return ops.permute(1, 2, 0)


def aggregate_2to0(edges, edge_index, batch, reduce="mean"):
    _, C = edges.shape
    G = batch.max() + 1
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col

    graph_agg = scatter(edges, edge_batch, dim=0, reduce=reduce)
    diag_agg = scatter(edges[is_diag], edge_batch[is_diag], dim=0, reduce=reduce)

    ops = edges.new_empty(2, G, C)
    ops[0] = graph_agg
    ops[1] = diag_agg
    return ops.permute(1, 2, 0)


def aggregate_2to1(edges, edge_index, batch, reduce="mean"):
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    row_agg = scatter(edges, row, dim=0, dim_size=N, reduce=reduce)
    col_agg = scatter(edges, col, dim=0, dim_size=N, reduce=reduce)
    graph_agg = scatter(edges, edge_batch, dim=0, dim_size=N, reduce=reduce)
    diag_agg = scatter(diags, row, dim=0, dim_size=N, reduce=reduce)

    ops = edges.new_empty(5, N, C)
    ops[0] = edges[is_diag]
    ops[1] = row_agg
    ops[2] = col_agg
    ops[3] = graph_agg[batch]
    ops[4] = diag_agg[batch]
    return ops.permute(1, 2, 0)


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
