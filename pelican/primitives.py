import torch


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


def aggregate_0to2(graph, edge_index, batch, reduce="mean", **kwargs):
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


def aggregate_1to2(nodes, edge_index, batch, reduce="mean", **kwargs):
    _, C = nodes.shape
    E = edge_index.size(1)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(nodes)

    nodes_agg = custom_scatter(nodes, batch, dim_size=E, C=C, reduce=reduce)
    is_diag = is_diag.unsqueeze(-1)

    ops = nodes.new_empty(5, E, C)
    ops[0] = nodes[row] * diag_mask
    ops[1] = nodes[row]
    ops[2] = nodes[col]
    ops[3] = nodes_agg[edge_batch] * diag_mask
    ops[4] = nodes_agg[edge_batch]
    return ops.permute(1, 2, 0)


def aggregate_2to0(edges, edge_index, batch, reduce="mean", G=None, **kwargs):
    _, C = edges.shape
    if G is None:
        # host synchronization causes slowdown; maybe there is a better way?
        G = batch[-1].item() + 1
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    graph_agg = custom_scatter(edges, edge_batch, dim_size=G, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, edge_batch, dim_size=G, C=C, reduce=reduce)

    ops = edges.new_empty(2, G, C)
    ops[0] = graph_agg
    ops[1] = diag_agg
    return ops.permute(1, 2, 0)


def aggregate_2to1(edges, edge_index, batch, reduce="mean", **kwargs):
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, row, dim_size=N, C=C, reduce=reduce)

    ops = edges.new_empty(5, N, C)
    ops[0] = edges[is_diag]
    ops[1] = row_agg
    ops[2] = col_agg
    ops[3] = graph_agg[batch]
    ops[4] = diag_agg[batch]
    return ops.permute(1, 2, 0)


def aggregate_2to2(edges, edge_index, batch, reduce="mean", perm_T=None, **kwargs):
    E, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    if perm_T is None:
        perm_T = get_transpose(edge_index)
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask

    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, row, dim_size=N, C=C, reduce=reduce)

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


def get_transpose(edge_index):
    row, col = edge_index
    key = (row << 32) | col
    rev_key = (col << 32) | row
    key_sorted, perm = key.sort()
    idx = torch.searchsorted(key_sorted, rev_key)
    return perm[idx]


def custom_scatter(src, index, dim_size, C, reduce):
    out = src.new_zeros(dim_size, C)
    out.scatter_reduce_(
        0, index.unsqueeze(-1).expand(-1, C), src, reduce=reduce, include_self=False
    )
    return out
