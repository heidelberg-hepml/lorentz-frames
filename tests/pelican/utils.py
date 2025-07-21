import torch
from lloca.utils.utils import get_edge_index_from_ptr, get_batch_from_ptr


def generate_batch(G=8, N_range=[10, 20], C=16, batch=None, edge_index=None):
    if batch is None or edge_index is None:
        length = torch.randint(low=N_range[0], high=N_range[1], size=(G,))
        ptr = torch.zeros(G + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(length, dim=0)
        batch = get_batch_from_ptr(ptr)
        edge_index = get_edge_index_from_ptr(ptr, remove_self_loops=False)
    graph = torch.randn(G, C)
    nodes = torch.randn(batch.numel(), C)
    edges = torch.randn(edge_index.size(1), C)
    return batch, edge_index, graph, nodes, edges
