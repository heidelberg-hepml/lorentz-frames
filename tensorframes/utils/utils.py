import torch
from torch_geometric.utils import dense_to_sparse
from xformers.ops.fmha import BlockDiagonalMask


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


def stable_arctanh(x, eps=1e-10):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.reshape(
        -1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :]
    )


def repeat_in_list(
    x: any, repeats: int, repeat_even_if_list: bool = False, repeat_if_none: bool = True
) -> list:
    """Repeats the given element `x` in a list `repeats` number of times.

    Args:
        x (any): The element to be repeated.
        repeats (int): The number of times to repeat the element.
        repeat_even_if_list (bool, optional): If True, repeats the element even if it is already a list. Defaults to False.
        repeat_if_none (bool, optional): If True, repeats the element even if it is None. Defaults to True.

    Returns:
        list: A list containing the repeated element.
    """
    if x is None and not repeat_if_none:
        return None
    if isinstance(x, list):
        if repeat_even_if_list:
            return [x for _ in range(repeats)]
        else:
            return x
    else:
        return [x for _ in range(repeats)]


def consistent_length_check(list_of_iterables: list) -> int:
    """Checks that all iterables in the list have the same length.

    Args:
        list_of_iterables (list): A list of iterables to be checked.

    Returns:
        int: The length of the iterables.

    Raises:
        AssertionError: If any of the iterables have a different length than the first iterable.
    """
    if len(list_of_iterables) == 0:
        return
    length = len(list_of_iterables[0])
    for i, iterable in enumerate(list_of_iterables):
        if iterable is None:
            continue
        assert (
            len(iterable) == length
        ), f"lengths must be the same but {i} has length {len(iterable)} and 0 has length {length}"
    return length


def batch_to_ptr(batch: torch.Tensor):
    """Converts torch tensor batch to slicing.

    Args:
        batch (torch.Tensor): The input tensor batch.

    Returns:
        torch.Tensor: The converted slicing tensor.

    Raises:
        AssertionError: If the input batch is not sorted.
    """
    # check that batch is sorted:
    assert torch.all(batch[:-1] <= batch[1:]), "batch must be sorted"

    diff_mask = batch - torch.roll(batch, 1) != 0
    diff_mask[0] = True  # first element is always different
    ptr = torch.zeros(batch.max() + 2, dtype=torch.long, device=batch.device)
    ptr[:-1] = torch.arange(len(batch), device=batch.device)[diff_mask]
    ptr[-1] = len(batch)
    return ptr


def get_batch_from_ptr(ptr):
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


def get_ptr_from_batch(batch):
    return torch.cat(
        [
            torch.tensor([0], device=batch.device),
            torch.where(batch[1:] - batch[:-1] != 0)[0] + 1,
            torch.tensor([batch.shape[0]], device=batch.device),
        ],
        0,
    )


def get_edge_index_from_ptr(ptr, remove_self_loops=True):
    diffs = torch.diff(ptr)
    edge_index = torch.cat(
        [
            dense_to_sparse(torch.ones(d, d, device=diffs.device))[0] + diffs[:i].sum()
            for i, d in enumerate(diffs)
        ],
        dim=-1,
    )
    if remove_self_loops:
        row, col = edge_index
        edge_index = edge_index[:, row != col]
    return edge_index


def build_edge_index_fully_connected(features_ref, remove_self_loops=True):
    batch_size, seq_len, _ = features_ref.shape
    device = features_ref.device

    nodes = torch.arange(seq_len, device=device)
    row, col = torch.meshgrid(nodes, nodes, indexing="ij")

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]
    edge_index_single = torch.stack([row.flatten(), col.flatten()], dim=0)

    edge_index_global = []
    for i in range(batch_size):
        offset = i * seq_len
        edge_index_global.append(edge_index_single + offset)
    edge_index_global = torch.cat(edge_index_global, dim=1)

    batch = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    return edge_index_global, batch


def get_xformers_attention_mask(batch, materialize=False, dtype=torch.float32):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(
            batch.device, dtype=dtype
        )
    return mask
