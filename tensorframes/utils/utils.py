import torch

from .lorentz import lorentz_squarednorm


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


def build_edge_index_fully_connected(features_ref, remove_self_loops=True):
    B, N, _ = features_ref.shape
    device = features_ref.device

    nodes = torch.arange(N, device=device)
    row = nodes.repeat_interleave(N)
    col = nodes.repeat(N)

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]

    edge_base = torch.stack([row, col], dim=0)

    offsets = torch.arange(B, device=device, dtype=torch.long) * N
    batched = edge_base.unsqueeze(2) + offsets.view(1, 1, -1)
    edge_index_global = batched.permute(0, 2, 1).reshape(2, -1)

    batch = torch.arange(B, device=device).repeat_interleave(N)
    return edge_index_global, batch


def get_edge_attr(fourmomenta, edge_index, eps=1e-10, use_float64=True):
    if use_float64:
        in_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
    mij2 = lorentz_squarednorm(fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]])
    edge_attr = mij2.clamp(min=eps).log()
    if use_float64:
        edge_attr = edge_attr.to(in_dtype)
    return edge_attr
