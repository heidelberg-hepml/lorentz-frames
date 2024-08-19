import torch


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
