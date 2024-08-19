from typing import Union

import torch

from tensorframes.reps.irreps import Irrep, Irreps
from tensorframes.reps.tensorreps import TensorRep, TensorReps


def extract_even_scalar_mask_from_reps(reps: Union[TensorReps, Irreps]) -> torch.Tensor:
    """Extracts a boolean mask indicating which elements in the input tensor_reps are even scalars.

    Args:
        tensor_reps (Union[TensorReps, Irreps]): The input tensor_reps or irreps.

    Returns:
        torch.Tensor: A boolean mask indicating which elements are even scalars.
    """
    scalar_mask = torch.zeros(reps.dim, dtype=torch.bool)
    start_ind = 0
    for rep_i in reps:
        end_ind = start_ind + rep_i.dim
        if rep_i.rep == TensorRep(order=0, p=1):
            scalar_mask[start_ind:end_ind] = True
        elif rep_i.rep == Irrep(angular_momentum=0, p=1):
            scalar_mask[start_ind:end_ind] = True
        start_ind = end_ind

    return scalar_mask


def parse_reps(reps: Union[TensorReps, Irreps, str]) -> Union[TensorReps, Irreps]:
    """Parses the input representations. I/i means irreps, T/t means tensor representations.

    Args:
        reps (Union[TensorReps, Irreps, str]): The input representations.

    Returns:
        Union[TensorReps, Irreps]: The parsed representations.
    """

    if isinstance(reps, str):
        # remove leading whitespace:
        reps = reps.strip()

        if reps[0].lower() == "i":
            return Irreps(reps[1:])
        elif reps[0].lower() == "t":
            return TensorReps(reps[1:])
        else:
            raise ValueError(f"Invalid representation type {reps}")
    elif isinstance(reps, TensorReps):
        return reps
    elif isinstance(reps, Irreps):
        return reps
    else:
        raise ValueError(f"Invalid representation type {reps}")
