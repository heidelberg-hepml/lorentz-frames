import torch
import random
import numpy as np
import os
import inspect
from contextlib import contextmanager
from tensorframes.utils.utils import get_ptr_from_batch


@contextmanager
def track_clamps():
    clamp_calls = []

    original_clamp_fn = torch.clamp
    original_tensor_clamp = torch.Tensor.clamp
    original_tensor_clamp_ = torch.Tensor.clamp_

    def record_clamp_call(op_type, before, after):
        stack = inspect.stack()
        for frame in stack[2:]:
            if "torch" not in frame.filename:
                diff = (before != after).sum().item()
                # if diff==0:
                #    return None
                return {
                    "filename": frame.filename,
                    "line": frame.lineno,
                    "function": frame.function,
                    "code": frame.code_context[0].strip()
                    if frame.code_context
                    else None,
                    "op_type": op_type,
                    "num_elements_clamped": diff,
                    "total_elements": before.numel(),
                }

    def tracking_clamp(*args, **kwargs):
        input_tensor = args[0]
        before = input_tensor.clone()
        result = original_clamp_fn(*args, **kwargs)
        callsite = record_clamp_call("torch.clamp", before, result)
        if callsite:
            clamp_calls.append(callsite)
        return result

    def tracking_tensor_clamp(self, *args, **kwargs):
        before = self.clone()
        result = original_tensor_clamp(self, *args, **kwargs)
        callsite = record_clamp_call("tensor.clamp", before, result)
        if callsite:
            clamp_calls.append(callsite)
        return result

    def tracking_tensor_clamp_(self, *args, **kwargs):
        before = self.clone()
        original_tensor_clamp_(self, *args, **kwargs)
        after = self
        callsite = record_clamp_call("tensor.clamp_", before, after)
        if callsite:
            clamp_calls.append(callsite)
        return self

    torch.clamp = tracking_clamp
    torch.Tensor.clamp = tracking_tensor_clamp
    torch.Tensor.clamp_ = tracking_tensor_clamp_

    try:
        yield clamp_calls  # gives results
    finally:
        # Restore originals
        torch.clamp = original_clamp_fn
        torch.Tensor.clamp = original_tensor_clamp
        torch.Tensor.clamp_ = original_tensor_clamp_


def crop_particles(data, n=None):
    """
    Crops the amount of particles in the jet to the first n

    Args:
        data: dictionary with fourmomenta x, scalars, batch, label, ptr
        n: number of particles to crop to, None for no cropping, defaults to None
    Returns:
        data: changed data
    """
    if n is None:
        return data

    batch = data.batch
    x = data.x
    scalars = data.scalars
    label = data.label

    _, counts = torch.unique_consecutive(batch, return_counts=True)

    # create a mask: first n elements per group
    idx_within_group = torch.cat([torch.arange(c) for c in counts])
    idx_within_group = idx_within_group.to(batch.device)

    mask = idx_within_group < n

    # apply mask
    x = x[mask]
    scalars = scalars[mask]
    batch = batch[mask]
    label = label
    ptr = get_ptr_from_batch(batch)

    data = data.clone()
    data.x = x
    data.scalars = scalars
    data.label = label
    data.batch = batch
    data.ptr = ptr

    return data


def fix_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
