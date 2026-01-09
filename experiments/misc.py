from collections.abc import Mapping

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ModuleNotFoundError:
    pass
try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
except ModuleNotFoundError:
    pass


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device, dtype=dtype)
    return mask


def get_flex_attention_mask(batch: torch.Tensor):
    """Returns a mask for the attention mechanism.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector, maps each token to its sequence in the batch.

    Returns
    -------
    BlockMask
        Block-diagonal BlockMask for flex attention, with one block per sequence.
    """
    N = batch.size(0)

    def jagged_masking(b, h, q_idx, kv_idx):
        return batch[q_idx] == batch[kv_idx]

    mask = create_block_mask(jagged_masking, None, None, N, N, device=batch.device, _compile=True)
    return mask


def get_attention_mask(
    batch: torch.Tensor,
    attention_backend: str,
    dtype: torch.dtype,
):
    """Returns the attention mask according to the backend.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector, maps each token to its sequence in the batch.
    attention_backend : str
        Attention backend to use ("xformers", "flex", or "flash").
    dtype : torch.dtype
        Data type of the attention mask (for xformers backend).

    Returns
    -------
    dict[str, torch.Tensor | BlockMask | BlockDiagonalMask]
        Attention mask for the specified backend.
    """
    on_cpu = batch.device == torch.device("cpu")
    if attention_backend == "xformers":
        mask = get_xformers_attention_mask(batch=batch, dtype=dtype, materialize=on_cpu)
        if not on_cpu:
            return {"attn_bias": mask}
        else:
            # fallback to default attention
            return {"attn_mask": mask}
    elif attention_backend == "flash":
        seqlens = torch.bincount(batch).to(torch.int32)
        maxlen = int(seqlens.max().item())
        cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=seqlens.device), cu_seqlens], dim=0
        )
        if not on_cpu:
            return {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": maxlen,
                "max_seqlen_k": maxlen,
            }
        else:
            # fallback to default attention
            mask = get_xformers_attention_mask(batch=batch, dtype=dtype, materialize=on_cpu)
            return {"attn_mask": mask}
    elif attention_backend == "flex":
        mask = get_flex_attention_mask(batch=batch)
        return {"block_mask": mask}
    else:
        raise ValueError(
            f"Unsupported attention backend: {attention_backend}. "
            'Supported backends are "xformers", "flex", and "flash".'
        )
