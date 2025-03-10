import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.tensorreps import TensorReps


class TensorRepsTransform(torch.nn.Module):
    def __init__(
        self,
        reps: TensorReps,
        use_naive=False,
    ):
        super().__init__()
        self.reps = reps
        self.transform = (
            self._transform_naive if use_naive else self._transform_efficient
        )

        # cache idx_start and idx_end for each rep
        self.start_end_idx = []
        idx = 0
        for mul_rep in self.reps:
            _, rep = mul_rep
            self.start_end_idx.append([idx, idx + mul_rep.dim])
            idx += mul_rep.dim

        # build parity_mask
        parity_odd = torch.zeros(self.reps.dim, dtype=torch.bool)
        idx = 0
        for mul_rep in self.reps:
            _, rep = mul_rep
            parity_odd[idx : idx + mul_rep.dim] = True if rep.parity else False
            idx += mul_rep.dim
        self.register_buffer("parity_odd", parity_odd.unsqueeze(0))

        if not use_naive:
            # build mapping from order to element in the reps list
            # only used in _transform_efficient
            self.map_rep = [None for _ in range(self.reps.max_rep.rep.order + 1)]
            idx_rep = 0
            for i in range(self.reps.max_rep.rep.order + 1):
                if reps[idx_rep].rep.order == i:
                    self.map_rep[i] = idx_rep
                    idx_rep += 1

    def forward(self, tensor: torch.Tensor, lframes: LFrames):
        """
        Parameters
        ----------
        tensor: torch.tensor of shape (*shape, self.reps.dim)
        lframes: LFrames
            lframes.matrices has shape (*shape, 4, 4)

        Returns
        -------
        tensor_transformed: torch.tensor of shape (*shape, self.reps.dim)
        """
        assert self.reps.dim == tensor.shape[-1]

        if lframes.is_identity or self.reps.mul_without_scalars == 0:
            return tensor

        in_shape = tensor.shape
        assert in_shape[:-1] == lframes.shape[:-2]
        tensor = tensor.reshape(-1, tensor.shape[-1])
        lframes = lframes.reshape(-1, 4, 4)

        tensor_transformed = self.transform(tensor, lframes)
        tensor_transformed = self.transform_parity(tensor_transformed, lframes)

        tensor_transformed = tensor_transformed.reshape(*in_shape)
        return tensor_transformed

    def _transform_naive(self, tensor, lframes):
        """Naive transform: Apply n transformations to a tensor of n'th order"""
        output = tensor.clone()
        for mul_rep, [idx_start, idx_end] in zip(self.reps, self.start_end_idx):
            mul, rep = mul_rep
            if mul == 0 or rep.order == 0:
                continue

            x = tensor[:, idx_start:idx_end].reshape(-1, mul, *([4] * rep.order))

            einsum_string = get_einsum_string(rep.order)
            x_transformed = torch.einsum(
                einsum_string, *([lframes.matrices] * rep.order), x
            )
            output[:, idx_start:idx_end] = x_transformed.reshape(-1, mul_rep.dim)

        return output

    def _transform_efficient(self, tensor, lframes):
        """
        Efficient transform:
        Starting with the highest-order tensor contribution,
        add the next contribution, apply lframes transformation
        and flatten first dimension before continueing with next order.

        This is more efficient, because we use the
        maximum amount of parallelization possible.
        """
        output = None
        for order in reversed(range(self.reps.max_rep.rep.order + 1)):
            if self.map_rep[order] is not None:
                # add new contribution to the mix
                idx_start, idx_end = self.start_end_idx[self.map_rep[order]]
                contribution = tensor[:, idx_start:idx_end].reshape(
                    tensor.shape[0], -1, *(order * (4,))
                )
                output = (
                    torch.cat([contribution, output], dim=1)
                    if order < self.reps.max_rep.rep.order
                    else contribution
                )

            if order > 0:
                # apply transformation, then flatten because transformation is done
                output = torch.einsum("ijk,ilk...->ilj...", lframes.matrices, output)
                output = output.flatten(start_dim=1, end_dim=2)

        return output

    def transform_parity(self, tensor, lframes):
        """Parity transform: Multiply parity-odd states by sign(det Lambda)"""
        return torch.where(
            self.parity_odd, lframes.det.sign().unsqueeze(-1) * tensor, tensor
        )


def get_einsum_string(order):
    if order > 12:
        raise NotImplementedError("Running out of letters for order>12")

    einsum = ""
    start = ord("A")
    batch_index = ord("a")

    # list of lframes
    for i in range(order):
        einsum += chr(batch_index) + chr(start + 2 * i) + chr(start + 2 * i + 1) + ","

    # tensor
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i + 1)

    # output
    einsum += "->"
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i)

    return einsum
