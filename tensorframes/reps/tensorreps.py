from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from tensorframes.lframes.lframes import LFrames


class TensorRep(Tuple):
    """Represents a tensor with its order and p-value.

    Example usage:
    >>> t = TensorRep(3, 1)
    >>> t.order
    3
    >>> t.p
    1
    >>> t.__repr__()
    '3n'
    """

    def __new__(cls, order: Union[int, str, "TensorRep"], p: int = None) -> "TensorRep":
        """Initializes a new TensorRep object.

        Parameters:
            order (int): The order of the tensor.
            p (int, optional): The p-value of the tensor. Default is None.

        If `p` is not provided, it can be inferred from the `order` parameter.
        If `order` is a string, it will be parsed to extract the order and p-value.
        If `order` is a tuple, it will be unpacked to get the order and p-value.
        """
        if p is None:
            if isinstance(order, TensorRep):
                return order

            if isinstance(order, str):
                try:
                    name = order.strip()
                    order = int(name[:-1])
                    p = {
                        "n": 1,
                        "p": -1,
                    }[name[-1]]

                except Exception:
                    raise ValueError(f"Invalid tensor_rep string {order}")
            elif isinstance(order, tuple):
                order, p = order

        assert isinstance(order, int) and order >= 0, order
        assert p in [-1, 1], p
        return super().__new__(cls, (order, p))

    @property
    def order(self) -> int:
        """
        int: The order of the tensor.
        """
        return self[0]

    @property
    def p(self):
        """
        int: The p-value of the tensor.
        """
        return self[1]

    def __repr__(self):
        """
        str: Returns a string representation of the tensor.
        """
        return f"{self[0]}{'n' if self[1] == 1 else 'p'}"


class _TensorMulRep(Tuple):
    """Represents a tensor multiplication with a corresponding tensor representation."""

    def __new__(cls, mul, rep=None):
        """Initializes a new _TensorMulRep object.

        Args:
            mul (int): The multiplier for the tensor representation.
            rep (TensorRep, optional): The tensor representation. If not provided, `mul` is expected to be a tuple of `(mul, rep)`.
        """

        # This needs to be made because of the way deepcopy works
        if rep is None:
            mul, rep = mul

        assert isinstance(mul, int), "mul must be an integer"
        assert isinstance(rep, TensorRep), "rep must be an instance of TensorRep"

        return super().__new__(cls, (mul, rep))

    @property
    def mul(self):
        """
        int: The multiplier for the tensor representation.
        """
        return self[0]

    @property
    def rep(self):
        """
        TensorRep: The tensor representation.
        """
        return self[1]

    @property
    def dim(self):
        """
        int: The dimension of the tensor multiplication.
        """
        if self.rep.order == 0:
            return 1 * self.mul
        else:
            return (3**self.rep.order) * self.mul

    # to be checked how exactly the memo thing works
    # def __deepcopy__(self, memodict={}):
    #     return _TensorMulRep(self.mul, self.rep)

    def __repr__(self):
        """
        str: Returns a string representation of the tensor multiplication.
        """
        return f"{self.mul}x{self.rep}"


class TensorReps(Tuple):
    """Represents a collection of tensor representations."""

    def __new__(cls, tensor_reps, spatial_dim=3):
        """Initializes a new TensorReps object.

        Args:
            tensor_reps (Union[TensorReps, str, List[Tuple[int, TensorRep]]]): The tensor reps to initialize the object with.
                If `tensor_reps` is an instance of `TensorReps`, a copy of `tensor_reps` is created.
                If `tensor_reps` is a string, it is parsed to extract the reps.
                If `tensor_reps` is a list of tuples, each tuple represents a tensor irrep, where the first element is the multiplicity and the second element is the `TensorRep` object.

            spatial_dim (int, optional): The spatial dimension of the tensor. Defaults to 3.
        """
        if isinstance(tensor_reps, TensorReps):
            tensor_rep = super().__new__(cls, tensor_reps)
        elif isinstance(tensor_reps, str):
            out = []
            try:
                # remove whitespace
                tensor_reps = tensor_reps.replace(" ", "")
                # split into single TensorMulRep
                tensor_reps_list = tensor_reps.split("+")
                for i in range(len(tensor_reps_list)):
                    if tensor_reps_list[i][-1] == "n":
                        p = 1
                        tensor_reps_list[i] = tensor_reps_list[i][:-1]
                    elif tensor_reps_list[i][-1] == "p":
                        p = -1
                        tensor_reps_list[i] = tensor_reps_list[i][:-1]
                    else:
                        p = 1

                    mul, order = tensor_reps_list[i].split("x")
                    mul = int(mul)
                    order = int(order)
                    out.append(_TensorMulRep(mul, TensorRep(order, p)))
                    tensor_rep = super().__new__(cls, out)

            except Exception:
                raise ValueError(f"Invalid tensor_reps string {tensor_reps}")

        else:
            out = []
            for mul_rep in tensor_reps:
                mul = None
                rep = None

                if isinstance(mul_rep, _TensorMulRep):
                    mul = mul_rep.mul
                    rep = mul_rep.rep
                elif len(mul_rep) == 2:
                    mul = mul_rep[0]
                    rep = mul_rep[1]

                if not isinstance(mul, int):
                    raise ValueError("Can't parse tensor_reps")

                out.append(_TensorMulRep(mul, rep))
            tensor_rep = super().__new__(cls, out)

        return tensor_rep

    def __init__(self, tensor_reps, spatial_dim=3) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self._dim = None

    def __repr__(self):
        """
        str: Returns a string representation of the tensor reps.
        """
        return "+".join(f"{mul_ir}" for mul_ir in self)

    @property
    def dim(self) -> int:
        """
        int: The total dimension of the tensor reps.
        """
        if self._dim is None:
            self._dim = sum(mul_ir.dim for mul_ir in self)
        return self._dim

    @property
    def max_rep(self) -> TensorRep:
        """
        TensorRep: The tensor irrep with the highest order.
        """
        return max(self, key=lambda x: x.rep.order)

    @property
    def mul_without_scalars(self) -> int:
        """
        int: The total multiplier of the tensor reps without the scalars.
        """
        return sum(mul_ir.mul for mul_ir in self if mul_ir.rep != 0)

    @property
    def mul(self) -> int:
        """
        int: The total multiplier of the tensor reps.
        """
        return sum(mul_ir.mul for mul_ir in self)

    @property
    def reps(self) -> set:
        """Set[TensorRep]: The set of tensor reps."""
        return {rep for _, rep in self}

    def __add__(self, tensor_reps) -> "TensorReps":
        """Adds tensor reps to the current tensor reps.

        Args:
            tensor_reps (TensorReps): The tensor reps to add.

        Returns:
            TensorReps: The sum of the tensor reps.
        """

        tensor_reps = TensorReps(tensor_reps)
        return TensorReps(super().__add__(tensor_reps))

    def simplify(self) -> "TensorReps":
        """Simplifies the tensor reps by combining the same reps.

        Returns:
            TensorReps: The simplified tensor reps.
        """
        out = []

        for mul, rep in self:
            if out and out[-1][1] == rep:
                out[-1] = (out[-1][0] + mul, rep)
            elif mul > 0:
                out.append((mul, rep))

        return TensorReps(out)

    def get_transform_class(self, use_parallel: bool = True, avoid_einsum: bool = False):
        """Returns the tensor reps transform class.

        Args:
            use_parallel (bool, optional): Whether to use parallel computation for the transformation. Defaults to True.
            avoid_einsum (bool, optional): Whether to avoid using `torch.einsum` for the transformation. Defaults to False.

        Returns:
            TensorRepsTransform: The tensor reps transform class.
        """
        return TensorRepsTransform(self, use_parallel, avoid_einsum)

    def sort(self):
        """Sorts the tensor reps by the order of the reps.

        Returns:
            TensorReps: The sorted tensor reps.
        """
        # TODO: look at the IRREPS class in e3nn
        return TensorReps(sorted(self, key=lambda x: x.rep.order))


class TensorRepsTransform(Module):
    """A module for transforming tensor representations.

    Args:
        tensor_reps (TensorReps): The tensor representations to be transformed.
        use_parallel (bool, optional): Whether to use parallel computation for the transformation. Defaults to True.
        avoid_einsum (bool, optional): Whether to avoid using `torch.einsum` for the transformation. Defaults to False.
    """

    def __init__(
        self, tensor_reps: TensorReps, use_parallel: bool = True, avoid_einsum: bool = False
    ):
        """Initialize a TensorReps object.

        Args:
            tensor_reps (TensorReps): The tensor representations.
            use_parallel (bool, optional): Whether to use parallel computation. Defaults to True.
            avoid_einsum (bool, optional): Whether to avoid using einsum. Defaults to False.
        """
        super().__init__()
        self.tensor_reps = tensor_reps
        self.use_parallel = use_parallel
        self.avoid_einsum = avoid_einsum
        self.spatial_dim = tensor_reps.spatial_dim

        # prepare for fast transform
        n_start_index_dict = {n: [] for n in range(0, self.tensor_reps.max_rep.rep.order + 1)}
        pseudo_mask = torch.zeros(self.tensor_reps.dim, dtype=bool)
        start_idx = 0
        for i, mul_reps in enumerate(self.tensor_reps):
            n_start_index_dict[mul_reps.rep.order].append((start_idx, i))

            # prepare parity mask:
            if mul_reps[1].p == -1:
                pseudo_mask[start_idx : start_idx + mul_reps.dim] = True

            start_idx += mul_reps.dim

        # sort start indices and reps by angular momentum largest first, n_masks can also be precomputed
        self.sorted_n = sorted(n_start_index_dict.keys(), reverse=True)
        self.is_sorted = all(
            mul_reps.rep.order == self.sorted_n[::-1][i]
            for i, mul_reps in enumerate(self.tensor_reps)
        )

        self.n_masks = []
        self.n_muls = []
        self.start_end_indices = []
        self.scalar_dim = 0
        for n in self.sorted_n:
            n_mask = torch.zeros(self.tensor_reps.dim, dtype=torch.bool)
            mul_per_n = 0
            n_start_idx = self.tensor_reps.dim
            n_end_idx = 0
            # concat all the reps with the same angular momentum
            for start_idx, rep_idx in n_start_index_dict[n]:
                end_idx = start_idx + self.tensor_reps[rep_idx].dim
                n_mask[start_idx:end_idx] = True
                mul_per_n += self.tensor_reps[rep_idx].mul
                n_start_idx = min(n_start_idx, start_idx)
                n_end_idx = max(n_end_idx, end_idx)
            if n == 0:
                self.scalar_dim = mul_per_n
            else:
                self.n_muls.append(mul_per_n)
                self.n_masks.append(n_mask)
                self.start_end_indices.append((n_start_idx, n_end_idx))

        # remove scalars from l_sorted:
        self.sorted_n.remove(0)
        pseudo_tensor = torch.where(
            pseudo_mask, -torch.ones(self.tensor_reps.dim), torch.ones(self.tensor_reps.dim)
        ).float()
        self.register_buffer("pseudo_tensor", pseudo_tensor)

    def _get_einsum_string(self, order: int) -> str:
        """Generate the einsum string for a given order.

        Args:
            order (int): The order of the einsum string.

        Returns:
            str: The generated einsum string.

        Raises:
            NotImplementedError: If the order is greater than 12.
        """
        if order > 12:
            raise NotImplementedError("Not implemented for more than order 12")

        einsum = ""

        start = ord("A")

        batch_index = ord("a")

        for i in range(order):
            einsum += chr(batch_index) + chr(start + 2 * i) + chr(start + 2 * i + 1) + ","

        einsum += chr(batch_index)

        einsum += chr(start + 2 * order + 1)

        for i in range(order):
            einsum += chr(start + 2 * i + 1)

        einsum += "->"

        einsum += chr(batch_index)

        einsum += chr(start + 2 * order + 1)

        for i in range(order):
            einsum += chr(start + 2 * i)

        return einsum

    def transform_coeffs_parallel(
        self,
        coeffs: Tensor,
        basis_change: LFrames,
        avoid_einsum: bool = False,
        inplace: bool = False,
    ) -> Tensor:
        """Transforms the coefficients using more parallel computation.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.
            avoid_einsum (bool, optional): Whether to avoid using einsum for the transformation. Defaults to False.
            inplace (bool, optional): Whether to perform the transformation inplace. Defaults to False.

        Returns:
            Tensor: The transformed coefficients.
        """

        if self.tensor_reps.dim == 0:
            return coeffs

        if isinstance(basis_change, torch.Tensor):
            basis_change = LFrames(basis_change)

        if self.pseudo_tensor.device != coeffs.device:
            self.pseudo_tensor = self.pseudo_tensor.to(coeffs.device)

        N = coeffs.shape[0]
        rot_matrix_t = basis_change.inv

        if inplace:
            output_coeffs = coeffs
        else:
            output_coeffs = coeffs.clone()

        largest_tensor = torch.tensor([], device=coeffs.device)
        for i, l in enumerate(self.sorted_n):
            if self.is_sorted:
                start_idx, end_idx = self.start_end_indices[i]
                smaller_tensor = coeffs[:, start_idx:end_idx].view(
                    N, -1, *(l * (self.spatial_dim,))
                )
            else:
                n_mask = self.n_masks[i]
                smaller_tensor = coeffs[:, n_mask].view(N, -1, *(l * (self.spatial_dim,)))

            if i == 0:
                # highest n
                largest_tensor = smaller_tensor
            else:
                largest_tensor = torch.cat([smaller_tensor, largest_tensor], dim=1)

            if avoid_einsum:
                # apply transformation at axis 2 (0 is batch, 1 is channel)
                # move the axis 2 to the last index transform it and move it back
                # this is to avoid einsum (maybe there is a better way)
                largest_tensor = largest_tensor.moveaxis(2, -1)
                largest_shape = largest_tensor.shape
                largest_tensor = torch.matmul(
                    largest_tensor.reshape(N, -1, self.spatial_dim), rot_matrix_t
                )
                largest_tensor = largest_tensor.reshape(*largest_shape)
                largest_tensor = largest_tensor.moveaxis(-1, 2)
            else:
                largest_tensor = torch.einsum(
                    "ijk,ilk...->ilj...", basis_change.matrices, largest_tensor
                )

            # no need to transform again along this axis so flatten it for now into channels:
            largest_tensor = largest_tensor.flatten(start_dim=1, end_dim=2)

        # all computations are now done in largest_tensor and have to be unpacked into coeffs:
        if self.is_sorted:
            output_coeffs[:, self.scalar_dim :] = largest_tensor
        else:
            n_mask_rev = self.n_masks[::-1]
            n_muls_rev = self.n_muls[::-1]
            for i, n in enumerate(self.sorted_n[::-1]):
                if n == 0:
                    # scalars
                    continue

                # this could be faster if things where sorted. then largest would just be coeffs
                n_mask = n_mask_rev[i]
                l_mul = n_muls_rev[i]
                smaller_tensor = largest_tensor[:, : l_mul * 3**n]
                output_coeffs[:, n_mask] = smaller_tensor
                largest_tensor = largest_tensor[:, l_mul * 3**n :]

        # apply parity:
        # get the determinants of the rotation matrices:
        is_det_neg = basis_change.det < 0
        output_coeffs[is_det_neg] = output_coeffs[is_det_neg] * self.pseudo_tensor

        return output_coeffs

    def transform_coeffs(self, coeffs: Tensor, basis_change: LFrames) -> Tensor:
        """Transforms the coefficients using less parallel computation.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.

        Returns:
            Tensor: The transformed coefficients.
        """
        current_index = 0
        length = 0

        output = torch.zeros_like(coeffs)

        for mul_reps in self.tensor_reps:
            mul, rep = mul_reps

            rep_n = rep.order

            length = mul_reps.dim

            left_index = current_index
            right_index = current_index + length

            if rep_n == 0:
                if rep.p == -1:
                    det_sign = basis_change.det.sign()
                    output[:, left_index:right_index] = torch.einsum(
                        "i,ij->ij", det_sign, coeffs[:, left_index:right_index]
                    )
                else:
                    output[:, left_index:right_index] = coeffs[:, left_index:right_index]
                current_index += length
                continue

            einsum_str = self._get_einsum_string(rep_n)

            tensor = coeffs[:, left_index:right_index].reshape(
                coeffs.shape[0], mul, *([3] * rep_n)
            )

            trafo_tensors = torch.einsum(einsum_str, *([basis_change.matrices] * rep_n), tensor)

            output[:, left_index:right_index] = trafo_tensors.flatten(start_dim=1)

            if rep.p == -1:
                det_sign = basis_change.det.sign()
                output[:, left_index:right_index] = torch.einsum(
                    "i,ij->ij", det_sign, trafo_tensors.flatten(start_dim=1)
                )
            else:
                output[:, left_index:right_index] = trafo_tensors.flatten(start_dim=1)

            current_index += length

        return output

    def forward(
        self, coeffs: Tensor | None, basis_change: LFrames, inplace: bool = False
    ) -> Tensor:
        """Applies the forward transformation to the input coefficients.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.
            inplace (bool, optional): Whether to perform the transformation inplace. Only relevant for parallel trafo. Defaults to False.

        Returns:
            Tensor: The transformed coefficients.
        """
        if coeffs is None:
            assert self.tensor_reps.dim == 0, "No coeffs are provided for non-trivial transform"
            return None

        if self.use_parallel:
            return self.transform_coeffs_parallel(
                coeffs, basis_change, self.avoid_einsum, inplace=inplace
            )
        else:
            return self.transform_coeffs(coeffs, basis_change)
