from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.utils.wigner import _Jd


class Irrep(Tuple):
    """Tuple representing a single irreducible representation (Irrep) of angular momentum.

    Examples:
        >>> irrep1 = Irrep(2, 1)
        >>> irrep1.angular_momentum
        2
        >>> irrep1.p
        1

        >>> irrep2 = Irrep("3p")
        >>> irrep2.angular_momentum
        3
        >>> irrep2.p
        -1
    """

    def __new__(cls, angular_momentum: Union[int, str, "Irrep"], p: int = None) -> "Irrep":
        """Initializes the `Irrep` object.

        Args:
            angular_momentum (int): The value of the angular momentum.
            p (int, optional): The parity of the Irrep, should be -1 or 1. Defaults to None.

        Returns:
            Irrep: An instance of the Irrep class.
        """

        if p is None:
            if isinstance(angular_momentum, Irrep):
                return angular_momentum

            if isinstance(angular_momentum, str):
                try:
                    name = angular_momentum.strip()
                    angular_momentum = int(name[:-1])
                    p = {
                        "n": 1,
                        "p": -1,
                    }[name[-1]]

                except Exception:
                    raise ValueError("Invalid tensor_rep string")
            elif isinstance(angular_momentum, tuple):
                angular_momentum, p = angular_momentum

        assert isinstance(angular_momentum, int) and angular_momentum >= 0, angular_momentum
        assert p in [-1, 1], p
        return super().__new__(cls, (angular_momentum, p))

    @property
    def angular_momentum(self) -> int:
        """
        int: The value of the angular momentum.
        """
        return self[0]

    @property
    def p(self) -> int:
        """
        int: The parity of the Irrep.
        """
        return self[1]

    def __repr__(self) -> str:
        return f"{self[0]}{'n' if self[1] == 1 else 'p'}"


class _IrMulRep(Tuple):
    """Tuple which represents a multiplication of a scalar value and an instance of `Irrep`."""

    def __new__(cls, mul: int, rep: Irrep = None) -> "_IrMulRep":
        """Initializes the `_IrMulRep` object.

        Args:
            mul (int): The multiplicity of the `Irrep`.
            rep (Irrep, optional): An instance of `Irrep`. If not provided, `mul` is expected to be a tuple of `(mul, rep)`.

        Returns:
            _IrMulRep: An instance of `_IrMulRep`.
        """
        # This is necessary because of the way how deepcopy works
        if rep is None:
            mul, rep = mul

        assert isinstance(mul, int), "mul must be an integer"
        assert isinstance(rep, Irrep), "rep must be an instance of TensorRep"

        return super().__new__(cls, (mul, rep))

    @property
    def mul(self) -> int:
        """
        int: multiplicity.
        """
        return self[0]

    @property
    def rep(self) -> Irrep:
        """
        Irrep: The instance of `Irrep`.
        """
        return self[1]

    @property
    def dim(self) -> int:
        """
        int: The dimension of the `_IrMulRep` object.
        """
        if self.rep.angular_momentum == 0:
            return 1 * self.mul
        else:
            return (2 * self.rep.angular_momentum + 1) * self.mul

    def __repr__(self):
        return f"{self.mul}x{self.rep}"


class Irreps(Tuple):
    """Represents a collection of irreducible representations (irreps) of a group."""

    def __new__(cls, irreps, spatial_dim=3):
        """Initializes the `Irreps` object.

        Args:
            irreps (Union[Irreps, str, List[Tuple[int, Irrep]]]): The irreps to initialize the object with.
                - If `irreps` is an instance of `Irreps`, it creates a new `Irreps` object with the same irreps.
                - If `irreps` is a string, it parses the string and creates the irreps accordingly.
                - If `irreps` is a list of tuples, each tuple should contain an integer representing the multiplicity
                and an instance of `Irrep` representing the irreducible representation.

        Returns:
            Irreps: An instance of `Irreps`.
        """
        if isinstance(irreps, Irreps):
            irrep = super().__new__(cls, irreps)
        elif isinstance(irreps, str):
            out = []
            try:
                # remove whitespace
                irreps = irreps.replace(" ", "")
                # split into single IrMulRep
                irreps_list = irreps.split("+")
                for i in range(len(irreps_list)):
                    if irreps_list[i][-1] == "n":
                        p = 1
                        irreps_list[i] = irreps_list[i][:-1]
                    elif irreps_list[i][-1] == "p":
                        p = -1
                        irreps_list[i] = irreps_list[i][:-1]
                    else:
                        p = 1

                    mul, angular_momentum = irreps_list[i].split("x")
                    mul = int(mul)
                    angular_momentum = int(angular_momentum)
                    out.append(_IrMulRep(mul, Irrep(angular_momentum, p)))
                    irrep = super().__new__(cls, out)

            except Exception:
                raise ValueError(f"Invalid irreps string {irreps}")

        else:
            out = []
            for mul_rep in irreps:
                mul = None
                rep = None

                if isinstance(mul_rep, _IrMulRep):
                    mul = mul_rep.mul
                    rep = mul_rep.rep
                elif len(mul_rep) == 2:
                    mul = mul_rep[0]
                    rep = mul_rep[1]

                if not isinstance(mul, int):
                    raise ValueError("Can't parse irreps")

                out.append(_IrMulRep(mul, rep))
            irrep = super().__new__(cls, out)

        return irrep

    def __init__(self, irreps, spatial_dim=3) -> None:
        self.spatial_dim = spatial_dim
        self._dim = None

    def __repr__(self) -> str:
        """
        str: Returns a string representation of the `Irreps` object.
        """
        return "+".join(f"{mul_ir}" for mul_ir in self)

    @property
    def dim(self) -> int:
        """
        int: The total dimension of the `Irreps` object.
        """
        if self._dim is None:
            self._dim = sum(mul_ir.dim for mul_ir in self)
        return self._dim

    @property
    def max_rep(self) -> _IrMulRep:
        """
        _IrMulRep: The irreducible representation with the highest angular momentum.
        """
        return max(self, key=lambda x: x.rep.angular_momentum)

    @property
    def mul_without_scalars(self) -> int:
        """
        int: The total multiplicity of the `Irreps` object without the scalar representation.
        """
        return sum(mul_ir.mul for mul_ir in self if mul_ir.rep != 0)

    @property
    def mul(self) -> int:
        """
        int: The total multiplicity of the `Irreps` object.
        """
        return sum(mul_ir.mul for mul_ir in self)

    @property
    def reps(self) -> set:
        """Set[Irrep]: The set of irreducible representations in the `Irreps` object."""
        return {rep for _, rep in self}

    def __add__(self, irreps) -> "Irreps":
        """Adds two `Irreps` objects together.

        Args:
            irreps (Irreps): The `Irreps` object to add.

        Returns:
            Irreps: The sum of the two `Irreps` objects.
        """
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def simplify(self) -> "Irreps":
        """Simplifies the `Irreps` object by combining representations with the same parity.

        Returns:
            Irreps: The simplified `Irreps` object.
        """
        out = []

        for mul, rep in self:
            if out and out[-1][1] == rep:
                out[-1] = (out[-1][0] + mul, rep)
            elif mul > 0:
                out.append((mul, rep))

        return Irreps(out)

    def get_transform_class(self) -> "IrrepsTransform":
        """Returns an instance of the `IrrepsTransform` class based on the `Irreps` object.

        Returns:
            IrrepsTransform: An instance of the `IrrepsTransform` class.
        """
        return IrrepsTransform(self)

    def sort(self):
        """Sorts the irreps by angular momentum in descending order.

        Returns:
            Irreps: The sorted `Irreps` object.
        """
        # TODO: look at the IRREPS class in e3nn
        return Irreps(sorted(self, key=lambda x: x.rep.angular_momentum))


class IrrepsTransform(Module):
    """A module that performs a transformation on coefficients based on irreducible representations
    (irreps).

    Args:
        irreps (Irreps): The irreducible representations to be used for the transformation.

    Attributes:
        irreps (Irreps): The irreducible representations used for the transformation.
        sorted_l (List[int]): A list of angular momentum values sorted in descending order.
        is_sorted (bool): Indicates whether the irreps are sorted by angular momentum in descending order.
        l_masks (List[Tensor]): A list of masks for each angular momentum value.
        l_muls (List[int]): A list of multiplicities for each angular momentum value.
        start_end_indices (List[Tuple[int, int]]): A list of start and end indices for each angular momentum value.
        scalar_dim (int): The dimension of the scalar representation.
        odd_tensor (Tensor): A tensor used for handling odd masks.
    """

    def __init__(self, irreps: Irreps) -> None:
        """Initializes the IrrepsTransform module.

        Args:
            irreps (Irreps): The irreducible representations to be used for the transformation.
        """
        super().__init__()
        self.irreps = irreps

        # prepare for fast transform
        l_start_index_dict = {
            l_val: [] for l_val in range(0, self.irreps.max_rep.rep.angular_momentum + 1)
        }
        odd_mask = torch.zeros(self.irreps.dim, dtype=bool)
        start_idx = 0
        for i, mul_reps in enumerate(self.irreps):
            l_start_index_dict[mul_reps.rep.angular_momentum].append((start_idx, i))

            # prepare odd mask:
            if (-1) ** mul_reps.rep.angular_momentum * mul_reps[1].p == -1:
                odd_mask[start_idx : start_idx + mul_reps.dim] = True

            start_idx += mul_reps.dim

        # sort start indices and reps by angular momentum largest first, l_masks can also be precomputed
        self.sorted_l = sorted(l_start_index_dict.keys(), reverse=True)
        self.is_sorted = all(
            mul_reps.rep.angular_momentum == self.sorted_l[::-1][i]
            for i, mul_reps in enumerate(self.irreps)
        )

        self.l_masks = []
        self.l_muls = []
        self.start_end_indices = []
        self.scalar_dim = 0
        for l_val in self.sorted_l:
            l_mask = torch.zeros(self.irreps.dim, dtype=torch.bool)
            mul_per_l = 0
            l_start_idx = self.irreps.dim
            l_end_idx = 0
            # concat all the reps with the same angular momentum
            for start_idx, rep_idx in l_start_index_dict[l_val]:
                end_idx = start_idx + self.irreps[rep_idx].dim
                l_mask[start_idx:end_idx] = True
                mul_per_l += self.irreps[rep_idx].mul
                l_start_idx = min(l_start_idx, start_idx)
                l_end_idx = max(l_end_idx, end_idx)
            if l_val == 0:
                self.scalar_dim = mul_per_l
            else:
                self.l_muls.append(mul_per_l)
                self.l_masks.append(l_mask)
                self.start_end_indices.append((l_start_idx, l_end_idx))

        # remove scalars from l_sorted:
        self.sorted_l.remove(0)
        odd_tensor = torch.where(
            odd_mask, -torch.ones(self.irreps.dim), torch.ones(self.irreps.dim)
        ).float()

        # alternative: calculate these once on the device and then cache them.
        self.register_buffer("odd_tensor", odd_tensor)

        # register the J_matrices as buffer:
        # this is a bit hacky with string in the name:
        for l_val in self.sorted_l:
            self.register_buffer(f"J_matrix_{l_val}", _Jd[l_val].float())

    def forward(
        self, coeffs: Tensor, basis_change: Union[LFrames, ChangeOfLFrames], inplace: bool = False
    ) -> Tensor:
        """Applies the transformation to the input coefficients.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the irreps.
            basis_change (ChangeOfLFrames): The change of frames to be applied. With matrices attribute of shape `(N, 3, 3)`.
            inplace (bool, optional): Whether to perform the transformation in-place. Defaults to False.

        Returns:
            Tensor: The transformed coefficients.
        """

        if coeffs is None:
            assert self.irreps.dim == 0, "No coeffs are provided for non-trivial transform"
            return None

        if inplace:
            output_coeffs = coeffs
        else:
            output_coeffs = coeffs.clone()

        if isinstance(basis_change, torch.Tensor):
            basis_change = LFrames(basis_change)

        if self.irreps.dim == 0:
            return output_coeffs

        if self.odd_tensor.device != coeffs.device:
            self.odd_tensor = self.odd_tensor.to(coeffs.device)

        N = coeffs.shape[0]

        for i, l in enumerate(self.sorted_l):
            # this for loop could be avoided if all l's have the same multiplicity, like in equiformerv2
            if self.is_sorted:
                start_idx, end_idx = self.start_end_indices[i]
                l_tensor = coeffs[:, start_idx:end_idx].view(N, -1, 2 * l + 1)
            else:
                l_mask = self.l_masks[i]
                l_tensor = coeffs[:, l_mask].view(N, -1, 2 * l + 1)

            # perform the transformation:
            J_matrix = getattr(self, f"J_matrix_{l}")
            wigner = basis_change.wigner_D(l, J=J_matrix).transpose(-1, -2)
            if self.is_sorted:
                output_coeffs[:, start_idx:end_idx] = torch.matmul(l_tensor, wigner).flatten(1)
            else:
                output_coeffs[:, l_mask] = torch.matmul(l_tensor, wigner).flatten(1)

        is_det_neg = basis_change.det < 0
        output_coeffs[is_det_neg] = output_coeffs[is_det_neg] * self.odd_tensor

        return output_coeffs
