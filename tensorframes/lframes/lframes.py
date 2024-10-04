import torch

from tensorframes.utils.wigner import _Jd, euler_angles_yxy, wigner_D_from_matrix


class LFrames:
    """Class representing a collection of o3 matrices."""

    def __init__(
        self, matrices: torch.Tensor, cache_wigner: bool = True, spatial_dim: int = 3
    ) -> None:
        """Initialize the LFrames class.

        Args:
            matrices (torch.Tensor): Tensor of shape (..., spatial_dim, spatial_dim) representing the rotation matrices.
            cache_wigner (bool, optional): Whether to cache the Wigner D matrices. Defaults to True.
            spatial_dim (int, optional): Dimension of the spatial vectors. Defaults to 3.

        .. note::
            So far this only supports 3D rotations.
        """
        assert spatial_dim == 3, "So far only 3D rotations are supported."
        assert matrices.shape[-2:] == (
            spatial_dim,
            spatial_dim,
        ), "Rotations must be of shape (..., spatial_dim, spatial_dim)"

        self.matrices = matrices
        self.spatial_dim = spatial_dim

        self._det = None
        self._inv = None
        self._angles = None

        self.cache_wigner = cache_wigner
        self.wigner_cache = {}

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = torch.linalg.det(self.matrices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.matrices.transpose(-1, -2)
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            self._angles = euler_angles_yxy(self.matrices)
        return self._angles

    @property
    def shape(self) -> torch.Size:
        """Shape of the o3 matrices.

        Returns:
            torch.Size: Size of the o3 matrices.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the o3 matrices.

        Returns:
            torch.device: Device of the o3 matrices.
        """
        return self.matrices.device

    def index_select(self, indices: torch.Tensor) -> "LFrames":
        """Selects the rotation matrices corresponding to the given indices.

        Args:
            indices (torch.Tensor): Tensor containing the indices to select.

        Returns:
            LFrames: LFrames object containing the selected rotation matrices.
        """

        new_lframes = LFrames(
            self.matrices.index_select(0, indices),
            cache_wigner=self.cache_wigner,
            spatial_dim=self.spatial_dim,
        )

        # need to copy the attributes if they are not None
        if self._det is not None:
            new_lframes._det = self.det.index_select(0, indices)
        if self._inv is not None:
            new_lframes._inv = self.inv.index_select(0, indices)
        if self._angles is not None:
            new_lframes._angles = self.angles.index_select(0, indices)

        if self.cache_wigner and self.wigner_cache is not {}:
            for l in self.wigner_cache:
                new_lframes.wigner_cache[l] = self.wigner_cache[l].index_select(
                    0, indices
                )

        return new_lframes

    def wigner_D(self, l: int, J: torch.Tensor) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        if self.cache_wigner and l in self.wigner_cache:
            return self.wigner_cache[l]
        else:
            wigner = wigner_D_from_matrix(
                l, self.det[:, None, None] * self.matrices, J=J, angles=self.angles
            )  # * self.det to ensure wigner gets rotation matrix
            if self.cache_wigner:
                self.wigner_cache[l] = wigner
            return wigner


class ChangeOfLFrames:
    """Represents a change of frames between two LFrames objects."""

    def __init__(self, lframes_start: LFrames, lframes_end: LFrames) -> None:
        """Initialize the ChangeOfLFrames class.

        Args:
            lframes_start (LFrames): The LFrames object from where to start the transform.
            lframes_end (LFrames): The LFrames object in which to end the transform.
        """
        assert (
            lframes_start.shape == lframes_end.shape
        ), "Both LFrames objects must have the same shape."
        self.lframes_start = lframes_start
        self.lframes_end = lframes_end
        self.matrices = torch.bmm(lframes_end.matrices, lframes_start.inv)
        self.spatial_dim = lframes_start.spatial_dim

        self._det = None
        self._inv = None
        self._angles = None

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self.lframes_start.det * self.lframes_end.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.matrices.transpose(-1, -2)
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            self._angles = euler_angles_yxy(self.matrices)
        return self._angles

    @property
    def shape(self) -> torch.Size:
        """Shape of the o3 matrices.

        Returns:
            torch.Size: Size of the o3 matrices.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the o3 matrices.

        Returns:
            torch.device: Device of the o3 matrices.
        """
        return self.matrices.device

    def wigner_D(self, l: int, J: torch.Tensor) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        # check if both LFrames objects have the Wigner D matrices cached:
        if l in self.lframes_start.wigner_cache and l in self.lframes_end.wigner_cache:
            wigner_start = self.lframes_start.wigner_cache[l]
            wigner_end = self.lframes_end.wigner_cache[l]
            return torch.bmm(wigner_end, wigner_start.transpose(-1, -2))
        else:
            return wigner_D_from_matrix(
                l, self.det[:, None, None] * self.matrices, J=J, angles=self.angles
            )


if __name__ == "__main__":
    # Example usage:
    matrices = torch.rand(2, 3, 3)
    lframes = LFrames(matrices)
    _Jd = [J.to(matrices.dtype).to(matrices.device) for J in _Jd]
    print("wigner_d for l=2:", lframes.wigner_D(2, J=_Jd[2]))

    matrices2 = torch.rand(2, 3, 3)
    lframes2 = LFrames(matrices2)
    print("wigner_d for l=2:", lframes2.wigner_D(2, J=_Jd[2]))
    print("wigner_d for l=0:", lframes2.wigner_D(0, J=_Jd[1]))

    change = ChangeOfLFrames(lframes, lframes2)
    print("wigner_d for l=1:", change.wigner_D(1, J=_Jd[1]))
    print("wigner_d for l=2:", change.wigner_D(2, J=_Jd[2]))
