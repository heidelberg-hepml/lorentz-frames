import torch
from experiments.logger import LOGGER


class LFrames:
    """Class representing a collection of Lorentz transformations."""

    def __init__(self, matrices: torch.Tensor, spatial_dim: int = 4) -> None:
        """Initialize the LFrames class.

        Args:
            matrices (torch.Tensor): Tensor of shape (..., spatial_dim, spatial_dim) representing the rotation matrices.
            spatial_dim (int, optional): Dimension of the spatial vectors. Defaults to 4.
        """
        assert spatial_dim == 4
        assert matrices.shape[-2:] == (
            spatial_dim,
            spatial_dim,
        ), f"Rotations must be of shape (..., spatial_dim, spatial_dim), but found dim {matrices.shape[-2:]} instead"

        self._matrices = matrices
        self.spatial_dim = spatial_dim

        self.metric = torch.diag(
            torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        )
        self._det = None
        self._inv = None

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = torch.linalg.det(self._matrices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.metric @ self._matrices.transpose(-1, -2) @ self.metric
        return self._inv

    @property
    def shape(self) -> torch.Size:
        """Shape of the Lorentz transformation.

        Returns:
            torch.Size: Size of Lorentz transformation.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the Lorentz transformation.

        Returns:
            torch.device: Device of the Lorentz transformation.
        """
        return self.matrices.device

    @property
    def matrices(self) -> torch.Tensor:
        """return the transformation matrices"""
        return self._matrices

    @matrices.setter
    def matrices(self, new_matrices):
        """clear cached values dependent on the matrices for safety"""
        self._inv = None
        self._det = None
        self._matrices = new_matrices

    def inverse_lframes(self) -> "LFrames":
        """Returns the inverse of the LFrames object.

        Returns:
           LFrames: LFrames object containing the inverse rotation matrices.
        """
        return InvLFrames(self)

    def index_select(self, indices: torch.Tensor) -> "LFrames":
        """Selects the rotation matrices corresponding to the given indices.

        Args:
            indices (torch.Tensor): Tensor containing the indices to select.

        Returns:
            LFrames: LFrames object containing the selected rotation matrices.
        """

        return IndexSelectLFrames(self, indices)


class InvLFrames(LFrames):
    """Represents the inverse of a collection of o3 matrices."""

    def __init__(self, lframes: LFrames) -> None:
        """Initialize the InvLFrames class.

        Args:
            lframes (LFrames): The LFrames object.

        Returns:
            None
        """
        self._lframes = lframes
        self.spatial_dim = lframes.spatial_dim

        self._det = None
        self._inv = None
        self._matrices = None

    @property
    def matrices(self) -> torch.Tensor:
        """Returns the matrices stored in the lframes object.

        Returns:
            torch.Tensor: The matrices stored in the lframes object.
        """
        if self._matrices is None:
            self._matrices = self._lframes.inv
        return self._matrices

    @matrices.setter
    def matrices(self, _):
        raise RuntimeError(
            "Attempted to directly set matrices of a InvLFrames object. Try setting the values of the original LFrame instead!"
        )

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self._lframes.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self._lframes.matrices
        return self._inv

    def index_select(self, indices: torch.Tensor) -> LFrames:
        """Selects the rotation matrices corresponding to the given indices.

        Args:
            indices (torch.Tensor): Tensor containing the indices to select.

        Returns:
            LFrames: LFrames object containing the selected rotation matrices.
        """
        return IndexSelectLFrames(self, indices)


class IndexSelectLFrames(LFrames):
    """Represents a selection of rotation matrices from an LFrames object.

    The selection is done on the fly.
    """

    def __init__(self, lframes: LFrames, indices: torch.Tensor) -> None:
        """Initialize the IndexSelectLFrames object.

        Args:
            lframes (LFrames): The LFrames object.
            indices (torch.Tensor): The indices.

        Returns:
            None
        """

        self._lframes = lframes
        self._indices = indices
        self.spatial_dim = lframes.spatial_dim

        self._matrices = None
        self._wigner_cache = {}
        self._det = None
        self._inv = None

    @property
    def matrices(self) -> torch.Tensor:
        """Returns the matrices stored in the lframes object.

        If the matrices have not been initialized, they are initialized by indexing the matrices
        attribute of the lframes object with the indices attribute of the current object.

        Returns:
            torch.Tensor: The matrices stored in the lframes object.
        """
        if self._matrices is None:
            self._matrices = self._lframes.matrices.index_select(0, self._indices)
        return self._matrices

    @matrices.setter
    def matrices(self, _):
        raise RuntimeError(
            "Attempted to directly set matrices of a InvLFrames object. Try setting the values of the original LFrame instead!"
        )

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self._lframes.det.index_select(0, self._indices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self._lframes.inv.index_select(0, self._indices)
        return self._inv

    def index_select(self, indices: torch.Tensor) -> LFrames:
        """Selects the rotation matrices corresponding to the given indices."""
        indexed_indices = self._indices.index_select(0, indices)
        return IndexSelectLFrames(lframes=self._lframes, indices=indexed_indices)

    def inverse_lframes(self) -> LFrames:
        """Returns the original reference to the LFrames object."""
        return InvLFrames(self)


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
        self._lframes_start = lframes_start
        self._lframes_end = lframes_end
        self._matrices = torch.bmm(lframes_end.matrices, lframes_start.inv)
        self.spatial_dim = lframes_start.spatial_dim

        self.metric = torch.diag(
            torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        )
        self._det = None
        self._inv = None

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self._lframes_start.det * self._lframes_end.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.metric @ self._matrices.transpose(-1, -2) @ self.metric
        return self._inv

    @property
    def shape(self) -> torch.Size:
        """Shape of the Lorentz transformation.

        Returns:
            torch.Size: Size of the Lorentz transformation.
        """
        LOGGER.info(
            f"{self.matrices=}, {self._lframes_start.matrices=}, {self._lframes_end.matrices=}"
        )
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the Lorentz transformation.

        Returns:
            torch.device: Device of the Lorentz transformation.
        """
        return self.matrices.device

    @property
    def matrices(self):
        """return transformation matrices"""
        return self._matrices

    def inverse_lframes(self) -> "ChangeOfLFrames":
        """Returns the inverse of the ChangeOfLFrames object.

        Returns:
            ChangeOfLFrames: ChangeOfLFrames object containing the inverse rotation matrices.
        """
        return ChangeOfLFrames(
            lframes_start=self._lframes_end, lframes_end=self._lframes_start
        )


if __name__ == "__main__":
    raise NotImplementedError

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
