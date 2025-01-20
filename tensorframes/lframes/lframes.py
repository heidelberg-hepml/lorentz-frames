import torch


class LFrames:
    """Class representing a collection of Lorentz transformations."""

    def __init__(
        self,
        matrices: torch.Tensor = None,
        is_global: bool = False,
        is_identity: bool = False,
        device: str = None,
        n_batch: int = None,
    ) -> None:
        """Initialize the LFrames class.

        Args:
            matrices (torch.Tensor): Tensor of shape (..., 4, 4) representing the rotation matrices.
            is_global (bool): signify global transformations
            is_identity (bool): whether to use identity lframes, implies is_global, this is also assumed if matrices is torch.nn.Identity()
            device (str): device to store the lframe on, only required when is_identity
            n_batch (int): number of batches, only required when is_identity
        """
        if is_identity:
            assert is_global and is_identity and device and n_batch
            self._is_identity = True
            self._device = device
            self._n_batch = n_batch
            self._dtype = None
            self._matrices = None
        else:
            assert (
                matrices is not None
            ), "No transformation matrices has been passed to the LFrames constructor."

            assert matrices.shape[-2:] == (
                4,
                4,
            ), f"Transformations must be of shape (..., 4, 4) or (4, 4), but found dim {matrices.shape[-2:]} instead"

            self._is_identity = False
            self._device = matrices.device
            self._dtype = matrices.dtype or None
            self._matrices = matrices

        self._is_global = is_global

        self.metric = torch.diag(
            torch.tensor(
                [1.0, -1.0, -1.0, -1.0], device=self._device, dtype=self._dtype
            )
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
            if self.is_identity:
                self._det = torch.tensor(1).repeat(self._n_batch)
            else:
                self._det = torch.linalg.det(self._matrices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            if self.is_identity:
                return None
            else:
                self._inv = self.metric @ self._matrices.transpose(-1, -2) @ self.metric
        return self._inv

    @property
    def shape(self) -> torch.Size:
        """Shape of the Lorentz transformation.

        Returns:
            torch.Size: Size of Lorentz transformation.
        """
        if self.is_identity:
            return (self._n_batch, 4, 4)
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the Lorentz transformation.

        Returns:
            torch.device: Device of the Lorentz transformation.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the Lorentz transformation.

        Returns:
            torch.dtype: Dtype of the Lorentz transformation.
        """
        return self._dtype

    @property
    def matrices(self) -> torch.Tensor:
        """return the transformation matrices"""
        if self.is_identity:
            return None
        return self._matrices

    @matrices.setter
    def matrices(self, new_matrices: torch.Tensor):
        """clear cached values dependent on the matrices for safety, this is still not recommended"""
        if self.is_identity:
            raise RuntimeError(
                "Can not change the matrices when is_identity is set in the definition."
            )
        self._inv = None
        self._det = None
        self._device = new_matrices.device
        self._matrices = new_matrices
        self._is_global = False

    @property
    def is_global(self):
        """check whether this is a global transformation"""
        return self._is_global

    @property
    def is_identity(self):
        """check whether this is a identity transformation"""
        return self._is_identity

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
    """Represents the inverse of a collection of Lorentz transformations."""

    def __init__(self, lframes: LFrames) -> None:
        """Initialize the InvLFrames class.

        Args:
            lframes (LFrames): The LFrames object.

        Returns:
            None
        """
        self._lframes = lframes
        self._is_global = lframes.is_global
        self._is_identity = lframes.is_identity

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
        """Determinant of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self._lframes.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the Lorentz transformation.

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
        self._is_global = lframes.is_global
        self._is_identity = lframes.is_identity
        self._n_batch = indices.shape[0]

        self._matrices = None
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
        if self._is_identity:
            return None
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
        """Determinant of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self._lframes.det.index_select(0, self._indices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the Lorentz transformation.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._is_identity:
            return None
        if self._inv is None:
            self._inv = self._lframes.inv.index_select(0, self._indices)
        return self._inv

    @property
    def device(self):
        return self._lframes.device

    @property
    def dtype(self):
        return self._lframes.dtype

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
        assert (
            lframes_start.device == lframes_end.device
        ), f"Both lframes need to be on the same device, but found {lframes_start.device} and {lframes_end.device}."
        self._lframes_start = lframes_start
        self._lframes_end = lframes_end
        self._device = lframes_start.device
        if self._lframes_start.is_global:
            # this makes it so that transformations in a global frame setting become identities, which are then skipped in tensorreps.py (TensorRepsTransform)
            self._matrices = None
            self._is_identity = True
        else:
            self._matrices = torch.bmm(lframes_end.matrices, lframes_start.inv)
            self._is_identity = False

        self.metric = torch.diag(
            torch.tensor(
                [1.0, -1.0, -1.0, -1.0],
                device=self.device,
                dtype=lframes_start.dtype,
            )
        )
        self._det = None
        self._inv = None

    @property
    def is_identity(self) -> bool:
        return self._is_identity

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
        if self._is_identity:
            return None
        if self._inv is None:
            self._inv = self.metric @ self._matrices.transpose(-1, -2) @ self.metric
        return self._inv

    @property
    def shape(self) -> torch.Size:
        """Shape of the Lorentz transformation.

        Returns:
            torch.Size: Size of the Lorentz transformation.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the Lorentz transformation.

        Returns:
            torch.device: Device of the Lorentz transformation.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the Lorentz transformation.

        Returns:
            torch.dtype: Dtype of the Lorentz transformation.
        """
        return self._lframes_start.dtype

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
