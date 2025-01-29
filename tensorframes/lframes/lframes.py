import torch

from tensorframes.utils.lorentz import lorentz_eye, lorentz_metric


class LFrames:
    """
    Class representing a collection of Lorentz transformations
    We use read-only properties (@property, underlying parameters called _parameter)
    to prevent the user from overwriting specific parameters
    We also cache det and inv for performance
    """

    def __init__(
        self,
        matrices: torch.Tensor = None,
        is_global: bool = False,
        is_identity: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        shape: int = None,
    ) -> None:
        """
        Args:
            matrices (torch.Tensor): Tensor of shape (..., 4, 4) representing the rotation matrices.
            is_global (bool): signify global transformations
            is_identity (bool): whether to use identity lframes, implies is_global, this is also assumed if matrices is torch.nn.Identity()
            device (str): device to store the lframe on, only required when is_identity
            n_batch (int): number of batches, only required when is_identity
        """
        self._is_identity = is_identity

        if is_identity:
            assert device and dtype and shape

            self._is_global = True
            self._matrices = lorentz_eye(shape, device=device, dtype=dtype)
            self._shape = shape
            self._device = device
            self._dtype = dtype
        else:
            assert matrices is not None
            assert matrices.shape[-2:] == (
                4,
                4,
            ), f"Transformations must be of shape (..., 4, 4), but found {matrices.shape[-2:]} instead"

            self._is_global = is_global
            self._matrices = matrices
            self._shape = matrices.shape
            self._device = matrices.device
            self._dtype = matrices.dtype

        self._metric = lorentz_metric(
            self._shape[:-2], device=self._device, dtype=self._dtype
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
            if self._is_identity:
                self._det = torch.ones(
                    self._shape, dtype=self._dtype, device=self._device
                )
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
            if not self._is_identity:
                self._inv = (
                    self._metric @ self._matrices.transpose(-1, -2) @ self._metric
                )
            else:
                # identity is its own inverse
                return self._matrices
        return self._inv

    @property
    def matrices(self) -> torch.Tensor:
        return self._matrices

    @property
    def shape(self) -> torch.Tensor:
        return self._shape

    @property
    def is_identity(self) -> torch.Tensor:
        return self._is_identity

    @property
    def is_global(self) -> torch.Tensor:
        return self._is_global


class InverseLFrames(LFrames):
    """Inverse of a collection of Lorentz transformations."""

    def __init__(self, lframes: LFrames) -> None:
        """
        Args:
            lframes (LFrames): The LFrames object.
        """
        super().__init__(
            matrices=lframes.matrices,
            is_global=lframes._is_global,
            is_identity=lframes._is_identity,
            device=lframes._device,
            dtype=lframes._dtype,
            shape=lframes.shape,
        )

        self._matrices = lframes.inv
        self._inv = lframes.matrices
        self._det = lframes.det


class IndexSelectLFrames(LFrames):
    """Index-controlled selection of rotation matrices from an LFrames object."""

    def __init__(self, lframes: LFrames, indices: torch.Tensor):
        super().__init__(
            matrices=lframes.matrices,
            is_global=lframes._is_global,
            is_identity=lframes._is_identity,
            device=lframes._device,
            dtype=lframes._dtype,
            shape=lframes.shape,
        )
        self._shape = indices.shape
        self._indices = indices

        self._matrices = lframes.matrices.index_select(0, indices)
        self._det = lframes.det.index_select(0, indices)
        self._inv = lframes.inv.index_select(0, indices)


class ChangeOfLFrames(LFrames):
    """
    Change of frames between two LFrames objects
    Formally, for L_start and L_end we have
    L_change = L_end * L_start^-1

    WARNING: This function does not mix the lframes of different point clouds
    It is used in TFMessagePassing where this mixing is performed using the edge_index
    """

    def __init__(self, lframes_start: LFrames, lframes_end: LFrames) -> None:
        """
        Args:
            lframes_start (LFrames): The LFrames object from where to start the transform.
            lframes_end (LFrames): The LFrames object in which to end the transform.
        """
        assert (
            lframes_start.shape == lframes_end.shape
        ), "Both LFrames objects must have the same shape."
        super().__init__(
            matrices=lframes_start.matrices,
            is_global=lframes_start._is_global,
            is_identity=lframes_start._is_identity,
            device=lframes_start._device,
            dtype=lframes_start._dtype,
            shape=lframes_start.shape,
        )

        if lframes_start._is_global:
            # get identity transformation
            self._is_identity = True
            self._shape = lframes_start.shape
            self._matrices = lorentz_eye(
                self._shape, device=self._device, dtype=self._dtype
            )
        else:
            self._matrices = lframes_end.matrices @ lframes_start.inv
            self._inv = lframes_start.matrices @ lframes_end.inv
            self._det = lframes_start.det * lframes_end.det


class LowerIndices(LFrames):
    """
    LFrames with lower indices
    Used in InvariantParticleAttention to lower the key indices
    """

    def __init__(self, lframes):
        """
        Args:
            lframes (LFrames): LFrames with indices to be lowered
        """
        super().__init__(
            matrices=lframes.matrices,
            is_global=lframes.is_global,
            is_identity=lframes.is_identity,
            device=lframes._device,
            dtype=lframes._dtype,
            shape=lframes.shape,
        )
        self._matrices = self._metric @ self._matrices
