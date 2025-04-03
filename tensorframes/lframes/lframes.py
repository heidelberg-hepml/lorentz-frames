import torch

from tensorframes.utils.lorentz import lorentz_eye, lorentz_metric


class LFrames:
    """
    Collection of Lorentz transformations, represented as (*dims, 4, 4) matrices.
    Expensive properties like det and inv are cached for performance.
    Attributes should not be changed after initialization to avoid inconsistencies.
    Shapes can be modified with e.g. .reshape(), .expand() and .repeat().
    """

    def __init__(
        self,
        matrices: torch.Tensor = None,
        is_global: bool = False,
        det: torch.Tensor = None,
        inv: torch.Tensor = None,
        is_identity: bool = False,
        shape=None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """
        There are 2 ways to initialize an LFrames object:
        - From matrices: Set matrices and optionally is_global, det, inv
        - As identity: Set is_identity=True, shape, device, dtype

        Parameters
        ----------
            matrices: torch.tensor of shape (*dims, 4, 4)
                Transformation matrices
            is_global: bool
                Whether lframes are the same for all particles in the point cloud
            inv: torch.Tensor of shape (*dims, 4, 4)
                Optional cached inverse
            det: torch.Tensor of shape (*dims)
                Optional cached determinant
            is_identity: bool
                Sets matrices to diagonal
            shape: List[int]
                Specifies matrices.shape[:-2] if is_identity
            device: str
                Specifies device if is_identity
            dtype: torch.dtype
                Specifies dtype if is_identity
        """
        # straight-forward initialization
        self.is_identity = is_identity
        if is_identity:
            assert device and dtype
            if matrices is None:
                assert shape is not None
            else:
                shape = matrices.shape[:-2]

            self.matrices = lorentz_eye(shape, device=device, dtype=dtype)
            self.is_global = True
            self.det = None
            self.inv = None
        else:
            assert matrices is not None
            assert matrices.shape[-2:] == (
                4,
                4,
            ), f"matrices must be of shape (..., 4, 4), but found {matrices.shape[-2:]} instead"

            self.matrices = matrices
            self.is_global = is_global
            self.det = det
            self.inv = inv

        # cache expensive properties
        self.metric = lorentz_metric(
            self.shape[:-2], device=self.device, dtype=self.dtype
        )
        if self.det is None:
            if self.is_identity:
                self.det = torch.ones(
                    self.shape[:-2], dtype=self.dtype, device=self.device
                )
            else:
                self.det = torch.linalg.det(self.matrices)
        if self.inv is None:
            if self.is_identity:
                self.inv = self.matrices
            else:
                self.inv = self.metric @ self.matrices.transpose(-1, -2) @ self.metric

    def __repr__(self):
        return repr(self.matrices)

    def reshape(self, *shape):
        assert shape[-2:] == (4, 4)
        return LFrames(
            matrices=self.matrices.reshape(*shape),
            is_global=self.is_global,
            inv=self.inv.reshape(*shape),
            det=self.det.reshape(*shape[:-2]),
        )

    def expand(self, *shape):
        assert shape[-2:] == (4, 4)
        return LFrames(
            matrices=self.matrices.expand(*shape),
            is_global=self.is_global,
            inv=self.inv.expand(*shape),
            det=self.det.expand(*shape[:-2]),
        )

    def repeat(self, *shape):
        assert shape[-2:] == (1, 1)
        return LFrames(
            matrices=self.matrices.repeat(*shape),
            is_global=self.is_global,
            inv=self.inv.repeat(*shape),
            det=self.det.repeat(*shape[:-2]),
        )

    def to(self, dtype=None, device=None):
        self.matrices = self.matrices.to(device=device, dtype=dtype)
        self.inv = self.inv.to(device=device, dtype=dtype)
        self.det = self.det.to(device=device, dtype=dtype)

    @property
    def device(self):
        return self.matrices.device

    @property
    def dtype(self):
        return self.matrices.dtype

    @property
    def shape(self):
        return self.matrices.shape


class InverseLFrames(LFrames):
    """Inverse of a collection of transformations."""

    def __init__(self, lframes: LFrames) -> None:
        super().__init__(
            matrices=lframes.inv,
            is_global=lframes.is_global,
            inv=lframes.matrices,
            det=lframes.det,
            is_identity=lframes.is_identity,
            device=lframes.device,
            dtype=lframes.dtype,
            shape=lframes.shape,
        )


class IndexSelectLFrames(LFrames):
    """Index-controlled selection of transformation matrices from an LFrames object."""

    def __init__(self, lframes: LFrames, indices: torch.Tensor):
        super().__init__(
            matrices=lframes.matrices.index_select(0, indices),
            is_global=lframes.is_global,
            inv=lframes.inv.index_select(0, indices),
            det=lframes.det.index_select(0, indices),
            is_identity=lframes.is_identity,
            device=lframes.device,
            dtype=lframes.dtype,
            shape=lframes.shape,
        )


class ChangeOfLFrames(LFrames):
    """
    Change of frames between two LFrames objects
    Formally, for L_start and L_end we have
    L_change = L_end * L_start^-1

    WARNING: This function does not mix the lframes of different point clouds
    It is used in TFMessagePassing where this mixing is performed using the edge_index
    """

    def __init__(self, lframes_start: LFrames, lframes_end: LFrames) -> None:
        assert lframes_start.shape == lframes_end.shape
        if lframes_start.is_global:
            super().__init__(
                is_identity=True,
                shape=lframes_start.shape,
                device=lframes_start.device,
                dtype=lframes_start.dtype,
            )
        else:
            super().__init__(
                matrices=lframes_end.matrices @ lframes_start.inv,
                is_global=False,
                inv=lframes_start.matrices @ lframes_end.inv,
                det=lframes_start.det * lframes_end.det,
            )


class LowerIndices(LFrames):
    """
    LFrames with lower indices
    Used in InvariantParticleAttention to lower the key indices
    """

    def __init__(self, lframes):
        super().__init__(
            matrices=lframes.metric @ lframes.matrices,
            inv=lframes.inv @ lframes.metric,
            det=-lframes.det,
            is_global=lframes.is_global,
            is_identity=lframes.is_identity,
            device=lframes.device,
            dtype=lframes.dtype,
            shape=lframes.shape,
        )
