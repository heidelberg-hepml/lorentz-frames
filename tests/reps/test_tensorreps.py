import torch
from tests.constants import TOLERANCES

from tensorframes.utils.transforms import rand_transform
from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps.tensorreps import TensorReps, TensorRepsTransform

# I quick-and-dirty-adapted this test from the original tensorframes repo
# this deserves some attention at some point


def test_tensorreps():
    rep_1 = "5x0n+3x1n+3x2n+5x3n"
    tensor_reps_1 = TensorReps(rep_1)

    random_rot = rand_transform([10])

    coeffs = torch.randn(10, tensor_reps_1.dim)
    tensor_reps_transform = TensorRepsTransform(tensor_reps_1)
    lframes = LFrames(random_rot)

    random_rot = rand_transform([20])
    flip_mask = torch.randint(0, 2, (20,), dtype=torch.bool)
    random_rot[flip_mask] *= -1
    basis_change = ChangeOfLFrames(LFrames(random_rot[:10]), LFrames(random_rot[10:]))

    # test that 0n transforms correctly:
    rep = TensorReps("5x0n")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    torch.testing.assert_close(transformed_coeffs, coeffs, **TOLERANCES)

    """
    # test that 0p transforms correctly:
    rep = TensorReps("5x0n")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)

    torch.testing.assert_close(
        transformed_coeffs, coeffs * basis_change.det[:, None], **TOLERANCES
    )
    """

    # test that 1n transforms correctly:
    rep = TensorReps("5x1n")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    torch.testing.assert_close(
        transformed_coeffs.reshape(10, 5, 4),
        torch.matmul(coeffs.reshape(10, 5, 4), basis_change.matrices.transpose(-1, -2)),
        **TOLERANCES,
    )

    """
    # test that 1p transforms correctly:
    rep = TensorReps("5x1p")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    torch.testing.assert_close(
        transformed_coeffs.reshape(10, 5, 4),
        torch.matmul(coeffs.reshape(10, 5, 4), basis_change.matrices.transpose(-1, -2))
        * basis_change.det[:, None, None],
        **TOLERANCES,
    )
    """

    # test that 2n transforms correctly:
    rep = TensorReps("5x2n")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    naive_trafo = torch.einsum(
        "kij, klm, kcjm -> kcil",
        basis_change.matrices,
        basis_change.matrices,
        coeffs.reshape(10, 5, 4, 4),
    )
    torch.testing.assert_close(
        transformed_coeffs.reshape(10, 5, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )

    """
    # test that 2p transforms correctly:
    rep = TensorReps("5x2p")
    coeffs = torch.randn(10, rep.dim)
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    naive_trafo = torch.einsum(
        "kij, klm, kcjm -> kcil",
        basis_change.matrices,
        basis_change.matrices,
        coeffs.reshape(10, 5, 4, 4),
    )
    """
    naive_trafo *= basis_change.det[:, None, None, None]
    torch.testing.assert_close(
        transformed_coeffs.reshape(10, 5, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )
