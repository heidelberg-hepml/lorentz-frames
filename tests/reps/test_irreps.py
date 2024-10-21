import torch
from e3nn import o3

from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps.irreps import Irrep, Irreps, IrrepsTransform


def test_irreps():
    rep_1 = "5x0+3x1+3x2p+5x3"
    rep_2 = "5x0"

    irreps_1 = Irreps(rep_1)
    irreps_2 = Irreps(rep_2)
    irreps_3 = Irreps(
        [(5, Irrep(0, 1)), (3, Irrep(0, 1)), (2, Irrep(0, 1)), (1, Irrep(0, 1))]
    )

    test = irreps_1 + irreps_2 + irreps_3

    random_rot = o3.rand_matrix(10)

    coeffs = torch.randn(10, irreps_1.dim)
    irreps_transform = IrrepsTransform(irreps_1)
    lframes = LFrames(random_rot)

    random_rot = o3.rand_matrix(20)
    flip_mask = torch.randint(0, 2, (20,), dtype=torch.bool)
    random_rot[flip_mask] *= -1
    basis_change = ChangeOfLFrames(LFrames(random_rot[:10]), LFrames(random_rot[10:]))

    # test that 0n transforms correctly:
    irrep = Irreps("5x0")
    coeffs = torch.randn(10, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(transformed_coeffs, coeffs)

    # test that 0p transforms correctly:
    irrep = Irreps("5x0p")
    coeffs = torch.randn(10, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs, coeffs * basis_change.det[:, None], atol=1e-7
    )

    # test that 1n transforms correctly:
    irrep = Irreps("5x1")
    coeffs = torch.randn(10, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(10, 5, 3),
        torch.matmul(coeffs.reshape(10, 5, 3), basis_change.matrices.transpose(-1, -2)),
        atol=1e-7,
    )

    # test that 1p transforms correctly:
    irrep = Irreps("5x1p")
    coeffs = torch.randn(10, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(10, 5, 3),
        torch.matmul(coeffs.reshape(10, 5, 3), basis_change.matrices.transpose(-1, -2))
        * basis_change.det[:, None, None],
        atol=1e-7,
    )
