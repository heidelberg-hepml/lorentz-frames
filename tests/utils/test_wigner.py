import torch
from e3nn.o3 import matrix_to_angles, rand_matrix

from tensorframes.utils.wigner import euler_angles_yxy, wigner_D_from_matrix


def test_wigner():
    # test angle implementation:
    R = rand_matrix(1000)

    # check that is works even for noise:
    R += 1e-7 * torch.randn_like(R)

    angles = matrix_to_angles(R)
    my_angles = euler_angles_yxy(R)

    for i, angle in enumerate(angles):
        my_angle = my_angles[:, i]
        assert torch.allclose(angle, my_angle, atol=1e-5)

    # test angle special cases:
    special_Rs = [torch.eye(3)]
    for i in range(3):
        R = -torch.eye(3)
        R[i, i] = 1
        special_Rs.append(R)
    special_Rs = torch.stack(special_Rs)

    angles = matrix_to_angles(special_Rs)
    my_angles = euler_angles_yxy(special_Rs, handle_special_cases=True)

    for i, angle in enumerate(angles):
        my_angle = my_angles[:, i]
        assert torch.allclose(angle, my_angle, atol=1e-5)

    # test that wigner for the identity matrix is the identity matrix
    for l in range(4):
        assert torch.allclose(
            wigner_D_from_matrix(l, torch.eye(3), handle_special_cases=True),
            torch.eye(2 * l + 1),
            atol=1e-6,
        )

    # test that wigner is 1 for l = 0:
    assert torch.allclose(wigner_D_from_matrix(0, torch.rand(10, 3, 3)), torch.ones(10, 1, 1))

    # test that wigner for l=1 is the same as the rotation matrix:

    matrix = rand_matrix(10)
    assert torch.allclose(wigner_D_from_matrix(1, matrix), matrix)

    # test that wigner is a representation:
    matrix1 = rand_matrix(10)
    matrix2 = rand_matrix(10)
    assert torch.allclose(
        wigner_D_from_matrix(1, torch.bmm(matrix1, matrix2)),
        torch.bmm(wigner_D_from_matrix(1, matrix1), wigner_D_from_matrix(1, matrix2)),
        atol=1e-6,
    )

    # test that wigner is an orthogonal matrix:
    wigner = wigner_D_from_matrix(3, rand_matrix(10))
    assert torch.allclose(wigner @ wigner.transpose(-1, -2), torch.eye(2 * 3 + 1), atol=1e-6)

    # test that wigner is an orthogonal representation:
    matrix = rand_matrix(10)
    assert torch.allclose(
        wigner_D_from_matrix(4, matrix).transpose(-1, -2),
        wigner_D_from_matrix(4, matrix.transpose(-1, -2)),
        atol=1e-6,
    )
