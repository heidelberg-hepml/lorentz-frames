import torch


def quaternions_to_matrix(quats, eps=1e-6):
    """Convert quaternions to rotation matrices.

    based on https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
    Args:
        quats: [B, 4] quaternions
        eps: a small number to prevent division by zero
    Returns:
        [B, 3, 3] rotation matrices
    """
    quat_norm = torch.linalg.norm(quats, dim=-1)
    zero_mask = quat_norm < eps
    q = torch.zeros_like(quats)
    q[~zero_mask] = quats[~zero_mask] / quat_norm[~zero_mask, None]
    q[zero_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quats.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = quats.shape[0]
    m = torch.zeros(B, 3, 3, device=quats.device)
    x2 = x**2
    y2 = y**2
    z2 = z**2
    xz = x * z
    xy = x * y
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    m[:, 0, 0] = 1 - 2 * y2 - 2 * z2
    m[:, 0, 1] = 2 * xy - 2 * wz
    m[:, 0, 2] = 2 * xz + 2 * wy
    m[:, 1, 0] = 2 * xy + 2 * wz
    m[:, 1, 1] = 1 - 2 * x2 - 2 * z2
    m[:, 1, 2] = 2 * yz - 2 * wx
    m[:, 2, 0] = 2 * xz - 2 * wy
    m[:, 2, 1] = 2 * yz + 2 * wx
    m[:, 2, 2] = 1 - 2 * x2 - 2 * y2
    return m


def matrix_to_quaternions(m):
    """Convert rotation matrices to quaternions.

    based on https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Args:
        m: [B, 3, 3] rotation matrices
    Returns:
        [B, 4] quaternions
    """
    B = m.shape[0]
    q = torch.zeros(B, 4, device=m.device)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    positive_trace = t > 0

    # calculation for positive trace
    r = torch.sqrt(1 + t[positive_trace])
    s = 0.5 / r
    q[positive_trace, 0] = 0.5 * r
    q[positive_trace, 1] = s * (m[positive_trace, 2, 1] - m[positive_trace, 1, 2])
    q[positive_trace, 2] = s * (m[positive_trace, 0, 2] - m[positive_trace, 2, 0])
    q[positive_trace, 3] = s * (m[positive_trace, 1, 0] - m[positive_trace, 0, 1])

    # calculation for negative trace, where m[:, 0, 0] is the largest diagonal element
    m00_mask = (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    m00_mask = m00_mask & ~positive_trace
    r = torch.sqrt(1 + m[m00_mask, 0, 0] - m[m00_mask, 1, 1] - m[m00_mask, 2, 2])
    s = 0.5 / r
    q[m00_mask, 0] = s * (m[m00_mask, 2, 1] - m[m00_mask, 1, 2])
    q[m00_mask, 1] = 0.5 * r
    q[m00_mask, 2] = s * (m[m00_mask, 0, 1] + m[m00_mask, 1, 0])
    q[m00_mask, 3] = s * (m[m00_mask, 0, 2] + m[m00_mask, 2, 0])

    # calculation for negative trace, where m[:, 1, 1] is the largest diagonal element
    m11_mask = (m[:, 1, 1] >= m[:, 2, 2]) & (m[:, 1, 1] >= m[:, 0, 0])
    m11_mask = m11_mask & ~positive_trace
    r = torch.sqrt(1 + m[m11_mask, 1, 1] - m[m11_mask, 0, 0] - m[m11_mask, 2, 2])
    s = 0.5 / r
    q[m11_mask, 0] = s * (m[m11_mask, 0, 2] - m[m11_mask, 2, 0])
    q[m11_mask, 1] = s * (m[m11_mask, 0, 1] + m[m11_mask, 1, 0])
    q[m11_mask, 2] = 0.5 * r
    q[m11_mask, 3] = s * (m[m11_mask, 1, 2] + m[m11_mask, 2, 1])

    # calculation for negative trace, where m[:, 2, 2] is the largest diagonal element
    m22_mask = (m[:, 2, 2] >= m[:, 0, 0]) & (m[:, 2, 2] >= m[:, 1, 1])
    m22_mask = m22_mask & ~positive_trace
    r = torch.sqrt(1 + m[m22_mask, 2, 2] - m[m22_mask, 0, 0] - m[m22_mask, 1, 1])
    s = 0.5 / r
    q[m22_mask, 0] = s * (m[m22_mask, 1, 0] - m[m22_mask, 0, 1])
    q[m22_mask, 1] = s * (m[m22_mask, 2, 0] + m[m22_mask, 0, 2])
    q[m22_mask, 2] = s * (m[m22_mask, 1, 2] + m[m22_mask, 2, 1])
    q[m22_mask, 3] = 0.5 * r

    return q


def quaternion_slerp(q1, q2, t, eps=1e-6, uniform_step_adaptation=False):
    """For small t one is close to q1, for large t one is close to q2."""
    assert q1.shape == q2.shape, "q1 and q2 must have the same shape"

    # make t a tensor of shape q1.shape[0] if it is a scalar
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=q1.device)
        t = t.expand(q1.shape[0])

    dot = (q1 * q2).sum(dim=-1)
    negative_dot = dot < 0
    q2[negative_dot] = -q2[negative_dot]
    dot[negative_dot] = -dot[negative_dot]

    angle_between_quats = torch.acos(torch.clamp(dot, min=-1, max=1))
    parallel_mask = angle_between_quats.abs() < eps

    tp = t[parallel_mask]
    t = t[~parallel_mask]

    # if the two quaternions are parallel, we can just interpolate linearly:
    interp_quat = torch.zeros_like(q1)
    interp_quat[parallel_mask] = (1 - tp)[:, None] * q1[parallel_mask] + tp[:, None] * q2[
        parallel_mask
    ]

    # apply slerp in the non-parallel case:
    angle = angle_between_quats[~parallel_mask]
    sin_angle = torch.sin(angle)

    if uniform_step_adaptation:
        # angle between rots is 2 * acos(abs(dot)):
        angle_between_rots = 2 * torch.where(angle < torch.pi / 2, angle, torch.pi - angle)

        # (angle_between_quats - torch.sin(angle_between_quats)) / angle_between_quats is the ratio of desired/ actual angle
        #  1 - ratio to make step larger if desired angle is smaller!
        # (1-t) is needed to cap it to the same max angle as with slerp
        t = 1 - (1 - t) * (angle_between_rots - torch.sin(angle_between_rots)) / angle_between_rots

    weight1 = torch.sin((1 - t) * angle) / sin_angle
    weight2 = torch.sin(t * angle) / sin_angle
    interp_quat[~parallel_mask] = (
        weight1[:, None] * q1[~parallel_mask] + weight2[:, None] * q2[~parallel_mask]
    )

    return interp_quat


def quaternion_product(q1, q2):
    """Compute the product of two quaternions.

    based on https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def sample_gaussian_rotation(ref_rot, width, num_samples=1):
    """ref_rot should be a 3x3 matrix.

    samples a rotation which deviates in angle distributed according to gaussian modulo pi.
    """
    ref_rot = ref_rot.reshape(-1, 3, 3)
    ref_quat = matrix_to_quaternions(ref_rot)

    gauss_angles = torch.abs(torch.randn(num_samples)) * width
    gauss_angles = gauss_angles % torch.pi

    gauss_quats = torch.zeros(num_samples, 4, device=ref_rot.device)
    gauss_quats[:, 0] = torch.cos(gauss_angles / 2)
    vector_component = torch.randn(num_samples, 3)  # uniform random unit vectors
    vector_component = vector_component / torch.linalg.norm(vector_component, dim=-1, keepdim=True)
    gauss_quats[:, 1:] = torch.sin(gauss_angles / 2)[:, None] * vector_component

    gauss_quats = quaternion_product(gauss_quats, ref_quat)
    return quaternions_to_matrix(gauss_quats)


def nlerp_interpolation(rot1, rot2, t):
    """For small t one is close to rot1, for large t one is close to rot2."""
    q1 = matrix_to_quaternions(rot1)
    q2 = matrix_to_quaternions(rot2)

    dot = (q1 * q2).sum(dim=-1)
    negative_dot = dot < 0
    q2[negative_dot] = -q2[negative_dot]  # note that q and -q represent the same rotation
    dot[negative_dot] = -dot[negative_dot]

    interp_quat = (1 - t) * q1 + t * q2
    interp_quat /= torch.linalg.norm(interp_quat, dim=-1, keepdim=True)
    interp_rot = quaternions_to_matrix(interp_quat)

    return interp_rot


def slerp_interpolation(rot1, rot2, t, uniform_step_adaptation=False):
    q1 = matrix_to_quaternions(rot1)
    q2 = matrix_to_quaternions(rot2)

    if torch.isnan(q1).any():
        print("q1 has nan")
    if torch.isnan(q2).any():
        print("q2 has nan")

    interp_quat = quaternion_slerp(q1, q2, t, uniform_step_adaptation=uniform_step_adaptation)

    if torch.isnan(interp_quat).any():
        print("interp_quat has nan")

    interp_rot = quaternions_to_matrix(interp_quat)

    if torch.isnan(interp_rot).any():
        print("interp_rot has nan")

    return interp_rot
