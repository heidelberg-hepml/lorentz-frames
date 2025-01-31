import torch


def orthogonalize_cross_o3(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 2

    def normalize(v):
        norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        return v / (norm + eps)

    vecs = [normalize(v) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = torch.cross(*orthogonal_vecs, *vecs[i:], dim=-1)
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next))

    return orthogonal_vecs
