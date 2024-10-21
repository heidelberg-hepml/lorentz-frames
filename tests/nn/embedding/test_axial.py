import torch
from e3nn.o3 import rand_matrix

from tensorframes.lframes import LFrames
from tensorframes.nn.embedding.axial import (
    AxisWiseBesselEmbedding,
    AxisWiseEmbeddingFromRadial,
    AxisWiseGaussianEmbedding,
)
from tensorframes.nn.embedding.radial import (
    BesselEmbedding,
    GaussianEmbedding,
    compute_edge_vec,
)


def test_axial():
    pos = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))
    lframes = rand_matrix(10)
    parity_mask = torch.randint(0, 2, (10,), dtype=torch.bool)
    lframes[parity_mask, :, 0] *= -1
    lframes = LFrames(matrices=lframes)

    edge_vec = compute_edge_vec(pos, edge_index, lframes)
    axis_gauss = AxisWiseGaussianEmbedding(
        num_gaussians=10, maximum_initial_range=1.0, minimum_initial_range=-1.0
    )
    assert (
        axis_gauss(edge_vec=edge_vec).shape
        == torch.Size((20, axis_gauss.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        axis_gauss(edge_vec=edge_vec),
        axis_gauss(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    axis_bessel = AxisWiseBesselEmbedding(num_frequencies=10)
    assert (
        axis_bessel(edge_vec=edge_vec).shape
        == torch.Size((20, axis_bessel.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        axis_bessel(edge_vec=edge_vec),
        axis_bessel(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    # # test axiswise from radial:
    axis_radial_gauss = AxisWiseEmbeddingFromRadial(
        normalize_edge_vec=True,
        axis_specific_radial=False,
        radial_embedding=GaussianEmbedding(
            num_gaussians=10, maximum_initial_range=1.0, minimum_initial_range=-1.0
        ),
    )
    assert (
        axis_radial_gauss(edge_vec=edge_vec).shape
        == torch.Size((20, axis_radial_gauss.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        axis_radial_gauss(edge_vec=edge_vec),
        axis_radial_gauss(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    # check if this yields the same result as above:
    assert torch.allclose(
        axis_gauss(edge_vec=edge_vec), axis_radial_gauss(edge_vec=edge_vec), atol=1e-7
    )

    # check the axis specific radial:
    specific_axis_radial_gauss = AxisWiseEmbeddingFromRadial(
        normalize_edge_vec=True,
        axis_specific_radial=True,
        radial_embedding=GaussianEmbedding(
            num_gaussians=10, maximum_initial_range=1.0, minimum_initial_range=-1.0
        ),
    )
    assert (
        specific_axis_radial_gauss(edge_vec=edge_vec).shape
        == torch.Size((20, specific_axis_radial_gauss.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        specific_axis_radial_gauss(edge_vec=edge_vec),
        specific_axis_radial_gauss(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    # check that without training the radial embeddings are the same:
    assert torch.allclose(
        specific_axis_radial_gauss(edge_vec=edge_vec),
        axis_gauss(edge_vec=edge_vec),
        atol=1e-7,
    )

    # test axiswise from radial using bessel:
    axis_radial_bessel = AxisWiseEmbeddingFromRadial(
        normalize_edge_vec=True,
        axis_specific_radial=False,
        radial_embedding=BesselEmbedding(num_frequencies=10, flip_negative=True),
    )
    assert (
        axis_radial_bessel(edge_vec=edge_vec).shape
        == torch.Size((20, axis_radial_bessel.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        axis_radial_bessel(edge_vec=edge_vec),
        axis_radial_bessel(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    # check that it is the same as the axis bessel:
    assert torch.allclose(
        axis_bessel(edge_vec=edge_vec), axis_radial_bessel(edge_vec=edge_vec), atol=1e-7
    )

    # check the axis specific radial:
    specific_axis_radial_bessel = AxisWiseEmbeddingFromRadial(
        normalize_edge_vec=True,
        axis_specific_radial=True,
        radial_embedding=BesselEmbedding(num_frequencies=10, flip_negative=True),
    )
    assert (
        specific_axis_radial_bessel(edge_vec=edge_vec).shape
        == torch.Size((20, axis_radial_bessel.out_dim))
        == torch.Size([20, 30])
    )
    assert torch.allclose(
        specific_axis_radial_bessel(edge_vec=edge_vec),
        specific_axis_radial_bessel(pos=pos, edge_index=edge_index, lframes=lframes),
        atol=1e-7,
    )

    # check that without training the radial embeddings are the same:
    assert torch.allclose(
        specific_axis_radial_bessel(edge_vec=edge_vec),
        axis_bessel(edge_vec=edge_vec),
        atol=1e-7,
    )

    # test flip negative:
    out1 = axis_radial_bessel(
        edge_vec=torch.tensor([[1.0, 2, 3], [-1, -2, -3], [4, 5, 6], [4, -5, 6]])
    )
    assert torch.allclose(out1[0], -out1[1])
    assert torch.allclose(out1[2][:10], out1[3][:10])
    assert torch.allclose(out1[2][10:20], -out1[3][10:20])
    assert torch.allclose(out1[2][20:], out1[3][20:])

    axis_radial_bessel.radial_modules.flip_negative = False
    out2 = axis_radial_bessel(
        edge_vec=torch.tensor([[1.0, 2, 3], [-1, -2, -3], [4, 5, 6], [4, -5, 6]])
    )
    assert torch.allclose(out2[0], out2[1])
    assert torch.allclose(out2[2], out2[3])
