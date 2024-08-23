import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


def test_edge_conv_layer():
    # Test the EdgeConv layer
    # First test the EdgeConv layer with tensor_reps

    from e3nn.o3 import rand_matrix

    # Test the EdgeConv layer
    # First test the EdgeConv layer with TensorReps
    # Define the input and output representations
    in_reps = TensorReps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the EdgeConv layer
    mlp_conv = EdgeConv(
        in_reps=in_reps, hidden_channels=[32, 32], out_channels=16, concatenate_edge_vec=True
    )

    # create some dummy data
    edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)
    x = torch.randn(10, in_reps.dim)
    pos = torch.randn(10, 3)
    batch = torch.zeros(10, dtype=torch.long)
    lframes = LFrames(rand_matrix(10))
    feature_transform = in_reps.get_transform_class()

    # Perform the forward pass
    x_local = feature_transform(coeffs=x.clone(), basis_change=lframes)
    output = mlp_conv(x=x_local, pos=pos, edge_index=edge_index, lframes=lframes, batch=batch)

    # add small invariance test:
    global_rot = rand_matrix(1).repeat(10, 1, 1)
    x_trafo = feature_transform(coeffs=x, basis_change=LFrames(global_rot))
    pos_trafo = []
    lframes_trafo = []
    for i in range(10):
        lframes_trafo.append(lframes.matrices[i] @ global_rot[0].T)
        pos_trafo.append(pos[i] @ global_rot[0].T)
    lframes_trafo = LFrames(torch.stack(lframes_trafo))
    pos_trafo = torch.stack(pos_trafo)
    x_trafo_local = feature_transform(coeffs=x_trafo.clone(), basis_change=lframes_trafo)
    assert torch.allclose(x_local, x_trafo_local, atol=1e-6)
    output_trafo = mlp_conv(
        x=x_trafo_local, pos=pos_trafo, edge_index=edge_index, lframes=lframes_trafo, batch=batch
    )

    assert torch.allclose(output, output_trafo, atol=1e-6)

    # test if layer is differentiable:
    x_local = feature_transform(coeffs=x.clone(), basis_change=lframes)
    x_local.requires_grad = True
    output = mlp_conv(x=x_local, pos=pos, edge_index=edge_index, lframes=lframes, batch=batch)
    output.sum().backward()

    # get gradients on x:
    assert x_local.grad is not None

    # Second test the EdgeConv layer with Irreps

    # Define the input and output representations
    in_reps = Irreps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the EdgeConv layer
    mlp_conv = EdgeConv(
        in_reps=in_reps, hidden_channels=[32, 32], out_channels=16, concatenate_edge_vec=True
    )

    # create some dummy data
    edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)
    x = torch.randn(10, in_reps.dim)
    pos = torch.randn(10, 3)
    batch = torch.zeros(10, dtype=torch.long)
    lframes = LFrames(rand_matrix(10))
    feature_transform = in_reps.get_transform_class()

    # Perform the forward pass
    x_local = feature_transform(coeffs=x.clone(), basis_change=lframes)
    output = mlp_conv(x=x_local, pos=pos, edge_index=edge_index, lframes=lframes, batch=batch)

    # add small invariance test:
    global_rot = rand_matrix(1).repeat(10, 1, 1)
    x_trafo = feature_transform(coeffs=x, basis_change=LFrames(global_rot))
    pos_trafo = []
    lframes_trafo = []
    for i in range(10):
        lframes_trafo.append(lframes.matrices[i] @ global_rot[0].T)
        pos_trafo.append(pos[i] @ global_rot[0].T)
    lframes_trafo = LFrames(torch.stack(lframes_trafo))
    pos_trafo = torch.stack(pos_trafo)
    x_trafo_local = feature_transform(coeffs=x_trafo.clone(), basis_change=lframes_trafo)
    assert torch.allclose(x_local, x_trafo_local, atol=1e-6)
    output_trafo = mlp_conv(
        x=x_trafo_local, pos=pos_trafo, edge_index=edge_index, lframes=lframes_trafo, batch=batch
    )

    assert torch.allclose(output, output_trafo, atol=1e-6)

    # test if layer is differentiable:
    x_local = feature_transform(coeffs=x.clone(), basis_change=lframes)
    x_local.requires_grad = True
    output = mlp_conv(x=x_local, pos=pos, edge_index=edge_index, lframes=lframes, batch=batch)
    output.sum().backward()

    # get gradients on x:
    assert x_local.grad is not None
