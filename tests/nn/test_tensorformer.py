import e3nn
import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.tensorformer import TensorFormer
from tensorframes.reps.tensorreps import TensorReps


def test_tensorformer_init_and_forward():
    tensor_reps = TensorReps("10x0n+5x1n+2x2n")
    num_heads = 4
    hidden_layers = [128, 64]
    hidden_value_dim = 64
    hidden_scalar_dim = 64
    edge_embedding_dim = 32

    model = TensorFormer(
        tensor_reps=tensor_reps,
        num_heads=num_heads,
        hidden_layers=hidden_layers,
        hidden_value_dim=hidden_value_dim,
        hidden_scalar_dim=hidden_scalar_dim,
        edge_embedding_dim=edge_embedding_dim,
    )

    # create test data
    x = torch.randn(10, tensor_reps.dim)
    pos = torch.randn(10, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    edge_embedding = torch.randn(10, edge_embedding_dim)

    lframes_mat = e3nn.o3.rand_matrix(10)
    lframes = LFrames(lframes_mat)

    # forward pass
    out = model(x, lframes, edge_index, pos, edge_embedding)
