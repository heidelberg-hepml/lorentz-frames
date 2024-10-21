import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.tensormace import TensorMACE
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


def test_tensormace_init_and_forward():
    in_tensor_reps = TensorReps("10x0n+5x1n+2x2n")
    out_tensor_reps = Irreps("10x0n+5x1n+2x2n")
    edge_emb_dim = 32
    hidden_dim = 64
    order = 3
    dropout = 0.1

    model = TensorMACE(
        in_tensor_reps, out_tensor_reps, edge_emb_dim, hidden_dim, order, dropout
    )

    x = torch.randn(10, in_tensor_reps.dim)
    # create a big edge_index
    edge_index = torch.randint(0, 10, (2, 100))
    edge_embedding = torch.randn(100, edge_emb_dim)
    import e3nn

    lframes = LFrames(e3nn.o3.rand_matrix(10))

    out = model(x, edge_index, edge_embedding, lframes)
