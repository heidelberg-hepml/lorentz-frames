import torch
from torch import nn
from functools import partial
from torch_geometric.nn.aggr import MeanAggregation

from experiments.amplitudes.preprocessing import preprocess_momentum

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.lframes.lframes import LFrames


class AmplitudeWrapper(nn.Module):
    def __init__(
        self,
        particle_type,
        lframesnet,
    ):
        super().__init__()
        self.register_buffer("particle_type", torch.tensor(particle_type))
        self.register_buffer("mom_mean", torch.tensor(0.0))
        self.register_buffer("mom_std", torch.tensor(1.0))
        if isinstance(lframesnet, partial):
            self.lframesnet = lframesnet(in_nodes=0)
        else:
            self.lframesnet = lframesnet

        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_momentum_preprocessing(self, mean, std):
        self.mom_mean = mean
        self.mom_std = std

    def forward(self, fourmomenta):
        if self.lframesnet.is_global:
            lframes, tracker = self.lframesnet(fourmomenta, return_tracker=True)
        else:
            shape = fourmomenta.shape
            edge_index, batch = build_edge_index(fourmomenta, remove_self_loops=True)
            fourmomenta_flat = fourmomenta.reshape(-1, 4)
            scalars = torch.zeros_like(fourmomenta_flat[:, []])
            lframes, tracker = self.lframesnet(
                fourmomenta_flat,
                scalars,
                edge_index=edge_index,
                batch=batch,
                return_tracker=True,
            )
            lframes = lframes.reshape(*shape[:-1], 4, 4)

        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, lframes)

        features_local, _, _ = preprocess_momentum(
            fourmomenta_local, self.mom_mean, self.mom_std
        )
        features_local = torch.arcsinh(features_local)  # suppress tails

        particle_type = self.encode_particle_type(fourmomenta.shape[0])
        return features_local, particle_type, lframes, tracker

    def encode_particle_type(self, batchsize):
        particle_type = torch.nn.functional.one_hot(
            self.particle_type, num_classes=self.particle_type.max() + 1
        )
        particle_type = particle_type.unsqueeze(0).repeat(batchsize, 1, 1)
        return particle_type


class MLPWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    def forward(self, fourmomenta_global):
        features_local, _, _, tracker = super().forward(fourmomenta_global)
        features = features_local.view(features_local.shape[0], -1)

        amp = self.net(features)
        return amp, tracker


class TransformerWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    def forward(self, fourmomenta_global):
        features_local, particle_type, lframes, tracker = super().forward(
            fourmomenta_global
        )
        features = torch.cat([features_local, particle_type], dim=-1)

        output = self.net(features, lframes)
        amp = output.mean(dim=-2)
        return amp, tracker


class GraphNetWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.aggregator = MeanAggregation()

    def forward(self, fourmomenta_global):
        features_local, particle_type, lframes, tracker = super().forward(
            fourmomenta_global
        )
        features = torch.cat([features_local, particle_type], dim=-1)
        edge_index, batch = build_edge_index(features)

        # flatten over (batch_dim, seq_len)
        features_flat = features.view(-1, features.shape[-1])
        mat = lframes.matrices.reshape(-1, 4, 4)
        lframes = LFrames(mat)

        output = self.net(features_flat, lframes, edge_index=edge_index, batch=batch)
        amp = self.aggregator(output, index=batch)
        return amp, tracker


def build_edge_index(features_ref, remove_self_loops=False):
    batch_size, seq_len, _ = features_ref.shape
    device = features_ref.device

    nodes = torch.arange(seq_len, device=device)
    row, col = torch.meshgrid(nodes, nodes, indexing="ij")

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]
    edge_index_single = torch.stack([row.flatten(), col.flatten()], dim=0)

    edge_index_global = []
    for i in range(batch_size):
        offset = i * seq_len
        edge_index_global.append(edge_index_single + offset)
    edge_index_global = torch.cat(edge_index_global, dim=1)

    batch = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    return edge_index_global, batch
