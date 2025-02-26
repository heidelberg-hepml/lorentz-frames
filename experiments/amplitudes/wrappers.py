import torch
from torch import nn
from functools import partial
from torch_geometric.nn.aggr import MeanAggregation

from experiments.amplitudes.preprocessing import preprocess_momentum

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.lorentz import lorentz_squarednorm


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

        particle_type = self.encode_particle_type(fourmomenta.shape[0]).to(
            features_local.dtype
        )
        return features_local, fourmomenta_local, particle_type, lframes, tracker

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
        features_local, _, _, _, tracker = super().forward(fourmomenta_global)
        features = features_local.view(features_local.shape[0], -1)

        amp = self.net(features)
        return amp, tracker


class TransformerWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    def forward(self, fourmomenta_global):
        features_local, _, particle_type, lframes, tracker = super().forward(
            fourmomenta_global
        )
        features = torch.cat([features_local, particle_type], dim=-1)

        output = self.net(features, lframes)
        amp = output.mean(dim=-2)
        return amp, tracker


class GraphNetWrapper(AmplitudeWrapper):
    def __init__(self, net, include_edges, include_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.aggregator = MeanAggregation()
        self.include_edges = include_edges
        self.include_nodes = include_nodes

        if self.include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.zeros(0))
            self.register_buffer("edge_std", torch.ones(1))

    def forward(self, fourmomenta_global):
        (
            features_local,
            fourmomenta_local,
            particle_type,
            lframes,
            tracker,
        ) = super().forward(fourmomenta_global)

        # select features
        node_attr = particle_type
        if self.include_nodes:
            node_attr = torch.cat([node_attr, features_local], dim=-1)
        edge_index, batch = build_edge_index(node_attr)
        node_attr = node_attr.view(-1, node_attr.shape[-1])
        lframes = lframes.reshape(-1, 4, 4)
        if self.include_edges:
            fourmomenta = fourmomenta_local.reshape(-1, 4)
            edge_attr = self.get_edge_attr(fourmomenta, edge_index)
        else:
            edge_attr = None

        output = self.net(
            node_attr,
            lframes,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )
        amp = self.aggregator(output, index=batch)
        return amp, tracker

    def get_edge_attr(self, fourmomenta, edge_index):
        mij2 = lorentz_squarednorm(
            fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
        )
        edge_attr = mij2.clamp(min=1e-10).log()
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean()
            self.edge_std = edge_attr.std().clamp(min=1e-5)
            self.edge_inited.fill_(True)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


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
