import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from experiments.amplitudes.utils import standardize_momentum
from experiments.baselines.gatr.interface import embed_vector, extract_scalar

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.utils import build_edge_index_fully_connected


class AmplitudeWrapper(nn.Module):
    def __init__(
        self,
        particle_type,
        lframesnet,
    ):
        super().__init__()
        self.lframesnet = lframesnet

        self.register_buffer("particle_type", torch.tensor(particle_type))
        self.register_buffer("mom_mean", torch.tensor(0.0))
        self.register_buffer("mom_std", torch.tensor(1.0))

        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_standardization(self, fourmomenta):
        _, self.mom_mean, self.mom_std = standardize_momentum(fourmomenta)

    def forward(self, fourmomenta):
        scalars = torch.zeros_like(fourmomenta[..., []])
        lframes, tracker = self.lframesnet(
            fourmomenta, scalars, ptr=None, return_tracker=True
        )

        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, lframes)
        features_local, _, _ = standardize_momentum(
            fourmomenta_local, self.mom_mean, self.mom_std
        )
        particle_type = self.encode_particle_type(fourmomenta_local.shape[0]).to(
            dtype=fourmomenta.dtype, device=fourmomenta.device
        )
        return (
            features_local,
            fourmomenta_local,
            particle_type,
            lframes,
            tracker,
        )

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
        (
            features_local,
            _,
            particle_type,
            lframes,
            tracker,
        ) = super().forward(fourmomenta_global)
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
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def init_standardization(self, fourmomenta):
        super().init_standardization(fourmomenta)

        if self.include_edges:
            # edge feature standardization parameters
            edge_index, _ = build_edge_index_fully_connected(fourmomenta)
            fourmomenta = fourmomenta.reshape(-1, 4)
            mij2 = lorentz_squarednorm(
                fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
            )
            edge_attr = mij2.clamp(min=1e-10).log()
            self.edge_mean = edge_attr.mean()
            self.edge_std = edge_attr.std().clamp(min=1e-10)

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
        edge_index, batch = build_edge_index_fully_connected(node_attr)
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
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class GATrWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    def forward(self, fourmomenta_global):
        (
            _,
            fourmomenta_local,
            particle_type,
            _,
            tracker,
        ) = super().forward(fourmomenta_global)

        # prepare multivectors and scalars
        multivectors = embed_vector(fourmomenta_local.unsqueeze(-2))
        scalars = particle_type

        # call network
        out_mv, _ = self.net(multivectors, scalars)
        out_mv = extract_scalar(out_mv)[..., 0]  # extract 0th channel of scalar

        # mean aggregation
        amp = out_mv.mean(dim=-2)
        return amp, tracker


class DSIWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    def forward(self, fourmomenta_global):
        (
            _,
            fourmomenta_local,
            particle_type,
            _,
            tracker,
        ) = super().forward(fourmomenta_global)

        amp = self.net(fourmomenta_local)
        return amp, tracker
