import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from experiments.amplitudes.utils import standardize_momentum
from lgatr import embed_vector, extract_scalar

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import build_edge_index_fully_connected, get_edge_attr
from lloca.lframes.nonequi_lframes import IdentityLFrames


class AmplitudeWrapper(nn.Module):
    def __init__(
        self,
        particle_type,
        lframesnet,
        network_float64=False,
    ):
        super().__init__()
        self.lframesnet = lframesnet
        self.network_dtype = torch.float64 if network_float64 else torch.float32

        self.register_buffer("particle_type", torch.tensor(particle_type))
        self.register_buffer("mom_mean", torch.tensor(0.0))
        self.register_buffer("mom_std", torch.tensor(1.0))

        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_standardization(self, fourmomenta):
        _, self.mom_mean, self.mom_std = standardize_momentum(fourmomenta)

    def forward(self, fourmomenta):
        particle_type = self.encode_particle_type(fourmomenta.shape[0]).to(
            dtype=self.network_dtype, device=fourmomenta.device
        )
        lframes, tracker = self.lframesnet(
            fourmomenta, scalars=particle_type, ptr=None, return_tracker=True
        )

        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, lframes)
        features_local, _, _ = standardize_momentum(
            fourmomenta_local, self.mom_mean, self.mom_std
        )

        # move everything to less safe dtype
        features_local = features_local.to(self.network_dtype)
        lframes.to(self.network_dtype)
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
        features_local, _, _, lframes, tracker = super().forward(fourmomenta_global)
        features = features_local.reshape(features_local.shape[0], -1)

        amp = self.net(features)
        return amp, tracker, lframes


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
        return amp, tracker, lframes


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
            edge_attr = get_edge_attr(fourmomenta, edge_index)
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
            edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(
                self.network_dtype
            )
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
        return amp, tracker, lframes

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class LGATrWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        assert isinstance(self.lframesnet, IdentityLFrames)

    def forward(self, fourmomenta_global):
        (
            _,
            fourmomenta_local,
            particle_type,
            lframes,
            tracker,
        ) = super().forward(fourmomenta_global)

        # prepare multivectors and scalars
        multivectors = embed_vector(
            fourmomenta_local.unsqueeze(-2).to(self.network_dtype)
        )
        scalars = particle_type

        # call network
        out_mv, _ = self.net(multivectors, scalars)
        out_mv = extract_scalar(out_mv)[..., 0]  # extract 0th channel of scalar

        # mean aggregation
        amp = out_mv.mean(dim=-2)
        return amp, tracker, lframes


class DSIWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        assert isinstance(self.lframesnet, IdentityLFrames)

    def forward(self, fourmomenta_global):
        (
            _,
            fourmomenta_local,
            _,
            lframes,
            tracker,
        ) = super().forward(fourmomenta_global)

        amp = self.net(fourmomenta_local.to(self.network_dtype))
        return amp, tracker, lframes


class PELICANWrapper(AmplitudeWrapper):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        assert isinstance(self.lframesnet, IdentityLFrames)

        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

    def init_standardization(self, fourmomenta):
        super().init_standardization(fourmomenta)

        # edge feature standardization parameters
        edge_index, _ = build_edge_index_fully_connected(
            fourmomenta, remove_self_loops=False
        )
        fourmomenta = fourmomenta.reshape(-1, 4)
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        self.edge_mean = edge_attr.mean()
        self.edge_std = edge_attr.std().clamp(min=1e-10)

    def forward(self, fourmomenta_global):
        (
            _,
            fourmomenta_local,
            particle_type,
            lframes,
            tracker,
        ) = super().forward(fourmomenta_global)
        num_graphs = fourmomenta_local.shape[0]

        edge_index, batch = build_edge_index_fully_connected(
            particle_type, remove_self_loops=False
        )
        fourmomenta = fourmomenta_local.reshape(-1, 4)
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(self.network_dtype)
        node_attr = particle_type.reshape(-1, particle_type.shape[-1])

        out = self.net(
            in_rank2=edge_attr,
            in_rank1=node_attr,
            edge_index=edge_index,
            batch=batch,
            num_graphs=num_graphs,
        )
        return out, tracker, lframes

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)
