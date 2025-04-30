import torch
from torch import nn

from experiments.eventgen.cfm import EventCFM
from experiments.tagging.embedding import get_spurion
from tensorframes.lframes.lframes import LFrames, InverseLFrames
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.utils import build_edge_index_fully_connected


class CFMWrapper(EventCFM):
    def __init__(
        self,
        lframesnet,
        cfm,
        odeint,
        n_particles,
        spurions,
    ):
        super().__init__(
            cfm,
            odeint,
        )
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

        self.register_buffer("particle_type", torch.arange(n_particles))
        self.scalar_dims = spurions.scalar_dims

        self.spurions = get_spurion(
            beam_reference=spurions.beam_reference,
            add_time_reference=spurions.add_time_reference,
            two_beams=spurions.two_beams,
            dtype=torch.float64,
            device="cpu",
        )
        self.n_spurions = self.spurions.shape[0]

    def preprocess_velocity(self, x, t):
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        particle_type = self.encode_particle_type(x.shape)

        if self.lframesnet.is_identity:
            # shortcut
            # local frame = global frame -> nothing to do
            lframes, tracker = self.lframesnet(x, return_tracker=True)
            x_local = x

        else:
            # long route (also works for identity but is slower)
            fm = self.coordinates.x_to_fourmomenta(x)
            scalars = torch.cat([t_embedding, particle_type], dim=-1)

            spurions = self.spurions.to(x.device).unsqueeze(0).repeat(fm.shape[0], 1, 1)
            fm_withspurions = torch.cat((spurions, fm), dim=-2)
            scalars_zeros = torch.zeros(
                scalars.shape[0],
                self.n_spurions,
                scalars.shape[2],
                device=scalars.device,
                dtype=scalars.dtype,
            )
            scalars_withspurions = torch.cat((scalars_zeros, scalars), dim=-2)
            lframes_withspurions, tracker = self.lframesnet(
                fm_withspurions,
                scalars=scalars_withspurions,
                ptr=None,
                return_tracker=True,
            )
            lframes = LFrames(
                matrices=lframes_withspurions.matrices[:, self.n_spurions :],
                det=lframes_withspurions.det[:, self.n_spurions :],
                inv=lframes_withspurions.inv[:, self.n_spurions :],
                is_global=lframes_withspurions.is_global,
                is_identity=lframes_withspurions.is_identity,
            )

            fm_local = self.trafo_fourmomenta(fm, lframes)
            x_local = self.coordinates.fourmomenta_to_x(fm_local)

        tracker["lframes_absmax"] = (
            lframes.matrices.detach()
            .abs()
            .max(-1)[0]
            .max(-1)[0]
            .max(-1)[0]
            .mean()
            .cpu()
        )
        tracker["lframes_00"] = (
            lframes.matrices[..., 0, 0].detach().max(-1)[0].mean().cpu()
        )
        tracker["lframes_norm"] = (
            torch.linalg.norm(lframes.matrices.detach(), ord=2, dim=[-2, -1])
            .max(-1)[0]
            .mean()
            .cpu()
        )

        tracker["lframes_absmax_max"] = (
            lframes.matrices.detach().abs().max(-1)[0].max(-1)[0].max(-1)[0].max().cpu()
        )
        tracker["lframes_00_max"] = (
            lframes.matrices[..., 0, 0].detach().max(-1)[0].max().cpu()
        )
        tracker["lframes_norm_max"] = (
            torch.linalg.norm(lframes.matrices.detach(), ord=2, dim=[-2, -1])
            .max(-1)[0]
            .max()
            .cpu()
        )

        # move everything to self.input_dtype
        x_local = x_local.to(self.input_dtype)
        lframes.to(self.input_dtype)

        return (
            x_local,
            t_embedding,
            particle_type,
            lframes,
            tracker,
        )

    def postprocess_velocity(self, v_mixed_local, x, lframes, tracker):
        v_fm_local, v_s_local = v_mixed_local[..., 0:4], v_mixed_local[..., 4:]

        if self.lframesnet.is_identity:
            # shortcut
            # interpret network output as velocity in x-coordinates
            v_x = v_fm_local

        else:
            # long route (also works for identity but is slower)
            v_fm = self.trafo_fourmomenta(v_fm_local, InverseLFrames(lframes))
            fm = self.coordinates.x_to_fourmomenta(x)

            v_x, _ = self.coordinates.velocity_fourmomenta_to_x(v_fm, fm)
            v_x[..., self.scalar_dims] = v_s_local

            tracker["v_fm_local_absmax"] = (
                v_fm_local.detach().abs().max(-1)[0].max(-1)[0].mean().cpu()
            )
            tracker["v_fm_global_absmax"] = (
                v_fm.detach().abs().max(-1)[0].max(-1)[0].mean().cpu()
            )
            tracker["v_x_absmax"] = (
                v_x.detach().abs().max(-1)[0].max(-1)[0].mean().cpu()
            )

            tracker["v_fm_local_absmax_max"] = (
                v_fm_local.detach().abs().max(-1)[0].max(-1)[0].max().cpu()
            )
            tracker["v_fm_global_absmax_max"] = (
                v_fm.detach().abs().max(-1)[0].max(-1)[0].max().cpu()
            )
            tracker["v_x_absmax_max"] = (
                v_x.detach().abs().max(-1)[0].max(-1)[0].max().cpu()
            )

        v_x = v_x.to(torch.float64)
        return v_x, tracker

    def encode_particle_type(self, shape):
        particle_type = torch.nn.functional.one_hot(
            self.particle_type, num_classes=self.particle_type.max() + 1
        )
        particle_type = particle_type.unsqueeze(0).repeat(shape[0], 1, 1)
        return particle_type


class MLPCFM(CFMWrapper):
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            _,
            lframes,
            tracker,
        ) = super().preprocess_velocity(x, t)

        x_local = x_local
        fts = torch.cat(
            [
                x_local.reshape(*x_local.shape[:-2], -1),
                t_embedding[..., 0, :],
            ],
            dim=-1,
        )
        v_local = self.net(fts)
        v_local = v_local.reshape(*x_local.shape[:-1], -1)
        v, tracker = self.postprocess_velocity(v_local, x, lframes, tracker)
        return v, tracker


class TransformerCFM(CFMWrapper):
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            particle_type,
            lframes,
            tracker,
        ) = super().preprocess_velocity(x, t)

        fts = torch.cat([x_local, particle_type, t_embedding], dim=-1)
        v_local = self.net(fts, lframes)
        v, tracker = self.postprocess_velocity(v_local, x, lframes, tracker)
        return v, tracker


class GraphNetCFM(CFMWrapper):
    def __init__(self, net, include_edges, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.include_edges = include_edges
        assert not include_edges

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            particle_type,
            lframes,
            tracker,
        ) = super().preprocess_velocity(x, t)
        edge_index, batch = build_edge_index_fully_connected(x_local)
        fts = torch.cat([x_local, particle_type, t_embedding], dim=-1)

        fts_flat = fts.flatten(0, 1)
        lframes_flat = lframes.reshape(-1, *lframes.shape[2:])
        v_local_flat = self.net(
            fts_flat, lframes_flat, edge_index=edge_index, batch=batch
        )
        v_local = v_local_flat.reshape(*fts.shape[:-1], v_local_flat.shape[-1])

        v, tracker = self.postprocess_velocity(v_local, x, lframes, tracker)
        return v, tracker


class LGATrCFM(CFMWrapper):
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        assert self.lframesnet.is_identity

    def get_velocity(self, x, t):
        raise NotImplementedError
        (
            x_local,
            t_embedding,
            particle_type,
            lframes,
            tracker,
        ) = super().preprocess_velocity(x, t)

        fts = torch.cat([x_local, particle_type, t_embedding], dim=-1)
        v_local = self.net(fts, lframes)
        v = self.postprocess_velocity(v_local, x, lframes)
        return v, tracker
