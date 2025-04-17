import torch
from torch import nn

from experiments.eventgen.cfm import EventCFM
from tensorframes.lframes.lframes import InverseLFrames
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform


class CFMWrapper(EventCFM):
    def __init__(
        self,
        lframesnet,
        cfm,
        odeint,
        n_particles,
        beam_reference="timelike",
        two_beams=True,
        add_time_reference=True,
        scalar_dims=[0, 3],
    ):
        super().__init__(
            cfm,
            odeint,
        )
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

        self.register_buffer("particle_type", torch.arange(n_particles))
        self.scalar_dims = scalar_dims

        self.beam_reference = beam_reference
        self.two_beams = two_beams
        self.add_time_reference = add_time_reference

    def preprocess_velocity(self, x, t):
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        particle_type = self.encode_particle_type(x.shape)

        # TODO: include spurions (as extra particles) for lframesnet input

        fm = self.coordinates.x_to_fourmomenta(x)
        scalars = torch.cat([t_embedding, particle_type], dim=-1)
        lframes, tracker = self.lframesnet(
            fm, scalars=scalars, ptr=None, return_tracker=True
        )

        # move everything to self.input_dtype
        fm_local = self.trafo_fourmomenta(fm, lframes)
        x_local = self.coordinates.fourmomenta_to_x(fm_local)
        x_local = x_local.to(self.input_dtype)
        lframes.to(self.input_dtype)

        return (
            x_local,
            t_embedding,
            particle_type,
            lframes,
            tracker,
        )

    def postprocess_velocity(self, v_mixed_local, x, lframes):
        v_fm_local, v_s_local = v_mixed_local[..., 0:4], v_mixed_local[..., 4:]

        v_fm = self.trafo_fourmomenta(v_fm_local, InverseLFrames(lframes))
        fm = self.coordinates.x_to_fourmomenta(x)

        v_x, _ = self.coordinates.velocity_fourmomenta_to_x(v_fm, fm)
        v_x[..., self.scalar_dims] = v_s_local
        return v_x

    def encode_particle_type(self, shape):
        particle_type = torch.nn.functional.one_hot(
            self.particle_type, num_classes=self.particle_type.max() + 1
        )
        particle_type = particle_type.unsqueeze(0).repeat(shape[0], 1, 1)
        return particle_type


class TransformerCFM(CFMWrapper):
    """
    Baseline Transformer velocity network
    """

    def __init__(
        self,
        net,
        **kwargs,
    ):
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
        v = self.postprocess_velocity(v_local, x, lframes)
        return v  # and tracker
