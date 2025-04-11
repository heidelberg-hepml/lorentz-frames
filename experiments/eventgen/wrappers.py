import torch
import numpy as np

from experiments.eventgen.cfm import EventCFM
from experiments.eventgen.utils import get_type_token, get_process_token


class MLPCFM(EventCFM):
    """
    Baseline MLP velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net

    def get_velocity(self, x, t, ijet):
        t_embedding = self.t_embedding(t).squeeze()
        x = x.reshape(x.shape[0], -1)

        x = torch.cat([x, t_embedding], dim=-1)
        v = self.net(x)
        v = v.reshape(v.shape[0], v.shape[1] // 4, 4)
        return v


class TransformerCFM(EventCFM):
    """
    Baseline Transformer velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        type_token_channels,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.type_token_channels = type_token_channels

    def get_velocity(self, x, t, ijet):
        # note: flow matching happens directly in x space
        type_token = get_type_token(x, self.type_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x = torch.cat([x, type_token, t_embedding], dim=-1)
        v = self.net(x)
        return v
