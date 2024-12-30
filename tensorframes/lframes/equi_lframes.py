import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.transforms import restframe_transform


class RestLFrames(LFramesPredictor):
    """Local frames corresponding to the rest frames of the particles"""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta):
        transform = restframe_transform(fourmomenta)
        return LFrames(transform)
