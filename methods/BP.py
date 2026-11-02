import math

import torch
from torch import nn

from methods.asge import ASGE


class BP(ASGE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.pool_after = config["pool_after"]
        self.epsilon = config["norm_epsilon"]
        self.dropout = nn.Dropout(config["dropout"])
        self.goodness = nn.ModuleList()

    def prepare_input(self, images):
        rms = torch.linalg.vector_norm(images, dim=(1, 2, 3), keepdim=True)
        rms = rms / math.sqrt(images[0].numel())
        return images / (rms + self.epsilon)

    def propagate(self, index, activations):
        if index in self.pool_after:
            patches = activations.unfold(2, 2, 2).unfold(3, 2, 2)
            activations = torch.linalg.vector_norm(patches, dim=(-2, -1)) / 2
        return self.prepare_input(self.dropout(activations))
