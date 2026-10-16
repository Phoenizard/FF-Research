import math

import torch
from torch import nn
from torch.nn import functional as functional


class SpatialGoodness(nn.Module):
    def __init__(self, channels, last_channels, spatial_size, classes, alpha):
        super().__init__()
        self.patch_size = min(max(1, math.floor(alpha * last_channels / channels)), spatial_size)
        self.register_buffer("projection", torch.randn(channels * self.patch_size ** 2, classes)
                             / math.sqrt(classes))
        self.register_buffer("bias", torch.randn(classes) / math.sqrt(classes))

    def forward(self, activations):
        goodness = functional.adaptive_avg_pool2d(activations.square(), self.patch_size)
        return goodness.flatten(1) @ self.projection + self.bias


class ASGE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool_after = config["pool_after"]
        self.epsilon = config["norm_epsilon"]
        self.dropout = nn.Dropout(config["dropout"])
        self.goodness = nn.ModuleList()
        spatial_size = 32
        for index, channels in enumerate(config["channels"]):
            self.goodness.append(SpatialGoodness(channels, config["channels"][-1],
                                                spatial_size, config["num_classes"], config["alpha"]))
            if index in self.pool_after:
                spatial_size //= 2

    def prepare_input(self, images):
        rms = images.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return images / (rms + self.epsilon)

    def local_loss(self, index, activations, labels):
        logits = self.local_logits(index, activations)
        return functional.cross_entropy(logits, labels), logits.detach()

    def local_logits(self, index, activations):
        return self.goodness[index](activations)

    def propagate(self, index, activations):
        if index in self.pool_after:
            activations = functional.avg_pool2d(activations.square(), 2).sqrt()
        return self.prepare_input(self.dropout(activations))

