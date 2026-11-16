import math

import torch
from torch import nn
from torch.nn import functional as functional


class _PooledEnergy(torch.autograd.Function):
    """adaptive_avg_pool2d(F^2, patch) with the backward of both steps fused: dF = F * 2 * dg / block_size.
    Same values and gradients as the two autograd steps, without their full-size temporaries."""

    @staticmethod
    def forward(ctx, activations, patch):
        ctx.save_for_backward(activations)
        ctx.patch = patch
        return functional.adaptive_avg_pool2d(activations.square(), patch)

    @staticmethod
    def backward(ctx, gradient):
        (activations,) = ctx.saved_tensors
        batch, channels, height, width = activations.shape
        patch, block_h, block_w = ctx.patch, height // ctx.patch, width // ctx.patch
        scaled = (gradient * (2.0 / (block_h * block_w))).reshape(batch, channels, patch, 1, patch, 1)
        blocks = activations.reshape(batch, channels, patch, block_h, patch, block_w)
        return (blocks * scaled).reshape(batch, channels, height, width), None


def pooled_energy(activations, patch):
    if activations.shape[2] % patch or activations.shape[3] % patch:
        return functional.adaptive_avg_pool2d(activations.square(), patch)
    return _PooledEnergy.apply(activations, patch)


class SpatialGoodness(nn.Module):
    def __init__(self, channels, last_channels, spatial_size, classes, alpha):
        super().__init__()
        self.patch_size = min(max(1, math.floor(alpha * last_channels / channels)), spatial_size)
        self.register_buffer("projection", torch.randn(channels * self.patch_size ** 2, classes)
                             / math.sqrt(classes))
        self.register_buffer("bias", torch.randn(classes) / math.sqrt(classes))

    def forward(self, activations):
        goodness = pooled_energy(activations, self.patch_size)
        return goodness.flatten(1) @ self.projection + self.bias


class ASGE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool_after = config["pool_after"]
        self.epsilon = config["norm_epsilon"]
        self.dropout = nn.Dropout(config["dropout"])
        self.goodness = nn.ModuleList()
        spatial_size = config["augmentation"]["crop_size"]
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

    def prepare_activations(self, index, activations):
        return activations

    def local_logits(self, index, activations):
        return self.goodness[index](activations)

    def propagate(self, index, activations):
        if index in self.pool_after:
            activations = functional.avg_pool2d(activations.square(), 2).sqrt()
        return self.prepare_input(self.dropout(activations))

