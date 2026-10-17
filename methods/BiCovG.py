import torch
from torch import nn
from torch.nn import functional as functional

from methods.asge import ASGE


DEFAULTS = {
    "cc_reduction": 8,
    "gradient_clip": 1.0,
    "source": "https://github.com/Phoenizard/SIR-SID",
    "source_commit": "c656210dd9b30478f75120d1cad0626dd673b3c4",
}


class BiCovGReadout(nn.Module):
    def __init__(self, channels, spatial_size, classes, reduction):
        super().__init__()
        scales = [1, 2] if channels >= 256 else [2, 4] if channels >= 128 else [4, 8]
        self.scales = [scale for scale in scales if scale < spatial_size] or [min(scales)]
        projected_channels = channels // reduction
        self.projection = nn.Conv2d(channels, projected_channels, 1, bias=False)
        nn.init.orthogonal_(self.projection.weight.view(projected_channels, channels))
        dimensions = (channels + projected_channels) * sum(scale ** 2 for scale in self.scales)
        self.readout = nn.Linear(dimensions, classes)
        nn.init.kaiming_normal_(self.readout.weight, nonlinearity="relu")
        nn.init.zeros_(self.readout.bias)

    def forward(self, activations):
        energies = (activations.square(), self.projection(activations).square())
        goodness = torch.cat([functional.adaptive_avg_pool2d(energy, scale).flatten(1)
                              for energy in energies for scale in self.scales], dim=1)
        return self.readout(goodness)


class BiCovG(ASGE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.pool_after = config["pool_after"]
        self.epsilon = config["norm_epsilon"]
        self.dropout = nn.Dropout(config["dropout"])
        self.goodness = nn.ModuleList()
        spatial_size = 32
        for index, channels in enumerate(config["channels"]):
            self.goodness.append(BiCovGReadout(channels, spatial_size, config["num_classes"],
                                              config["cc_reduction"]))
            if index in self.pool_after:
                spatial_size //= 2
