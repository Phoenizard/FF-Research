import torch
from torch import nn
from torch.nn import functional as functional

from methods.asge import ASGE


DEFAULTS = {
    "temperature_min": 0.5,
    "temperature_max": 2.0,
    "pool": "avg_2x2_stride2",
    "norm": "channel_layer_norm_no_affine",
    "norm_epsilon": 1e-5,
    "source": "https://arxiv.org/abs/2606.06539v1",
    "source_commit": None,
}


class DTGFF(ASGE):
    def __init__(self, config):
        super().__init__(config)
        self.temperature_min = config["temperature_min"]
        self.temperature_max = config["temperature_max"]
        for head in self.goodness:
            head.temperature_logit = nn.Parameter(torch.zeros(()))

    def prepare_input(self, images):
        return images

    def local_logits(self, index, activations):
        head = self.goodness[index]
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * \
            head.temperature_logit.sigmoid()
        goodness = functional.adaptive_avg_pool2d(activations.square(), head.patch_size)
        return (goodness.flatten(1) / temperature) @ head.projection + head.bias

    def propagate(self, index, activations):
        if index in self.pool_after:
            activations = functional.avg_pool2d(activations, 2)
        activations = functional.layer_norm(activations.permute(0, 2, 3, 1),
                                            (activations.shape[1],), eps=self.epsilon)
        return self.dropout(activations.permute(0, 3, 1, 2))
