from torch import nn
from torch.nn import functional as functional

from methods.asge import ASGE


DEFAULTS = {
    "pool": "avg_2x2_stride2",
    "dropout": 0.2,
    "norm": "class_group_norm_no_affine",
    "norm_epsilon": 1e-5,
    "input_norm": "sample_layer_norm_no_affine",
    "optimizer": "AdamW",
    "learning_rate": 0.08,
    "minimum_learning_rate": 0.008,
    "weight_decay": 5e-4,
    "scheduler": "CosineAnnealingLR",
    "source": "https://github.com/tobysunsun/deeperforward",
    "source_commit": "f6f5e481a5fa97f89a73c9de3049f5239117ca35",
}


class DeeperForward(ASGE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_classes = config["num_classes"]
        self.pool_after = config["pool_after"]
        self.epsilon = config["norm_epsilon"]
        self.goodness = nn.ModuleList([nn.Dropout(config["dropout"])
                                       for _ in config["channels"]])

    def prepare_input(self, images):
        return functional.layer_norm(images, images.shape[1:], eps=self.epsilon)

    def prepare_activations(self, index, activations):
        return self.goodness[index](activations)

    def local_logits(self, index, activations):
        return activations.reshape(activations.shape[0], self.num_classes, -1).mean(dim=2)

    def propagate(self, index, activations):
        activations = functional.group_norm(activations, self.num_classes, eps=self.epsilon)
        if index in self.pool_after:
            activations = functional.avg_pool2d(activations, 2)
        return activations
