from torch import nn

from methods.asge import ASGE


DEFAULTS = {
    "pool": "max_2x2_stride2",
    "dropout": 0.0,
    "norm": "batch_norm_affine",
    "norm_epsilon": 1e-4,
    "norm_momentum": 0.1,
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "minimum_learning_rate": 0.01,
    "weight_decay": 0.0,
    "scheduler": "ConstantLR",
    "source": "https://github.com/andreaspapac/CwComp",
    "source_commit": "254960ab17451a25290c9c41b15c8abd587f9090",
}


class CwC(ASGE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_classes = config["num_classes"]
        self.goodness = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2) if index in config["pool_after"] else nn.Identity(),
                nn.BatchNorm2d(channels, eps=config["norm_epsilon"],
                               momentum=config["norm_momentum"]),
            )
            for index, channels in enumerate(config["channels"])
        ])

    def prepare_input(self, images):
        return images

    def prepare_activations(self, index, activations):
        return self.goodness[index](activations)

    def local_logits(self, index, activations):
        return activations.reshape(activations.shape[0], self.num_classes, -1).square().mean(dim=2)

    def propagate(self, index, activations):
        return activations
