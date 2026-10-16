import torch
from torch import nn
from torch.nn import functional as functional


class LocalCNN(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.convs = nn.ModuleList()
            input_channels = 3
            for output_channels in config["channels"]:
                conv = nn.Conv2d(input_channels, output_channels, **config["conv"])
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                nn.init.zeros_(conv.bias)
                self.convs.append(conv)
                input_channels = output_channels
        self.method = method
        self.fusion_layers = config["fusion"]["layers"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.fusion = nn.Linear(sum(config["channels"][layer - 1]
                                        for layer in self.fusion_layers), config["num_classes"])
            nn.init.kaiming_normal_(self.fusion.weight, nonlinearity="relu")
            nn.init.zeros_(self.fusion.bias)

    def fusion_features(self, activations):
        return activations.detach().mean(dim=(2, 3))

    def predict(self, features):
        return self.fusion(torch.cat([features[layer - 1].detach()
                                      for layer in self.fusion_layers], dim=1))

    def forward(self, images, return_local=False):
        hidden = self.method.prepare_input(images)
        prediction_features = []
        local_logits = []
        for index, conv in enumerate(self.convs):
            activations = functional.relu(conv(hidden.detach()))
            prediction_features.append(self.fusion_features(activations))
            if return_local:
                local_logits.append(self.method.local_logits(index, activations))
            hidden = self.method.propagate(index, activations).detach()
        logits = self.predict(prediction_features)
        return (logits, local_logits) if return_local else logits
