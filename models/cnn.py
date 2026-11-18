import torch
from torch import nn
from torch.nn import functional as functional


class ShortcutStream:
    """Identity shortcuts over [first, second] conv pairs (1-based, config "shortcut_blocks"): the tensor entering the
    first conv is added, unit weight, to the second conv's propagated output. Shared by forward and train_step."""

    def __init__(self, blocks):
        self.seconds = {second - 1: first - 1 for first, second in blocks}
        self.block_input = None

    def before(self, index, hidden):
        if index in self.seconds.values():
            self.block_input = hidden

    def after(self, index, hidden):
        return hidden + self.block_input if index in self.seconds else hidden


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
        self.global_bp = config.get("global_bp", False)
        self.fusion_layers = config["fusion"]["layers"]
        self.shortcut_blocks = config.get("shortcut_blocks", [])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.fusion = nn.Linear(sum(config["channels"][layer - 1]
                                        for layer in self.fusion_layers), config["num_classes"])
            nn.init.kaiming_normal_(self.fusion.weight, nonlinearity="relu")
            nn.init.zeros_(self.fusion.bias)
        # Last-BP: a trainable head on the last layer carries the only training loss; built in its own fork so the
        # conv and Fusion initial values stay those of Global BP.
        self.head = None
        if "bp_head" in config:
            self.head_layers = config["bp_head"]["layers"]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"])
                self.head = nn.Linear(sum(config["channels"][layer - 1] for layer in self.head_layers),
                                      config["num_classes"])
                nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
                nn.init.zeros_(self.head.bias)

    def fusion_features(self, activations):
        features = activations if self.global_bp else activations.detach()
        return features.mean(dim=(2, 3))

    def predict(self, features):
        # Fusion trains the backbone only in Global BP; with a Last-BP head it is read-only.
        trains_backbone = self.global_bp and self.head is None
        return self.fusion(torch.cat([features[layer - 1] if trains_backbone
                                      else features[layer - 1].detach()
                                      for layer in self.fusion_layers], dim=1))

    def forward(self, images, return_local=False):
        hidden = self.method.prepare_input(images)
        stream = ShortcutStream(self.shortcut_blocks)
        prediction_features = []
        local_logits = []
        for index, conv in enumerate(self.convs):
            hidden = hidden if self.global_bp else hidden.detach()
            stream.before(index, hidden)
            activations = functional.relu(conv(hidden))
            prediction_features.append(self.fusion_features(activations))
            activations = self.method.prepare_activations(index, activations)
            if return_local and not self.global_bp:
                local_logits.append(self.method.local_logits(index, activations))
            hidden = stream.after(index, self.method.propagate(index, activations))
            if not self.global_bp:
                hidden = hidden.detach()
        logits = self.predict(prediction_features)
        if not return_local:
            return logits
        if self.head is not None:
            return logits, local_logits, self.head(torch.cat([prediction_features[layer - 1]
                                                              for layer in self.head_layers], dim=1))
        return logits, local_logits
