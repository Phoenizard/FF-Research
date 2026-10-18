import math

import torch
from torch import nn
from torch.nn import functional as functional

from methods.DeeperForward import DeeperForward


DEFAULTS = {
    "pool": "avg_2x2_stride2",
    "dropout": 0.0,
    "norm": "class_group_norm_no_affine",
    "norm_epsilon": 1e-5,
    "input_norm": "sample_layer_norm_no_affine",
    "optimizer": "AdamW",
    "learning_rate": 0.04,
    "minimum_learning_rate": 0.0002,
    "weight_decay": 1e-4,
    "scheduler": "LambdaLR",
    "projection_dim": 128,
    "projection_lr_multiplier": 2.0,
    "local_gradient_clip": 1.0,
    "contrastive_weight": 1.0,
    "temperature_high": 0.8,
    "temperature_mid": 0.2,
    "temperature_low": 0.1,
    "temperature_warmup_epochs": 50,
    "hierarchy_source": "author_released_class_tree",
    "source": "https://github.com/JNNNNYao/HCL-FF",
    "source_commit": "799349bd41f7e37b23aa03338d4f115a180e256c",
}

MERGES = {
    "cifar10": [
        (1, 7), (0, 9), (4, 6), (2, 8), (5, 13), (11, 12),
        (3, 10), (14, 15), (16, 17),
    ],
    "cifar100": [
        (35, 98), (11, 100), (47, 52), (50, 74), (62, 92), (55, 72),
        (10, 61), (78, 99), (30, 73), (2, 101), (34, 42), (27, 44),
        (96, 102), (0, 83), (3, 21), (70, 104), (4, 105), (60, 71),
        (7, 24), (26, 45), (81, 90), (5, 25), (23, 117), (88, 110),
        (14, 67), (38, 65), (95, 108), (77, 80), (53, 113), (6, 79),
        (33, 59), (8, 48), (41, 89), (18, 111), (15, 19), (22, 106),
        (36, 64), (57, 128), (87, 94), (29, 93), (46, 109), (16, 40),
        (103, 136), (58, 132), (107, 133), (97, 123), (91, 126), (118, 129),
        (49, 130), (82, 115), (51, 141), (66, 125), (54, 149), (119, 139),
        (39, 86), (31, 134), (13, 120), (85, 143), (17, 76), (1, 124),
        (127, 142), (20, 121), (12, 68), (122, 148), (84, 161), (114, 116),
        (135, 150), (37, 56), (75, 159), (43, 145), (9, 154), (63, 160),
        (138, 164), (69, 158), (151, 171), (28, 166), (144, 168), (32, 146),
        (153, 176), (162, 167), (163, 179), (165, 174), (131, 157), (137, 152),
        (147, 178), (112, 180), (170, 173), (175, 183), (156, 182), (172, 186),
        (155, 169), (177, 184), (181, 190), (188, 189), (185, 193), (140, 192),
        (187, 191), (195, 196), (194, 197),
    ],
}

LEVELS = {
    "cifar10": [1, 1, 2, 2, 3, 3, 4, 4],
    "cifar100": [1, 3, 6, 8, 10, 10, 10, 10],
}


def hierarchy_groups(dataset, classes):
    merges = MERGES[dataset]
    leaves = [[index] for index in range(classes)]
    for left, right in merges:
        leaves.append(leaves[left] + leaves[right])
    frontier = [len(leaves) - 1]
    levels = []
    while any(node >= classes for node in frontier):
        frontier = [child for node in frontier
                    for child in (merges[node - classes] if node >= classes else [node])]
        levels.append([leaves[node] for node in frontier])
    return [levels[level - 1] for level in LEVELS[dataset]]


class HCLFF(DeeperForward):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.epoch = 0
        groups_by_layer = hierarchy_groups(config["dataset"], self.num_classes)
        self.goodness = nn.ModuleList()
        for channels, groups in zip(config["channels"], groups_by_layer, strict=True):
            pool_size = 4 if channels <= 100 else 2 if channels <= 400 else 1
            head = nn.Sequential(nn.AdaptiveAvgPool2d(pool_size), nn.Flatten(),
                                 nn.Linear(channels * pool_size ** 2, config["projection_dim"],
                                           bias=False),
                                 nn.BatchNorm1d(config["projection_dim"]))
            label_map = torch.empty(self.num_classes, dtype=torch.long)
            aggregation = torch.zeros(self.num_classes, len(groups))
            for group_index, members in enumerate(groups):
                label_map[members] = group_index
                aggregation[members, group_index] = 1.0 / len(members)
            head.register_buffer("label_map", label_map)
            head.register_buffer("aggregation", aggregation)
            self.goodness.append(head)

    def prepare_activations(self, index, activations):
        return activations

    def temperature(self):
        warmup = self.config["temperature_warmup_epochs"]
        high, middle, low = (self.config[key] for key in
                             ("temperature_high", "temperature_mid", "temperature_low"))
        if self.epoch < warmup:
            return high + (middle - high) * self.epoch / warmup
        progress = (self.epoch - warmup) / (self.config["epochs"] - warmup)
        return low + 0.5 * (middle - low) * (1 + math.cos(math.pi * progress))

    def local_loss(self, index, activations, labels):
        logits = self.local_logits(index, activations)
        head = self.goodness[index]
        classification = functional.cross_entropy(logits @ head.aggregation,
                                                   head.label_map[labels])
        normalized = functional.group_norm(activations, self.num_classes, eps=self.epsilon)
        embeddings = functional.normalize(head(normalized), dim=1)
        similarities = embeddings @ embeddings.T / self.temperature()
        diagonal = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        positives = labels[:, None].eq(labels[None, :]) & ~diagonal
        log_probabilities = similarities - similarities.masked_fill(
            diagonal, -torch.inf).logsumexp(dim=1, keepdim=True)
        contrastive = -((positives * log_probabilities).sum(dim=1) /
                        positives.sum(dim=1).clamp_min(1)).mean()
        return classification + self.config["contrastive_weight"] * contrastive, logits.detach()

    def optimizer_parameters(self, conv, index, config):
        return [{"params": conv.parameters()},
                {"params": self.goodness[index].parameters(),
                 "lr": config["learning_rate"] * config["projection_lr_multiplier"]}]

    def learning_rate_factor(self, epoch):
        minimum = self.config["minimum_learning_rate"] / self.config["learning_rate"]
        return minimum + 0.5 * (1 - minimum) * (1 + math.cos(math.pi * epoch /
                                                            self.config["epochs"]))

    def clip_gradients(self, index, conv):
        nn.utils.clip_grad_norm_(conv.parameters(), self.config["local_gradient_clip"])
        nn.utils.clip_grad_norm_(self.goodness[index].parameters(),
                                 self.config["local_gradient_clip"])

