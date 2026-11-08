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
    # Author tree leaf indices follow ImageFolder's sorted WNIDs. Preserve the
    # numeric indices; display names in the released JSON use a different order.
    "tinyimagenet": [
        (30, 32), (2, 3), (104, 174), (85, 134), (186, 187), (70, 94),
        (182, 190), (15, 16), (73, 135), (68, 171), (86, 128), (80, 172),
        (40, 42), (153, 205), (24, 27), (177, 179), (121, 133), (74, 158),
        (33, 34), (110, 141), (49, 53), (82, 93), (26, 214), (7, 10),
        (75, 156), (131, 159), (192, 206), (99, 125), (77, 123), (39, 212),
        (197, 198), (37, 38), (29, 222), (105, 119), (112, 169), (25, 28),
        (100, 138), (97, 98), (175, 225), (61, 126), (14, 196), (160, 211),
        (62, 234), (48, 56), (18, 19), (154, 157), (117, 155), (132, 137),
        (5, 88), (199, 207), (79, 122), (178, 215), (31, 47), (87, 219),
        (114, 164), (95, 101), (64, 167), (41, 223), (51, 220), (17, 46),
        (67, 151), (106, 224), (127, 168), (180, 228), (59, 92), (65, 147),
        (116, 216), (139, 247), (227, 238), (4, 201), (43, 229), (243, 258),
        (35, 50), (66, 162), (89, 260), (143, 209), (226, 251), (120, 236),
        (11, 22), (54, 55), (96, 195), (165, 170), (72, 241), (84, 267),
        (144, 248), (63, 261), (150, 262), (250, 268), (230, 280), (8, 9),
        (102, 152), (69, 140), (188, 204), (83, 163), (71, 124), (90, 149),
        (184, 189), (181, 191), (76, 286), (136, 265), (210, 256), (257, 289),
        (161, 183), (113, 130), (20, 52), (233, 242), (57, 272), (148, 217),
        (203, 305), (194, 288), (145, 166), (118, 255), (240, 259), (221, 299),
        (78, 107), (129, 290), (232, 235), (108, 115), (298, 315), (36, 231),
        (1, 58), (13, 23), (176, 303), (146, 314), (278, 304), (91, 193),
        (277, 285), (12, 295), (302, 327), (44, 45), (274, 291), (185, 318),
        (284, 287), (21, 60), (244, 297), (109, 322), (142, 325), (0, 321),
        (253, 264), (263, 326), (111, 333), (6, 317), (283, 311), (279, 320),
        (293, 335), (237, 313), (292, 296), (271, 306), (200, 218), (103, 337),
        (81, 310), (173, 281), (323, 341), (330, 345), (273, 331), (266, 350),
        (254, 352), (249, 269), (328, 340), (245, 309), (208, 344), (336, 358),
        (319, 329), (300, 353), (294, 351), (202, 334), (252, 348), (343, 349),
        (246, 360), (239, 363), (307, 339), (282, 368), (276, 365), (354, 361),
        (356, 367), (324, 374), (312, 357), (373, 375), (308, 370), (338, 369),
        (332, 342), (270, 301), (213, 355), (359, 364), (376, 381), (316, 366),
        (362, 377), (346, 371), (347, 385), (383, 386), (275, 380), (379, 387),
        (378, 390), (382, 389), (391, 392), (372, 384), (393, 395), (394, 396),
        (388, 397),
    ],
}

# Sample the author's 17-layer schedule at round(i * 16 / 7), i = 0..7.
LEVELS = {
    "cifar10": [1, 1, 2, 2, 3, 3, 4, 4],
    "cifar100": [1, 3, 6, 8, 10, 10, 10, 10],
    "tinyimagenet": [1, 3, 6, 8, 10, 12, 14, 14],
}
# 16-layer backbone: the author's CIFAR-100 table min(1 + i, 10), i = 0..16, sampled at round(i * 16 / 15).
LEVELS_16 = {"cifar100": [1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 10, 10, 10, 10, 10]}


def hierarchy_groups(dataset, classes, hierarchy_levels):
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
    return [levels[level - 1] for level in hierarchy_levels]


class HCLFF(DeeperForward):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.epoch = 0
        groups_by_layer = hierarchy_groups(config["dataset"], self.num_classes, config["hierarchy_levels"])
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

