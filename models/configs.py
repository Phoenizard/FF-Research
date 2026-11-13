from copy import deepcopy

from methods.BiCovG import DEFAULTS as BICOVG_DEFAULTS
from methods.CwC import DEFAULTS as CWC_DEFAULTS
from methods.DTG_FF import DEFAULTS as DTG_DEFAULTS
from methods.DeeperForward import DEFAULTS as DEEPERFORWARD_DEFAULTS
from methods.HCL_FF import DEFAULTS as HCL_DEFAULTS, LEVELS, LEVELS_16
from methods.TTG import DEFAULTS as TTG_DEFAULTS, TTGS_DEFAULTS


DATASETS = {
    "cifar10": {"num_classes": 10, "channels": [100, 100, 200, 200, 400, 400, 400, 400]},
    "cifar100": {"num_classes": 100, "channels": [100, 100, 200, 200, 400, 400, 400, 400]},
    "tinyimagenet": {"num_classes": 200, "channels": [200, 200, 200, 200, 400, 400, 400, 400]},
}


def experiment_config(dataset, seed, method="asge", depth=8, shortcut=False):
    if depth not in (8, 16, 31) or (depth == 16 and dataset not in ("cifar100", "tinyimagenet")):
        raise ValueError(f"The 16-layer backbone is defined for cifar100 and tinyimagenet only; got {dataset} depth {depth}")
    if depth == 31 and (dataset != "cifar100" or method == "hcl_ff"):
        raise ValueError(f"The ResNet-32 backbone (depth 31) is defined for cifar100 without hcl_ff; got {dataset} {method}")
    if shortcut and depth != 31:
        raise ValueError("Identity shortcuts are defined for the ResNet-32 backbone (depth 31) only")
    config = {
        "dataset": dataset,
        **deepcopy(DATASETS[dataset]),
        "method": method,
        "seed": seed,
        "epochs": 400,
        "batch_size": 256,
        "run_kind": "benchmark",
        "data_split": {"train": 50000, "test": 10000, "validation": 0},
        "evaluation_scope": "official_test",
        "conv": {"kernel_size": 3, "stride": 1, "padding": 1,
                 "dilation": 1, "groups": 1, "bias": True},
        "initialization": "kaiming_normal_relu_zero_bias",
        "alpha": 1.0,
        "pool_after": [1, 3, 4, 5],
        "pool": "rms_2x2_stride2",
        "dropout": 0.1,
        "norm": "sample_rms_no_affine",
        "norm_epsilon": 1.9287498479639178e-22,
        "prediction": "unified_fusion_relu_gap",
        "fusion": {
            "layers": list(range(1, 9)),
            "feature_location": "relu_before_pool_dropout_norm",
            "initialization": "kaiming_normal_relu_zero_bias",
            "optimizer": "AdamW",
            "learning_rate": 2e-4,
            "minimum_learning_rate": 1e-5,
            "weight_decay": 1e-3,
            "betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "scheduler": "CosineAnnealingLR",
        },
        "optimizer": "AdamW",
        "learning_rate": 2e-4,
        "minimum_learning_rate": 1e-5,
        "weight_decay": 1e-3,
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "scheduler": "CosineAnnealingLR",
        "precision": "float32",
        "augmentation": {"crop_size": 32, "padding": 4, "horizontal_flip": 0.5},
        "normalization": None,
        "source": "https://arxiv.org/abs/2509.12394v1",
        "source_commit": "3226665035a15da90c70f3df0a6bc5ea7de326cc",
    }
    if dataset == "tinyimagenet":
        config.update(
            data_split={"train": 100000, "test": 10000, "validation": 0},
            evaluation_scope="official_validation",
            augmentation={"crop_size": 64, "padding": 8, "horizontal_flip": 0.5})
    if method in ("bicovg", "cwc", "deeperforward", "hcl_ff"):
        config.update({"bicovg": BICOVG_DEFAULTS, "cwc": CWC_DEFAULTS,
                       "deeperforward": DEEPERFORWARD_DEFAULTS, "hcl_ff": HCL_DEFAULTS}[method])
        config.pop("alpha")
    if method == "dtg_ff":
        config.update(DTG_DEFAULTS)
    if method == "ttg":
        config.update(TTG_DEFAULTS)
    if method == "ttgs":
        config.update(TTGS_DEFAULTS)
    if depth == 16:
        # Each item of the 8-layer backbone is repeated once; pools move to the matching layers.
        config["channels"] = [channels for channels in config["channels"] for _ in range(2)]
        config["pool_after"] = [2 * index + 1 for index in config["pool_after"]]
        config["fusion"]["layers"] = list(range(1, 17))
    if depth == 31:
        # ResNet-32 layout (README consensus 4): stem + 3 stages x 10 convs, pools after the channel-changing convs
        # L12 and L22; identity shortcuts on the 13 same-shape blocks (L12-L13 and L22-L23 change shape).
        config["channels"] = [100] * 11 + [200] * 10 + [400] * 10
        config["pool_after"] = [11, 21]
        config["fusion"]["layers"] = list(range(1, 32))
        config["epochs"] = 300
        config["shortcut_blocks"] = ([[first, first + 1] for first in range(2, 32, 2) if first not in (12, 22)]
                                     if shortcut else [])
    if method in ("ttg", "ttgs"):
        config["tree_feature_layer"] = len(config["channels"])
    if method == "hcl_ff":
        config["hierarchy_levels"] = (LEVELS if depth == 8 else LEVELS_16)[dataset].copy()
    if method == "bp":
        config.update(global_bp=True, source="common_backbone_global_bp", source_commit=None)
        config.pop("alpha")
    return config
