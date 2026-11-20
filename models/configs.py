from copy import deepcopy

from methods.BiCovG import DEFAULTS as BICOVG_DEFAULTS
from methods.CwC import DEFAULTS as CWC_DEFAULTS
from methods.DTG_FF import DEFAULTS as DTG_DEFAULTS
from methods.DeeperForward import DEFAULTS as DEEPERFORWARD_DEFAULTS
from methods.HCL_FF import DEFAULTS as HCL_DEFAULTS, LEVELS, LEVELS_16, LEVELS_31, LEVELS_32
from methods.TTG import DEFAULTS as TTG_DEFAULTS, TTGS_DEFAULTS
from models.resnet17 import ARCHITECTURE as RESNET17, WIDTHS as RESNET17_WIDTHS


DATASETS = {
    "cifar10": {"num_classes": 10, "channels": [100, 100, 200, 200, 400, 400, 400, 400]},
    "cifar100": {"num_classes": 100, "channels": [100, 100, 200, 200, 400, 400, 400, 400]},
    "tinyimagenet": {"num_classes": 200, "channels": [200, 200, 200, 200, 400, 400, 400, 400]},
}


def resnet17_config(dataset, seed, method):
    """DeeperForward ResNet17 (models/resnet17.py): FF at the platform's 400 epochs, BP with the DeeperForward
    SGD recipe for 150 epochs."""
    if dataset not in ("cifar10", "cifar100") or method not in ("bp", "asge", "ttgs"):
        raise ValueError(f"ResNet17 (depth 17) is defined for cifar10/cifar100 with bp, asge, ttgs; "
                         f"got {dataset} {method}")
    base = experiment_config(dataset, seed, "ttgs" if method == "ttgs" else "asge")
    config = {
        "dataset": dataset, "num_classes": base["num_classes"], "method": method, "seed": seed,
        "epochs": 150 if method == "bp" else 400, "batch_size": base["batch_size"], "run_kind": "benchmark",
        "data_split": base["data_split"], "evaluation_scope": base["evaluation_scope"],
        "augmentation": base["augmentation"], "normalization": base["normalization"],
        "architecture": RESNET17, "widths": RESNET17_WIDTHS, "convs": 17,
        "merge": "add_after_b_concat_after_d_on_layer_output", "shortcut_downsample": "avg_pool_2x2",
        "conv_bias": False, "initialization": "kaiming_normal_fan_in_relu",
        "prediction": base["prediction"],
        "fusion": {"layers": "every_relu_output", "feature_location": "relu_output_gap", "read_only": True,
                   **{key: base["fusion"][key] for key in ("initialization", "optimizer", "learning_rate",
                                                           "minimum_learning_rate", "weight_decay", "betas",
                                                           "optimizer_epsilon", "scheduler")}},
        "precision": base["precision"],
        "source": "https://github.com/tobysunsun/deeperforward", "source_commit": "f6f5e48",
    }
    if method == "bp":
        config.update({"layer": "conv_bn_relu", "bn_init": "weight_one_bias_zero", "head": "gap_last_tensor_linear",
                       "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9, "nesterov": False,
                       "weight_decay": 1e-4, "scheduler": "MultiStepLR",
                       "milestones": [config["epochs"] // 2, (3 * config["epochs"]) // 4], "gamma": 0.1})
        return config
    config.update({key: base[key] for key in ("optimizer", "learning_rate", "minimum_learning_rate",
                                              "weight_decay", "betas", "optimizer_epsilon", "scheduler",
                                              "dropout", "norm", "norm_epsilon")})
    config.update({"learning_unit": "each_conv", "goodness_location": "relu_output", "alpha": 1.0})
    if method == "asge":
        config.update({"layer": "conv_relu_dropout_sample_rms", "readout": "fixed_random",
                       "input_norm": "sample_rms"})
    else:
        config.update({key: TTGS_DEFAULTS[key] for key in ("temperature_min", "temperature_max", "tree_build_epoch",
                                                           "tree_linkage", "tree_metric", "skip_beta_sq",
                                                           "stochastic_depth_max")})
        config.update({"layer": "conv_relu_channel_layer_norm", "readout": "tree_code_learnable_temperature",
                       "tree_source": "internal", "tree_feature_layer": 17, "input_norm": None,
                       "stream": "relax_every_same_shape_layer_output_after_merge_reset_on_shape_change",
                       "stream_dropout": "after_stream_on_next_input"})
    return config


def experiment_config(dataset, seed, method="asge", depth=8, shortcut=False):
    if depth == 17:
        if shortcut:
            raise ValueError("--shortcut is defined for the CNN32 backbone (depth 31) only")
        return resnet17_config(dataset, seed, method)
    if depth not in (8, 16, 31, 32) or (depth in (16, 32) and dataset not in ("cifar100", "tinyimagenet")):
        raise ValueError(f"The 16-layer and CNN32-depth backbones are defined for cifar100 and tinyimagenet only; "
                         f"got {dataset} depth {depth}")
    if depth == 31 and dataset not in ("cifar100", "tinyimagenet"):
        raise ValueError(f"The CNN32 backbone (depth 31) is defined for cifar100 and tinyimagenet only; got {dataset}")
    if shortcut and depth != 31:
        raise ValueError("Residual shortcuts are defined for the CNN32 backbone (depth 31) only")
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
    if depth in (16, 32):
        # Each item of the 8-layer backbone is repeated (2 convs per item at depth 16, 4 at depth 32, the
        # CNN32-depth backbone of README consensus 5); pools move to the matching layers.
        repeat = depth // 8
        config["channels"] = [channels for channels in config["channels"] for _ in range(repeat)]
        config["pool_after"] = [repeat * index + repeat - 1 for index in config["pool_after"]]
        config["fusion"]["layers"] = list(range(1, depth + 1))
    if depth == 31:
        # CNN32 layout (README consensus 4): the CNN per-layer design deepened to 31 convs, stem + 3 stages x 10 convs, pools after the channel-changing convs
        # L12 and L22; identity shortcuts on the 13 same-shape blocks (L12-L13 and L22-L23 change shape).
        # The first stage keeps the dataset's first 8-layer width (100 for CIFAR, 200 for Tiny-ImageNet).
        config["channels"] = [config["channels"][0]] * 11 + [200] * 10 + [400] * 10
        config["pool_after"] = [11, 21]
        config["fusion"]["layers"] = list(range(1, 32))
        config["epochs"] = 300
        config["shortcut_blocks"] = ([[first, first + 1] for first in range(2, 32, 2) if first not in (12, 22)]
                                     if shortcut else [])
    if method in ("ttg", "ttgs"):
        config["tree_feature_layer"] = len(config["channels"])
    if method == "hcl_ff":
        config["hierarchy_levels"] = {8: LEVELS, 16: LEVELS_16, 31: LEVELS_31, 32: LEVELS_32}[depth][dataset].copy()
    if method in ("bp", "lastbp"):
        config.update(global_bp=True, source="common_backbone_global_bp", source_commit=None)
        config.pop("alpha")
    if method == "lastbp":
        # Global BP whose trainable head reads only the last layer; the Fusion head becomes read-only.
        config.update(source="common_backbone_last_layer_bp",
                      bp_head={"layers": [len(config["channels"])],
                               "feature_location": "relu_before_pool_dropout_norm",
                               "initialization": "kaiming_normal_relu_zero_bias"})
    return config
