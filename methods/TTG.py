import json
import time

import torch
from scipy.cluster.hierarchy import linkage
from torch.nn import functional as functional

from methods.DTG_FF import DEFAULTS as DTG_DEFAULTS, DTGFF
from methods.HCL_FF import MERGES


DEFAULTS = {
    **DTG_DEFAULTS,
    "tree_source": "internal",
    "tree_build_epoch": 5,
    "tree_linkage": "average",
    "tree_metric": "cosine",
    "source": "ff_research_ttg",
    "source_commit": None,
}
TREE_SOURCES = ("internal", "published")


def tree_signs(merges, classes):
    """Node signs (K x (K-1)): +1 left subtree, -1 right subtree; class depth; side code (2(K-1) x K)."""
    if len(merges) != classes - 1:
        raise ValueError(f"A binary tree over {classes} classes needs {classes - 1} merges, got {len(merges)}")
    leaves = [[k] for k in range(classes)]
    sign = torch.zeros(classes, classes - 1, dtype=torch.float64)
    for i, (left, right) in enumerate(merges):
        sign[leaves[left], i] = 1.0
        sign[leaves[right], i] = -1.0
        leaves.append(leaves[left] + leaves[right])
    depth = sign.abs().sum(dim=1)
    side = torch.zeros(2 * (classes - 1), classes, dtype=torch.float64)
    side[0::2] = sign.t()
    side[1::2] = -sign.t()
    return depth, side


def unit_permutations(seed, dims):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(dim, generator=generator) for dim in dims]


def tree_code(dim, classes, side, depth, perm):
    """Sparse ternary code (dim x K): each node side owns dim // (2(K-1)) energy units; column norm² dim/K, sum 0."""
    sides = 2 * (classes - 1)
    rows = dim // sides
    if rows < 1:
        raise ValueError(f"Goodness dimension {dim} is smaller than the {sides} node sides")
    code = torch.zeros(dim, classes, dtype=torch.float64)
    for s in range(sides):
        code[perm[s * rows:(s + 1) * rows], :] = side[s, :]
    return code * torch.sqrt(dim / (classes * 2 * rows * depth))[None, :]


def centered_code(code, dim, classes):
    """Random code with column means removed, rescaled to column norm² dim/K."""
    code = code.to(torch.float64)
    code = code - code.mean(dim=0)[None, :]
    return code * (dim / classes) ** 0.5 / code.norm(dim=0)[None, :]


class TTG(DTGFF):
    """DTG-FF with a fixed tree-coded readout. tree_source="internal" builds the class tree once from training
    class means of layer tree_feature_layer during epoch tree_build_epoch (centred random code before that);
    tree_source="published" uses the released HCL-FF tree from the start."""

    def __init__(self, config):
        super().__init__(config)
        if config["tree_source"] not in TREE_SOURCES:
            raise ValueError(f"tree_source must be one of {TREE_SOURCES}, got {config['tree_source']!r}")
        self.epoch = 0
        self.tree_source = config["tree_source"]
        self.build_epoch = config["tree_build_epoch"]
        self.feature_index = config["tree_feature_layer"] - 1
        self.linkage = (config["tree_linkage"], config["tree_metric"])
        self.classes = config["num_classes"]
        self.dataset = config["dataset"]
        self.seed = config["seed"]
        channels = config["channels"][self.feature_index]
        self.register_buffer("tree_sums", torch.zeros(self.classes, channels, dtype=torch.float64))
        self.register_buffer("tree_counts", torch.zeros(self.classes, dtype=torch.float64))
        self.register_buffer("tree_merges", torch.full((self.classes - 1, 2), -1, dtype=torch.int64))
        self.register_buffer("tree_built", torch.zeros((), dtype=torch.bool))
        self.register_buffer("tree_build_seconds", torch.zeros((), dtype=torch.float64))
        self.built = False
        with torch.no_grad():
            if self.tree_source == "published":
                self.write_tree(MERGES[self.dataset])
            else:
                for head in self.goodness:
                    head.projection.copy_(centered_code(head.projection, head.projection.shape[0],
                                                        self.classes).to(torch.float32))
                    head.bias.zero_()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.built = bool(self.tree_built)

    def write_tree(self, merges):
        depth, side = tree_signs(merges, self.classes)
        dims = [head.projection.shape[0] for head in self.goodness]
        for head, dim, perm in zip(self.goodness, dims, unit_permutations(self.seed, dims), strict=True):
            head.projection.copy_(tree_code(dim, self.classes, side, depth, perm).to(torch.float32))
            head.bias.zero_()
        self.tree_merges.copy_(torch.tensor(merges))
        self.tree_built.fill_(True)
        self.built = True

    @torch.no_grad()
    def build_tree(self):
        started = time.perf_counter()
        if bool((self.tree_counts == 0).any()):
            raise RuntimeError("Tree building needs every class in the accumulated epoch")
        means = (self.tree_sums / self.tree_counts[:, None]).cpu().numpy()
        tree = linkage(means - means.mean(axis=0), method=self.linkage[0], metric=self.linkage[1])
        self.write_tree([(int(tree[i, 0]), int(tree[i, 1])) for i in range(self.classes - 1)])
        if self.tree_sums.is_cuda:
            torch.cuda.synchronize()
        self.tree_build_seconds.fill_(time.perf_counter() - started)
        print(json.dumps({"ttg_tree_built_epoch": self.epoch + 1,
                          "build_seconds": float(self.tree_build_seconds),
                          "merges": self.tree_merges.tolist()}), flush=True)

    def local_loss(self, index, activations, labels):
        if self.training and index == 0 and not self.built and self.epoch >= self.build_epoch:
            self.build_tree()
        loss, logits = super().local_loss(index, activations, labels)
        if (self.training and not self.built and self.epoch == self.build_epoch - 1
                and index == self.feature_index):
            with torch.no_grad():
                features = activations.detach().mean(dim=(2, 3)).double()
                onehot = functional.one_hot(labels, self.classes).double()
                self.tree_sums += onehot.t() @ features
                self.tree_counts += onehot.sum(dim=0)
        return loss, logits
