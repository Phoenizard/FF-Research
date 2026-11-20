"""BP, ASGE and TTGS on the DeeperForward ResNet17 backbone (train.py --depth 17).

Backbone (DeeperForward ResNet17, official code tobysunsun/deeperforward f6f5e48, model/resnet.py class Resnet):
17 convs, all 3x3, padding 1, no bias. Stem 3->100 (stride 1), then 4 blocks of 4 convs with widths 100/200/400/800
and strides 1/2/2/2 on the first conv of each block. In each block (convs a, b, c, d; every conv is C->C):
  b's output + the block input (add);  d's output concatenated with the post-add tensor (channels C -> 2C).
The stem output and each post-add tensor are kept as the next shortcut; each concat output is kept after a 2x2
stride-2 AvgPool (it is the add shortcut of the next block). Merges act on the per-layer output after its norm.
Feature maps: 100@32 -> 200@32 -> 400@16 -> 800@8 -> 1600@4.

Per-layer output o_l (the tensor that enters the merge):
  bp    conv -> BN -> ReLU                                       (end-to-end SGD, GAP + FC head on the last tensor)
  asge  conv -> ReLU -> [goodness] -> Dropout 0.1 -> sample RMS   (images sample-RMS normalized, fixed random readout)
  ttgs  conv -> ReLU -> [goodness] -> channel LayerNorm           (tree-coded readout with a learnable temperature)
For ttgs the merged tensor m_l then enters the stream: s_l = xi*(a*s_{l-1} + beta*m_l) + (1-xi)*s_{l-1} when
m_l has the shape of s_{l-1}, else s_l = m_l (reset); s_l replaces the layer output (next input after Dropout 0.1,
and the stored shortcut). beta^2 = 1/4, a = sqrt(3/4), d_l = 0.5 * l / 17 (l = 1..17), expected coefficients at test.
FF learning unit = one conv with its goodness (AvgPool F^2 on the ReLU output, alpha = 1) and its own AdamW;
stop-gradient between units. Read-only Fusion on the GAP of all 17 ReLU outputs for every method.
"""
import json

import torch
import torch.distributed as dist
from scipy.cluster.hierarchy import linkage
from torch import nn
from torch.nn import functional as functional

from methods.TTG import centered_code, tree_code, tree_signs, unit_permutations
from methods.asge import SpatialGoodness, pooled_energy

ARCHITECTURE = "deeperforward_resnet17"
WIDTHS = [100, 200, 400, 800]
NO, ADD, CONCAT = 0, 1, 2


def layout():
    """(in_channels, out_channels, stride, merge, keep, pool_kept) for the 17 convs, as in DeeperForward."""
    layers = [(3, WIDTHS[0], 1, NO, True, False)]
    for block, width in enumerate(WIDTHS):
        stride = 1 if block == 0 else 2
        last = block == len(WIDTHS) - 1
        layers += [(width, width, stride, NO, False, False), (width, width, 1, ADD, True, False),
                   (width, width, 1, NO, False, False), (width, width, 1, CONCAT, not last, True)]
    return layers


def sample_rms(x, epsilon):
    return x / (x.square().mean(dim=(1, 2, 3), keepdim=True).sqrt() + epsilon)


def channel_layer_norm(x, epsilon):
    return functional.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), eps=epsilon).permute(0, 3, 1, 2)


def merge(kind, out, shortcut):
    if kind == ADD:
        return out + shortcut
    if kind == CONCAT:
        return torch.cat([out, shortcut], dim=1)
    return out


def adamw(parameters, setting):
    return torch.optim.AdamW(parameters, lr=setting["learning_rate"], betas=tuple(setting["betas"]),
                             eps=setting["optimizer_epsilon"], weight_decay=setting["weight_decay"])


class ResNet17(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["method"] not in ("bp", "asge", "ttgs"):
            raise ValueError(f"ResNet17 supports bp, asge and ttgs; got {config['method']}")
        if config.get("tree_source", "internal") != "internal":
            raise ValueError("ResNet17 TTGS builds its tree internally; --tree-source published is not defined")
        self.method, self.layout = config["method"], layout()
        self.convs = nn.ModuleList(nn.Conv2d(i, o, 3, s, 1, bias=False) for i, o, s, *_ in self.layout)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        classes = config["num_classes"]
        self.head = None
        if self.method == "bp":
            self.norms = nn.ModuleList(nn.BatchNorm2d(o) for _, o, *_ in self.layout)
            self.head = nn.Linear(2 * WIDTHS[-1], classes)
            self.local_units = 0
        else:
            sizes, size = [], config["augmentation"]["crop_size"]
            for _, _, stride, *_ in self.layout:
                size //= stride
                sizes.append(size)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"])
                self.goodness = nn.ModuleList(SpatialGoodness(o, WIDTHS[-1], s, classes, config["alpha"])
                                              for (_, o, *_), s in zip(self.layout, sizes))
            self.dropout = nn.Dropout(config["dropout"])
            self.epsilon = config["norm_epsilon"]
            self.local_units = len(self.convs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.fusion = nn.Linear(sum(o for _, o, *_ in self.layout), classes)
            nn.init.kaiming_normal_(self.fusion.weight, nonlinearity="relu")
            nn.init.zeros_(self.fusion.bias)
        if self.method == "ttgs":
            self.temperature_min, self.temperature_max = config["temperature_min"], config["temperature_max"]
            for head in self.goodness:
                head.temperature_logit = nn.Parameter(torch.zeros(()))
                head.projection.copy_(centered_code(head.projection, head.projection.shape[0], classes))
                head.bias.zero_()
            self.classes, self.seed = classes, config["seed"]
            self.build_epoch = config["tree_build_epoch"]
            self.skip_beta = config["skip_beta_sq"] ** 0.5
            self.skip_a = (1.0 - config["skip_beta_sq"]) ** 0.5
            self.drop_max = config["stochastic_depth_max"]
            self.register_buffer("tree_sums", torch.zeros(classes, WIDTHS[-1], dtype=torch.float64))
            self.register_buffer("tree_counts", torch.zeros(classes, dtype=torch.float64))
            self.register_buffer("tree_merges", torch.full((classes - 1, 2), -1, dtype=torch.int64))
            self.register_buffer("tree_built", torch.zeros((), dtype=torch.bool))
            self.built = False
            self.generator = None
        self.epoch = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.method == "ttgs":
            self.built = bool(self.tree_built)

    def start_epoch(self, epoch):
        self.epoch = epoch
        if self.method == "ttgs" and not self.built and epoch >= self.build_epoch:
            self.build_tree()

    # ---- BP ----
    def forward_bp(self, images):
        x, shortcut, relus = images, None, []
        for conv, norm, (_, _, _, kind, keep, pool) in zip(self.convs, self.norms, self.layout):
            out = functional.relu(norm(conv(x)))
            relus.append(out)
            x = merge(kind, out, shortcut)
            if keep:
                shortcut = functional.avg_pool2d(x, 2) if pool else x
        head = self.head(x.mean(dim=(2, 3)))
        return head, self.fusion_logits(relus)

    # ---- FF ----
    def prepare_input(self, images):
        return sample_rms(images, self.epsilon) if self.method == "asge" else images

    def local_logits(self, index, activations):
        head = self.goodness[index]
        if self.method == "asge":
            return head(activations)
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * \
            head.temperature_logit.sigmoid()
        return (pooled_energy(activations, head.patch_size).flatten(1) / temperature) @ head.projection + head.bias

    def layer_output(self, activations):
        if self.method == "asge":
            return sample_rms(self.dropout(activations), self.epsilon)
        return channel_layer_norm(activations, self.epsilon)

    def keep_mask(self, batch, device, drop):
        if self.generator is None or self.generator.device != device or self.generator_epoch != self.epoch:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(1000 * self.seed + self.epoch)
            self.generator_epoch = self.epoch
        # Multi-GPU: every rank draws the whole global batch from the same generator and keeps its own slice.
        world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
        keep = torch.rand((batch * world, 1, 1, 1), device=device, generator=self.generator) >= drop
        return keep[rank * batch:(rank + 1) * batch]

    def stream(self, index, merged, previous):
        """TTGS relaxation; previous is s_{l-1} (None at the stem)."""
        if previous is None or previous.shape != merged.shape:
            return merged
        drop = self.drop_max * (index + 1) / len(self.convs)
        if self.training:
            return torch.where(self.keep_mask(merged.shape[0], merged.device, drop),
                               self.skip_a * previous + self.skip_beta * merged, previous)
        return (drop + (1.0 - drop) * self.skip_a) * previous + ((1.0 - drop) * self.skip_beta) * merged

    def unit(self, index, x, shortcut, previous):
        """One FF unit on a detached input. Returns logits, ReLU output, the layer output s_l and the next input."""
        _, _, _, kind, _, _ = self.layout[index]
        activations = functional.relu(self.convs[index](x))
        logits = self.local_logits(index, activations)
        merged = merge(kind, self.layer_output(activations), shortcut)
        if self.method == "ttgs":
            output = self.stream(index, merged, previous)
            return logits, activations, output, self.dropout(output)
        return logits, activations, merged, merged

    def fusion_logits(self, features):
        """features: ReLU outputs (BP) or their GAP (FF, pooled at once so the full maps are not kept)."""
        return self.fusion(torch.cat([f.detach().mean(dim=(2, 3)) if f.dim() == 4 else f.detach()
                                      for f in features], dim=1))

    def ff_pass(self, images, labels=None, optimizers=None, average_gradients=None):
        """One FF pass over the 17 units; trains each unit when optimizers are given.
        Returns per-unit logits, per-unit losses (training only) and the GAP of each ReLU output."""
        x, shortcut, previous = self.prepare_input(images), None, None
        logits_all, losses, relus = [], [], []
        for index in range(len(self.convs)):
            if optimizers is not None:
                optimizers[index].zero_grad(set_to_none=True)
            logits, activations, output, x = self.unit(index, x.detach(), shortcut, previous)
            if optimizers is not None:
                loss = functional.cross_entropy(logits, labels)
                loss.backward()
                average_gradients(optimizers[index])
                optimizers[index].step()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss in layer {index + 1} at epoch {self.epoch + 1}")
                losses.append(loss.detach())
                if self.method == "ttgs" and not self.built and self.epoch == self.build_epoch - 1 \
                        and index == len(self.convs) - 1:
                    self.accumulate_tree(activations, labels)
            _, _, _, _, keep, pool = self.layout[index]
            output = output.detach()
            if keep:
                shortcut = functional.avg_pool2d(output, 2) if pool else output
            previous, x = output, x.detach()
            logits_all.append(logits.detach())
            relus.append(activations.detach().mean(dim=(2, 3)))
        return logits_all, losses, relus

    def forward(self, images, return_local=False):
        if self.method == "bp":
            head, fusion = self.forward_bp(images)
            return (fusion, [], head) if return_local else fusion
        logits_all, _, relus = self.ff_pass(images)
        fusion = self.fusion_logits(relus)
        return (fusion, logits_all) if return_local else fusion

    # ---- training (called from train.py) ----
    def build_optimizers(self, config):
        fusion = config["fusion"]
        if self.method == "bp":
            backbone = [p for n, p in self.named_parameters() if not n.startswith("fusion.")]
            sgd = torch.optim.SGD(backbone, lr=config["learning_rate"], momentum=config["momentum"],
                                  nesterov=config["nesterov"], weight_decay=config["weight_decay"])
            optimizers = [sgd, adamw(self.fusion.parameters(), fusion)]
            schedulers = [torch.optim.lr_scheduler.MultiStepLR(sgd, config["milestones"], config["gamma"])]
        else:
            optimizers = [adamw(list(conv.parameters()) + list(head.parameters()), config)
                          for conv, head in zip(self.convs, self.goodness)] + [adamw(self.fusion.parameters(), fusion)]
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, config["epochs"],
                                                                     config["minimum_learning_rate"])
                          for o in optimizers[:-1]]
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[-1], config["epochs"],
                                                                     fusion["minimum_learning_rate"]))
        return optimizers, schedulers

    def train_step(self, optimizers, images, labels, average_gradients):
        """Returns (Fusion logits, losses, correct) in train.py's layout: BP [head, Fusion] losses and
        [Fusion, head] correct; FF [unit 1..17, Fusion] losses and [Fusion, unit 1..17] correct."""
        if self.method == "bp":
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            head, fusion = self.forward_bp(images)
            fusion_loss = functional.cross_entropy(fusion, labels)
            head_loss = functional.cross_entropy(head, labels)
            (fusion_loss + head_loss).backward()  # Fusion inputs are detached: each loss reaches only its part
            for optimizer in optimizers:
                average_gradients(optimizer)
                optimizer.step()
            correct = torch.stack([(fusion.detach().argmax(1) == labels).sum(),
                                   (head.detach().argmax(1) == labels).sum()])
            return fusion.detach(), torch.stack([head_loss.detach(), fusion_loss.detach()]), correct
        logits_all, losses, relus = self.ff_pass(images, labels, optimizers[:-1], average_gradients)
        optimizers[-1].zero_grad(set_to_none=True)
        fusion = self.fusion_logits(relus)
        fusion_loss = functional.cross_entropy(fusion, labels)
        fusion_loss.backward()
        average_gradients(optimizers[-1])
        optimizers[-1].step()
        correct = torch.stack([(logits.argmax(1) == labels).sum() for logits in [fusion.detach(), *logits_all]])
        return fusion.detach(), torch.stack(losses + [fusion_loss.detach()]), correct

    def parameter_counts(self):
        counts = {"backbone_parameters": sum(p.numel() for n, p in self.named_parameters()
                                             if n.startswith(("convs.", "norms."))),
                  "fusion_parameters": sum(p.numel() for p in self.fusion.parameters()),
                  "fusion_features": len(self.convs)}
        if self.method == "bp":
            counts["head_parameters"] = sum(p.numel() for p in self.head.parameters())
        else:
            counts.update({"readout_parameters": sum(p.numel() for p in self.goodness.parameters()),
                           "fixed_readout_elements": sum(b.numel() for _, b in self.goodness.named_buffers()),
                           "goodness_dims": [head.projection.shape[0] for head in self.goodness],
                           "learning_units": len(self.convs)})
        return counts

    # ---- TTGS tree ----
    @torch.no_grad()
    def accumulate_tree(self, activations, labels):
        features = activations.detach().mean(dim=(2, 3)).double()
        onehot = functional.one_hot(labels, self.classes).double()
        sums, counts = onehot.t() @ features, onehot.sum(dim=0)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Multi-GPU: every rank adds the global increment, so checkpoints hold global sums.
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        self.tree_sums += sums
        self.tree_counts += counts

    @torch.no_grad()
    def build_tree(self):
        if bool((self.tree_counts == 0).any()):
            raise RuntimeError("Tree building needs every class in the accumulated epoch")
        means = (self.tree_sums / self.tree_counts[:, None]).cpu().numpy()
        tree = linkage(means - means.mean(axis=0), method="average", metric="cosine")
        merges = [(int(tree[i, 0]), int(tree[i, 1])) for i in range(self.classes - 1)]
        depth, side = tree_signs(merges, self.classes)
        dims = [head.projection.shape[0] for head in self.goodness]
        for head, dim, perm in zip(self.goodness, dims, unit_permutations(self.seed, dims), strict=True):
            head.projection.copy_(tree_code(dim, self.classes, side, depth, perm).to(torch.float32))
            head.bias.zero_()
        self.tree_merges.copy_(torch.tensor(merges))
        self.tree_built.fill_(True)
        self.built = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(json.dumps({"ttgs_tree_built_epoch": self.epoch + 1, "merges": merges}), flush=True)
