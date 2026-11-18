"""Frozen TTGS shallow layers + deep layers trained by global gradient through Fusion (experiment, tmp only).

Question: with the first 3 resolution groups of the trained TTGS CNN32-depth fixed, can the last 12 conv layers
(groups 4 and 5, L21-L32) find parameters that give a higher Fusion accuracy than TTGS's own deep layers (64.64)?

- Start: results/cifar100/32-depth-v2/TTGS/seed1/last.pt. L1-L20 keep the trained TTGS weights and never change.
- L21-L32 and the Fusion head are reset to the initial weights of the same seed (the weights TTGS-32 started from).
- Forward path unchanged: every layer is computed exactly as in TTGS (pooling, layer norm, skip stream with
  stochastic depth, dropout; training mode in training, eval mode in evaluation). The class tree / goodness heads of
  L21-L32 are not used.
- Only the training signal changes: L21-L32 and Fusion are trained together on the Fusion cross-entropy; the gradient
  flows through L21-L32 including the skip stream; it is cut between L20 and L21.
- Platform settings: CIFAR-100, seed 1, batch 128, AdamW 2e-4 -> 1e-5 cosine over the epochs, weight decay 1e-3.
Variant --deep-dropout P: the dropout applied to the outputs of L21-L32 (the inputs of L22-L32) is P instead of 0.1;
the frozen layers keep 0.1. Nothing else changes.
Usage (repo root): python tmp/deep_bp/run.py --epochs 400 --output tmp/deep_bp/results
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as functional

import evaluate
from data.datasets import image_datasets, make_loader
from methods.TTG import TTGS

SOURCE = ROOT / "results/cifar100/32-depth-v2/TTGS/seed1/last.pt"
FIRST_TRAINED = 20  # 0-based index of L21
DEEP_DROPOUT = None  # --deep-dropout: dropout rate on the outputs of L21-L32 (None = the TTGS rate, 0.1)


class TTGSDeep(TTGS):
    """TTGS.propagate, identical except that the stored stream keeps its graph from layer FIRST_TRAINED on."""

    def propagate(self, index, activations):
        if index in self.pool_after:
            activations = functional.avg_pool2d(activations, 2)
        u = functional.layer_norm(activations.permute(0, 2, 3, 1), (activations.shape[1],),
                                  eps=self.epsilon).permute(0, 3, 1, 2)
        prev = self._skip_prev
        if index == 0 or prev is None or self._skip_index != index - 1 or prev.shape != u.shape:
            stream = u
        else:
            drop = self.drop_max * (index + 1) / self.depth
            if self.training:
                stream = torch.where(self.keep_mask(u.shape[0], u.device, drop),
                                     self.skip_a * prev + self.skip_beta * u, prev)
            else:
                stream = (drop + (1.0 - drop) * self.skip_a) * prev + ((1.0 - drop) * self.skip_beta) * u
        self._skip_prev = stream if (index >= FIRST_TRAINED and torch.is_grad_enabled()) else stream.detach()
        self._skip_index = index
        if DEEP_DROPOUT is not None and index >= FIRST_TRAINED:
            return functional.dropout(stream, DEEP_DROPOUT, self.training)
        return self.dropout(stream)


def forward(model, images):
    method = model.method
    features = []
    with torch.no_grad():
        hidden = method.prepare_input(images)
        for index in range(FIRST_TRAINED):
            activations = functional.relu(model.convs[index](hidden))
            features.append(activations.mean(dim=(2, 3)))
            hidden = method.propagate(index, method.prepare_activations(index, activations))
    for index in range(FIRST_TRAINED, len(model.convs)):
        activations = functional.relu(model.convs[index](hidden))
        features.append(activations.mean(dim=(2, 3)))
        hidden = method.propagate(index, method.prepare_activations(index, activations))
    method._skip_prev = None
    return model.fusion(torch.cat(features, dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-dropout", type=float, default=None)
    parser.add_argument("--head-only", action="store_true", help="train only the Fusion head; L21-L32 stay fixed too")
    parser.add_argument("--keep-weights", action="store_true",
                        help="start L21-L32 and Fusion from the TTGS weights instead of the seed's initial weights")
    args = parser.parse_args()
    global DEEP_DROPOUT
    DEEP_DROPOUT = args.deep_dropout
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    evaluate.TTGS = TTGSDeep
    torch.manual_seed(config["seed"])
    model = evaluate.build_model(config)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(checkpoint["model"])
    reset = [key for key in initial if key.startswith("fusion.") or
             (key.startswith("convs.") and int(key.split(".")[1]) >= FIRST_TRAINED)]
    if args.keep_weights:
        reset = []
    model.load_state_dict({**model.state_dict(), **{key: initial[key] for key in reset}})
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trained = [p for conv in model.convs[FIRST_TRAINED:] for p in conv.parameters()] + list(model.fusion.parameters())
    if args.head_only:
        trained = list(model.fusion.parameters())
    for parameter in trained:
        parameter.requires_grad_(True)
    frozen_before = torch.cat([p.detach().flatten() for conv in model.convs[:FIRST_TRAINED]
                               for p in conv.parameters()]).clone()
    optimizer = torch.optim.AdamW(trained, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                  betas=tuple(config["betas"]), eps=config["optimizer_epsilon"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=config["minimum_learning_rate"])
    training, testing = image_datasets(config, ROOT / "data/cifar")
    train_loader = make_loader(training, 128, config["seed"], 4, True, device)
    test_loader = make_loader(testing, 500, config["seed"], 4, False, device)
    run_config = {**config, "experiment": "frozen_ttgs_L1-20_plus_bp_through_fusion_L21-32", "epochs": args.epochs,
                  "batch_size": 128, "deep_dropout": args.deep_dropout, "head_only": args.head_only, "source_checkpoint": str(SOURCE.relative_to(ROOT)), "reset": reset}
    (args.output / "config.json").write_text(json.dumps(run_config, indent=2))
    history, seconds = [], 0.0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        model.method.epoch = epoch
        started = time.perf_counter()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = forward(model, images)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
        scheduler.step()
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - started
        seconds += epoch_seconds
        model.eval()
        hit = count = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = forward(model, images)
                test_loss += float(functional.cross_entropy(logits, labels, reduction="sum"))
                hit += int((logits.argmax(1) == labels).sum())
                count += len(labels)
        row = {"epoch": epoch + 1, "train/fusion_accuracy": 100.0 * correct / total,
               "train/fusion_loss": loss_sum / total, "test/fusion_accuracy": 100.0 * hit / count,
               "test/fusion_loss": test_loss / count, "time/epoch_seconds": epoch_seconds,
               "train/lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(history))
    frozen_after = torch.cat([p.detach().flatten() for conv in model.convs[:FIRST_TRAINED] for p in conv.parameters()])
    result = {"config": run_config, "metrics": {"fusion_accuracy": history[-1]["test/fusion_accuracy"],
                                                "train_fusion_accuracy": history[-1]["train/fusion_accuracy"]},
              "training": {"seconds": seconds, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                           "device": torch.cuda.get_device_name(device)},
              "frozen_layers_unchanged": bool(torch.equal(frozen_before, frozen_after))}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"config": run_config, "epoch": args.epochs, "model": model.state_dict()}, args.output / "last.pt")
    print(json.dumps({"done": True, **result["metrics"], "frozen_layers_unchanged": result["frozen_layers_unchanged"],
                      "hours": seconds / 3600}), flush=True)


if __name__ == "__main__":
    main()
