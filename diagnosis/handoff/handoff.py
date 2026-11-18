"""How learnable is the tensor a segment receives? (diagnostic only, not a method; frozen backbones, CIFAR-100, seed 1)
Run from the repo root: conda run --no-capture-output -n gpu python -m tmp.depth_diag.input_probe

For a frozen final checkpoint and a cut layer c (the last layer of a segment), the tensor that layer c+1 receives
(method.propagate output of layer c, eval mode: no dropout, expected stream coefficients) is the input of one fixed
small network trained by backprop: Conv3x3(C->256)-BN-ReLU-Conv3x3(256->256)-BN-ReLU-GAP-Linear(100).
Training: 20 epochs, the run's own train augmentation on the images, batch 128, AdamW lr 1e-3 -> 0 cosine, wd 1e-3,
torch seed 0. Reported: train accuracy of the last epoch (augmented images) and test accuracy after the last epoch.
The same network on matched cuts of the 16- and 32-layer backbones (same channels and resolution) says whether the
longer segment hands a less usable tensor to the next segment.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

from data.datasets import image_datasets, make_loader
from evaluate import build_model

RUNS = {"TTGS-16": "results/cifar100/16-depth/TTGS/seed1", "TTGS-32": "results/cifar100/32-depth-v2/TTGS/seed1",
        "TTG-16": "results/cifar100/16-depth/TTG/seed1", "TTG-32": "results/cifar100/32-depth-v2/TTG/seed1",
        "BP-16": "results/cifar100/16-depth/Global-BP/seed1", "BP-32": "results/cifar100/32-depth-v2/Global-BP/seed1"}
# (label, 16-layer cut, 32-layer cut): end of segment 1, 2, 3
CUTS = (("after segment 1 (100ch 16x16)", 4, 8), ("after segment 2 (200ch 8x8)", 8, 16),
        ("after segment 3 (400ch 4x4)", 10, 20))
ARMS = [(name, label, cut16 if name.endswith("16") else cut32)
        for label, cut16, cut32 in CUTS for name in ("TTGS-16", "TTGS-32", "TTG-16", "TTG-32", "BP-16", "BP-32")]
OUT = Path(__file__).resolve().parent
EPOCHS = 20


@torch.no_grad()
def cut_tensor(model, images, cut):
    method = model.method
    hidden = method.prepare_input(images)
    for index in range(cut):
        activations = functional.relu(model.convs[index](hidden))
        hidden = method.propagate(index, method.prepare_activations(index, activations))
    return hidden


def run_arm(name, cut, device):
    checkpoint = torch.load(Path(RUNS[name]) / "last.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    training, testing = image_datasets(config, Path("data/cifar"))
    train_loader = make_loader(training, 128, 0, 4, True, device)
    test_loader = make_loader(testing, 500, 0, 4, False, device)
    channels = config["channels"][cut - 1]
    torch.manual_seed(0)
    head = nn.Sequential(nn.Conv2d(channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, config["num_classes"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))
    for epoch in range(EPOCHS):
        head.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = head(cut_tensor(model, images, cut))
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    head.eval()
    hit = count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            hit += int((head(cut_tensor(model, images, cut)).argmax(1) == labels).sum())
            count += len(labels)
    return {"train": 100.0 * correct / total, "test": 100.0 * hit / count}


def main():
    device = torch.device("cuda")
    results = []
    for name, label, cut in ARMS:
        row = {"run": name, "cut": label, "layer": cut, **run_arm(name, cut, device)}
        results.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "input_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
