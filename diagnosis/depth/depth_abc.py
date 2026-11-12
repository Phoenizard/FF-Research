"""Depth-degradation A/B/C diagnosis on CIFAR-100, seed 1. Run from the repo root: python -m diagnosis.depth.depth_abc

Criteria were fixed before the BP-16 result existed.
- F_W: mean test/fusion_accuracy over epochs 381-400 (history.json).
- Linear probe: final checkpoint in eval mode, no augmentation, per-layer GAP features of all 50k train and 10k test
  images. Features are standardized with train mean/std (std floor 1e-6); multinomial logistic regression with L2
  1e-4, full-batch L-BFGS (strong Wolfe), at most 500 iterations, float32, zero init. f_l is the post-ReLU feature
  used by Fusion; for TTGS, s_l is the propagated stream (expected coefficients in eval).
- A (backbone ceiling): dF = F_W(BP-16) - F_W(BP-8); G = max probe test(f_13..16) - max probe test(f_1..12) on BP-16.
  Holds if both < 1.0, fails if both >= 1.0, otherwise partly holds.
- B/C (TTGS-16 tail): d_tr = probe train(s_16) - probe train(s_12); d_te = probe test(s_16) - probe test(s_12).
  B (tail only copies the stream): both < 1.0. C (new information only on train): d_tr >= 1.0 and d_te < 1.0.
  Neither: d_te >= 1.0.
- Threshold 1.0 follows the seed-to-seed std of Fusion (about 0.98); the probe itself is deterministic.
"""
import json
from collections import defaultdict
from copy import copy
from pathlib import Path

import torch
from torch.nn import functional as functional

from data.datasets import image_datasets, make_loader
from evaluate import build_model

RUNS = {"BP-8": "results/cifar100/8-depth/Global-BP/seed1",
        "BP-16": "results/cifar100/16-depth/Global-BP/seed1",
        "TTG-16": "results/cifar100/16-depth/TTG/seed1",
        "TTGS-16": "results/cifar100/16-depth/TTGS/seed1"}
OUT = Path(__file__).resolve().parent
WINDOW = range(381, 401)
L2 = 1e-4
THRESHOLD = 1.0


@torch.no_grad()
def extract(model, loader, device, stream):
    model.eval()
    method = model.method
    out = defaultdict(list)
    for images, labels in loader:
        hidden = method.prepare_input(images.to(device, non_blocking=True))
        for index, conv in enumerate(model.convs):
            activations = functional.relu(conv(hidden))
            out[f"f{index + 1}"].append(activations.mean(dim=(2, 3)).cpu())
            hidden = method.propagate(index, method.prepare_activations(index, activations))
            if stream:
                out[f"s{index + 1}"].append(hidden.mean(dim=(2, 3)).cpu())
        out["y"].append(labels)
    return {key: torch.cat(value) for key, value in out.items()}


def probe(train_x, train_y, test_x, test_y, classes, device):
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    train_x, test_x = ((train_x - mean) / std).to(device), ((test_x - mean) / std).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    weight = torch.zeros(train_x.shape[1], classes, device=device, requires_grad=True)
    bias = torch.zeros(classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=500, history_size=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = functional.cross_entropy(train_x @ weight + bias, train_y) + L2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        accuracy = lambda x, y: float(((x @ weight + bias).argmax(1) == y).double().mean() * 100)
        return accuracy(train_x, train_y), accuracy(test_x, test_y)


def analyse(name, path, device):
    run = Path(path)
    history = {row["epoch"]: row for row in json.loads((run / "history.json").read_text())}
    if sorted(history) != list(range(1, 401)):
        raise ValueError(f"{name}: history epochs are not 1..400")
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["epoch"] != config["epochs"]:
        raise ValueError(f"{name}: checkpoint is not the final epoch")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    training, testing = image_datasets(config, Path("data/cifar"))
    training = copy(training)
    training.transform = testing.transform
    stream = config["method"] == "ttgs"
    feats = {split: extract(model, make_loader(data, 500, 0, 4, False, device), device, stream)
             for split, data in (("train", training), ("test", testing))}
    result = {"F_W": sum(history[e]["test/fusion_accuracy"] for e in WINDOW) / len(WINDOW), "probe": {}}
    for kind in ("f", "s") if stream else ("f",):
        for layer in range(1, len(config["channels"]) + 1):
            key = f"{kind}{layer}"
            train_acc, test_acc = probe(feats["train"][key], feats["train"]["y"], feats["test"][key],
                                        feats["test"]["y"], config["num_classes"], device)
            result["probe"][key] = {"train": train_acc, "test": test_acc}
            print(json.dumps({"run": name, "feature": key, "train": train_acc, "test": test_acc}), flush=True)
    return result


def verdicts(results):
    bp16 = results["BP-16"]["probe"]
    delta_f = results["BP-16"]["F_W"] - results["BP-8"]["F_W"]
    gain = max(bp16[f"f{l}"]["test"] for l in range(13, 17)) - max(bp16[f"f{l}"]["test"] for l in range(1, 13))
    a = ("holds" if delta_f < THRESHOLD and gain < THRESHOLD else
         "fails" if delta_f >= THRESHOLD and gain >= THRESHOLD else "partly holds")
    ttgs = results["TTGS-16"]["probe"]
    d_train = ttgs["s16"]["train"] - ttgs["s12"]["train"]
    d_test = ttgs["s16"]["test"] - ttgs["s12"]["test"]
    bc = ("B" if d_train < THRESHOLD and d_test < THRESHOLD else
          "C" if d_train >= THRESHOLD and d_test < THRESHOLD else "neither")
    return {"A": {"delta_F_BP": delta_f, "G_BP": gain, "verdict": a},
            "BC": {"delta_train": d_train, "delta_test": d_test, "verdict": bc}}


def main():
    device = torch.device("cuda")
    results = {name: analyse(name, path, device) for name, path in RUNS.items()}
    results["verdicts"] = verdicts(results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
