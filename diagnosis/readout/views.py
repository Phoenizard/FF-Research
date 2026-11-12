"""Offline readout views on 16-layer models (seed 1, CIFAR-100 and TinyIN). Test set only, no training.
Run from the repo root: python -m diagnosis.readout.views

Views per model (all from last.pt, eval mode, no augmentation):
- fusion: the trained Fusion head.
- best_layer: best single local readout and its layer (picked on the test set, reference only).
- last_layer: local readout of layer 16. drop = best_layer - last_layer.
- softmax_all / softmax_first8: mean of per-layer softmax(local logits) over layers 1..16 / 1..8 (zero parameters).
- sip: summed local logits over the best interval [S,E], all 136 intervals (picked on the test set, reference only;
  ties -> shorter, then deeper).
- fusion_share: per-layer share of Fusion logit variance, Var(f_l W_l^T) / sum_l Var(f_l W_l^T) on the test set.
"""
import json
from pathlib import Path

import torch

from data.datasets import image_datasets, make_loader
from evaluate import build_model

METHODS = ["ASGE", "DTG-FF", "CwC", "DeeperForward", "HCL-FF", "TTG", "TTGS"]
DATASETS = ["cifar100", "tinyimagenet"]
OUT = Path(__file__).resolve().parent


@torch.inference_mode()
def extract(model, loader, device):
    model.eval()
    captured, fusion, local, feats, labels = {}, [], [], [], []
    original = model.predict

    def hook(features):
        captured["f"] = [f.detach() for f in features]
        return original(features)

    model.predict = hook
    for images, y in loader:
        logits, local_logits = model(images.to(device, non_blocking=True), return_local=True)
        fusion.append(logits.float().cpu())
        local.append(torch.stack([l.float() for l in local_logits], 1).cpu())
        feats.append([f.float().cpu() for f in captured["f"]])
        labels.append(y)
    model.predict = original
    return (torch.cat(fusion), torch.cat(local), [torch.cat(parts) for parts in zip(*feats)], torch.cat(labels))


def accuracy(scores, y):
    return float((scores.argmax(1) == y).double().mean() * 100)


def analyse(dataset, method, device):
    run = Path("results") / dataset / "16-depth" / method / "seed1"
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    assert checkpoint["epoch"] == config["epochs"] == 400
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    _, testing = image_datasets(config)
    fusion, local, feats, y = extract(model, make_loader(testing, 500, 0, 4, False, device), device)
    depth = local.shape[1]
    layer_acc = [accuracy(local[:, l], y) for l in range(depth)]
    best = max(range(depth), key=lambda l: (layer_acc[l], l))
    probs = local.softmax(2)
    cum = torch.cat([torch.zeros_like(local[:, :1]), local.cumsum(1)], 1).to(device)
    yd = y.to(device)
    sip = {(s, e): float(((cum[:, e] - cum[:, s - 1]).argmax(1) == yd).double().mean() * 100)
           for s in range(1, depth + 1) for e in range(s, depth + 1)}
    s, e = max(sip, key=lambda k: (sip[k], -(k[1] - k[0]), k[1]))
    weight = model.fusion.weight.detach().float().cpu()
    widths = [f.shape[1] for f in feats]
    blocks = torch.split(weight, widths, dim=1)
    variance = [float((f @ w.T).var(0).sum()) for f, w in zip(feats, blocks)]
    history = json.loads((run / "history.json").read_text())[-1]
    check = {"fusion_diff": abs(accuracy(fusion, y) - history["test/fusion_accuracy"]),
             "local_max_diff": max(abs(a - history[f"test/layer_{l + 1}_accuracy"]) for l, a in enumerate(layer_acc))}
    result = {"fusion": accuracy(fusion, y), "best_layer": layer_acc[best], "best_layer_index": best + 1,
              "last_layer": layer_acc[-1], "drop": layer_acc[best] - layer_acc[-1],
              "softmax_all": accuracy(probs.mean(1), y), "softmax_first8": accuracy(probs[:, :8].mean(1), y),
              "sip": sip[(s, e)], "sip_interval": [s, e],
              "fusion_share": [v / sum(variance) for v in variance], "layer_accuracy": layer_acc, "check": check}
    print(json.dumps({"dataset": dataset, "method": method,
                      **{k: v for k, v in result.items() if k not in ("fusion_share", "layer_accuracy")}}), flush=True)
    return result


def main():
    device = torch.device("cuda")
    results = {d: {m: analyse(d, m, device) for m in METHODS} for d in DATASETS}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
