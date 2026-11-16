"""Readout views (same definitions as diagnosis/readout/views.py) on CIFAR-100 CNN32 runs, seed 1, test set.
'first' = layers 1..21 (all layers before the 400-channel stage), the CNN32 counterpart of the 16-layer 'first 8'.
Run from the repo root: python -m diagnosis.readout.views32"""
import json
from pathlib import Path

import torch
from data.datasets import image_datasets, make_loader
from evaluate import build_model
from diagnosis.readout.views import extract, accuracy

RUNS = ["BP-Residual", "TTGS-Residual", "TTGS", "TTG-Residual", "TTG"]
FIRST = 21

def analyse(name, device):
    run = Path("results/cifar100/32-depth") / name / "seed1"
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    config = ck["config"]
    model = build_model(config).to(device); model.load_state_dict(ck["model"])
    _, testing = image_datasets(config)
    res = json.loads((run / "result.json").read_text())
    hist = json.loads((run / "history.json").read_text())[-1]
    out = {"train_peak_mib": res["training"]["peak_memory_bytes"] / 2**20, "train_device": res["training"]["device"]}
    if config.get("global_bp"):
        model.eval()
        with torch.inference_mode():
            correct = n = 0
            for x, y in make_loader(testing, 500, 0, 4, False, device):
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
        out.update(fusion=100 * correct / n, check={"fusion_diff": abs(100 * correct / n - hist["test/fusion_accuracy"])})
        print(json.dumps({"run": name, **out}), flush=True); return out
    fusion, local, feats, y = extract(model, make_loader(testing, 500, 0, 4, False, device), device)
    depth = local.shape[1]
    acc = [accuracy(local[:, l], y) for l in range(depth)]
    best = max(range(depth), key=lambda l: (acc[l], l))
    probs = local.softmax(2)
    cum = torch.cat([torch.zeros_like(local[:, :1]), local.cumsum(1)], 1).to(device); yd = y.to(device)
    sip = {(s, e): float(((cum[:, e] - cum[:, s - 1]).argmax(1) == yd).double().mean() * 100)
           for s in range(1, depth + 1) for e in range(s, depth + 1)}
    s, e = max(sip, key=lambda k: (sip[k], -(k[1] - k[0]), k[1]))
    out.update(fusion=accuracy(fusion, y), best_layer=acc[best], best_layer_index=best + 1, last_layer=acc[-1],
               drop=acc[best] - acc[-1], softmax_all=accuracy(probs.mean(1), y),
               softmax_first21=accuracy(probs[:, :FIRST].mean(1), y), sip=sip[(s, e)], sip_interval=[s, e],
               layer_accuracy=acc,
               check={"fusion_diff": abs(accuracy(fusion, y) - hist["test/fusion_accuracy"]),
                      "local_max_diff": max(abs(a - hist[f"test/layer_{l + 1}_accuracy"]) for l, a in enumerate(acc))})
    print(json.dumps({"run": name, **{k: v for k, v in out.items() if k != "layer_accuracy"}}), flush=True)
    return out

def main():
    device = torch.device("cuda")
    results = {r: analyse(r, device) for r in RUNS}
    Path(__file__).resolve().with_name("results_cnn32.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
