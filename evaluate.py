import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as functional

from data.datasets import cifar_datasets, evaluation_dataset, make_loader
from methods.asge import ASGE
from models.cnn import LocalCNN


ROOT = Path(__file__).resolve().parent


def build_model(config):
    if config["method"] != "asge":
        raise ValueError(f"Unsupported method: {config['method']}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        return LocalCNN(config, ASGE(config))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def accuracy_metrics(correct, total):
    accuracies = (100 * correct.double() / total).tolist()
    best_layer = max(range(1, len(accuracies)), key=accuracies.__getitem__)
    return {"fusion_accuracy": accuracies[0],
            **{f"layer_{index}_accuracy": accuracy
               for index, accuracy in enumerate(accuracies[1:], start=1)},
            "best_pred_accuracy": accuracies[best_layer], "best_pred_layer": best_layer}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = torch.zeros(len(model.convs) + 1, dtype=torch.long, device=device)
    loss_sum = torch.zeros((), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, local_logits = model(images, return_local=True)
        loss_sum += functional.cross_entropy(logits, labels, reduction="sum")
        correct += torch.stack([(prediction.argmax(dim=1) == labels).sum()
                                for prediction in [logits, *local_logits]])
        total += labels.numel()
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {**accuracy_metrics(correct, total), "fusion_loss": loss_sum.item() / total,
            "samples": total, "evaluation_seconds": elapsed,
            "evaluation_peak_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None}


def parameter_counts(model):
    return {"backbone_parameters": sum(parameter.numel() for parameter in model.convs.parameters()),
            "auxiliary_parameters": sum(parameter.numel() for parameter in model.method.parameters())
            + sum(parameter.numel() for parameter in model.fusion.parameters()),
            "fusion_parameters": sum(parameter.numel() for parameter in model.fusion.parameters()),
            "fixed_projection_elements": sum(buffer.numel() for buffer in model.method.buffers())}


def write_result(run_dir, config, model, metrics, training, device):
    result = {"config": config, "training": training, "metrics": metrics,
              "parameters": parameter_counts(model), "evaluation_device": str(device)}
    write_json(Path(run_dir) / "result.json", result)
    return result


def summarize(run_dirs, output):
    groups = defaultdict(list)
    for run_dir in run_dirs:
        result_path = Path(run_dir) / "result.json"
        if not result_path.exists():
            print(f"Missing result: {result_path}")
            continue
        result = json.loads(result_path.read_text())
        config = dict(result["config"])
        if config["run_kind"] == "smoke-test" and not Path(output).resolve().is_relative_to(ROOT / "tmp"):
            raise ValueError("Smoke Test summaries must stay inside the project tmp directory")
        seed = config.pop("seed")
        key = json.dumps(config, sort_keys=True)
        if any(entry[0] == seed for entry in groups[key]):
            raise ValueError(f"Duplicate seed {seed}: {run_dir}")
        groups[key].append((seed, result))
    rows = []
    for key, entries in groups.items():
        config = json.loads(key)
        accuracies = [result["metrics"]["fusion_accuracy"] for _, result in entries]
        best_predictions = [result["metrics"]["best_pred_accuracy"] for _, result in entries]
        rows.append({"dataset": config["dataset"], "method": config["method"],
                     "seeds": ",".join(str(seed) for seed, _ in sorted(entries)),
                     "status": "complete" if sorted(seed for seed, _ in entries) == [1, 2, 3]
                     else "incomplete",
                     "run_kind": config["run_kind"],
                     "evaluation_scope": config["evaluation_scope"],
                     "fusion_accuracy_mean": statistics.mean(accuracies),
                     "fusion_accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
                     "best_pred_accuracy_mean": statistics.mean(best_predictions),
                     "best_pred_accuracy_std": statistics.stdev(best_predictions)
                     if len(best_predictions) > 1 else None,
                     "batch_size": config["batch_size"], "epochs": config["epochs"],
                     "training_hours_mean": statistics.mean(
                         result["training"]["seconds"] / 3600 for _, result in entries),
                     "training_peak_memory_bytes": max(
                         (result["training"]["peak_memory_bytes"] or 0) for _, result in entries),
                     **entries[0][1]["parameters"], "prediction": config["prediction"]})
    if not rows:
        raise ValueError("No completed evaluations to summarize")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate final FF checkpoints or summarize runs.")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--checkpoint", type=Path)
    operation.add_argument("--summary", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "cifar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark.csv")
    args = parser.parse_args()
    if args.summary:
        summarize(args.summary, args.output)
        return
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["epoch"] != config["epochs"]:
        raise ValueError("Benchmark evaluation requires the final epoch checkpoint")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    training, testing = cifar_datasets(config, args.data_root)
    testing = evaluation_dataset(config, training, testing)
    loader = make_loader(testing, config["batch_size"], config["seed"], args.workers, False, device)
    metrics = evaluate(model, loader, device)
    write_result(args.checkpoint.parent, config, model, metrics, checkpoint["training"], device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
