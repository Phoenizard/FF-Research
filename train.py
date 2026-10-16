import argparse
import gc
import json
import platform
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchvision
from torch.nn import functional as functional

from data.datasets import cifar_datasets, evaluation_dataset, make_loader
from evaluate import (ROOT, accuracy_metrics, build_model, evaluate, parameter_counts,
                      synchronize, write_json, write_result)
from models.configs import DATASETS, experiment_config


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizers(model, config):
    parameter_groups = [conv.parameters() for conv in model.convs] + [model.fusion.parameters()]
    settings = [config] * len(model.convs) + [config["fusion"]]
    optimizers = [torch.optim.AdamW(parameters, lr=setting["learning_rate"],
                                   weight_decay=setting["weight_decay"], betas=tuple(setting["betas"]),
                                   eps=setting["optimizer_epsilon"])
                  for parameters, setting in zip(parameter_groups, settings, strict=True)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=setting["minimum_learning_rate"])
        for optimizer, setting in zip(optimizers, settings, strict=True)]
    return optimizers, schedulers


def train_step(model, optimizers, images, labels):
    hidden = model.method.prepare_input(images).detach()
    prediction_features = []
    local_losses = []
    local_correct = []
    for index, (conv, optimizer) in enumerate(zip(model.convs, optimizers[:-1], strict=True)):
        optimizer.zero_grad(set_to_none=True)
        activations = functional.relu(conv(hidden.detach()))
        loss, local_logits = model.method.local_loss(index, activations, labels)
        loss.backward()
        optimizer.step()
        local_losses.append(loss.detach())
        with torch.no_grad():
            local_correct.append((local_logits.argmax(dim=1) == labels).sum())
            prediction_features.append(model.fusion_features(activations))
            hidden = model.method.propagate(index, activations).detach()
    optimizers[-1].zero_grad(set_to_none=True)
    logits = model.predict(prediction_features)
    prediction_loss = functional.cross_entropy(logits, labels)
    prediction_loss.backward()
    optimizers[-1].step()
    correct = torch.stack([(logits.detach().argmax(dim=1) == labels).sum(), *local_correct])
    return logits.detach(), torch.stack(local_losses + [prediction_loss.detach()]), correct


def probe_batch(config, device, requested):
    candidate = 128 if requested == "auto" else int(requested)
    if candidate < 1:
        raise ValueError("batch_size must be positive")
    while candidate >= 1:
        seed_run(config["seed"])
        model = build_model(config).to(device)
        optimizers, _ = build_optimizers(model, config)
        images = labels = logits = losses = None
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            images = torch.randn(candidate, 3, 32, 32, device=device)
            labels = torch.arange(candidate, device=device) % config["num_classes"]
            model.train()
            synchronize(device)
            started = time.perf_counter()
            for _ in range(3):
                logits, losses, _ = train_step(model, optimizers, images, labels)
                if not torch.isfinite(losses).all():
                    raise RuntimeError("Non-finite loss during capacity check")
            model.eval()
            with torch.no_grad():
                logits = model(images)
            synchronize(device)
            return {"dataset": config["dataset"], "batch_size": candidate,
                    "three_training_steps_and_eval_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else None}
        except torch.cuda.OutOfMemoryError:
            if requested != "auto":
                raise
            print(f"Capacity check: batch_size={candidate} does not fit; halving.", flush=True)
            candidate //= 2
        finally:
            del model, optimizers, images, labels, logits, losses
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("Device cannot fit batch_size=1")


def random_state(loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy_algorithm": numpy_state[0],
            "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
            "numpy_tail": numpy_state[2:], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def restore_random_state(state, loader):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_algorithm"], state["numpy_keys"].numpy(), *state["numpy_tail"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def save_checkpoint(path, config, epoch, model, optimizers, schedulers, loader, training, history):
    temporary = path.with_suffix(".tmp")
    torch.save({"config": config, "epoch": epoch, "model": model.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                "random_state": random_state(loader), "training": training, "history": history}, temporary)
    temporary.replace(path)


def tracking_run(config, args, run_dir, training, counts):
    if config["run_kind"] == "smoke-test" or args.wandb_mode == "disabled":
        return nullcontext(None)
    import wandb

    previous = training.get("wandb")
    metadata = previous or {"id": uuid4().hex[:12], "entity": args.wandb_entity,
                            "project": args.wandb_project}
    tracker = wandb.init(
        **metadata, resume="must" if previous else "never", mode="online",
        name=f"{config['method']}-{config['dataset']}-seed{config['seed']}",
        group=f"{config['method']}-{config['dataset']}", job_type="train",
        config={**config, "environment": {key: training[key] for key in
                ("python", "torch", "torchvision", "device", "workers", "cpu_threads")},
                "parameters": counts}, dir=str(run_dir), save_code=False,
    )
    training["wandb"] = metadata
    write_json(run_dir / "wandb.json", {**metadata, "url": tracker.url})
    tracker.define_metric("epoch")
    for prefix in ("train/*", "test/*", "time/*", "memory/*"):
        tracker.define_metric(prefix, step_metric="epoch")
    return tracker


def run(config, args, device, checkpoint=None):
    seed_run(config["seed"])
    run_dir = args.resume.parent if checkpoint else (
        args.output / config["dataset"] / config["method"] / f"seed{config['seed']}")
    if config["run_kind"] == "smoke-test" and not run_dir.resolve().is_relative_to(ROOT / "tmp"):
        raise ValueError("Smoke Test outputs must stay inside the project tmp directory")
    if not checkpoint and run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}; use --resume or another --output")
    training_data, testing_data = cifar_datasets(config, args.data_root, args.download)
    testing_data = evaluation_dataset(config, training_data, testing_data)
    training_loader = make_loader(training_data, config["batch_size"], config["seed"],
                                  args.workers, True, device)
    testing_loader = make_loader(testing_data, config["batch_size"], config["seed"],
                                 args.workers, False, device)
    model = build_model(config).to(device)
    optimizers, schedulers = build_optimizers(model, config)
    training = {"seconds": 0.0, "peak_memory_bytes": None, "steps_per_layer": 0,
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "python": platform.python_version(), "cpu_threads": torch.get_num_threads(),
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "workers": args.workers}
    history = []
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, checkpoint["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        restore_random_state(checkpoint["random_state"], training_loader)
        training = checkpoint["training"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        del checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    counts = parameter_counts(model)
    evaluation_metrics = None
    evaluation_prefix = "smoke" if config["run_kind"] == "smoke-test" else "test"
    tracking_state = random_state(training_loader)
    with tracking_run(config, args, run_dir, training, counts) as tracker:
        restore_random_state(tracking_state, training_loader)
        save_checkpoint(run_dir / "last.pt", config, start_epoch, model, optimizers,
                        schedulers, training_loader, training, history)
        for epoch in range(start_epoch, config["epochs"]):
            model.train()
            total = 0
            correct = torch.zeros(len(model.convs) + 1, dtype=torch.long, device=device)
            loss_sum = torch.zeros(len(model.convs) + 1, device=device)
            learning_rate = optimizers[0].param_groups[0]["lr"]
            fusion_learning_rate = optimizers[-1].param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            started = time.perf_counter()
            for images, labels in training_loader:
                images, labels = images.to(device), labels.to(device)
                logits, losses, batch_correct = train_step(model, optimizers, images, labels)
                if not torch.isfinite(losses).all():
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
                loss_sum += losses * labels.numel()
                correct += batch_correct
                total += labels.numel()
                training["steps_per_layer"] += 1
            synchronize(device)
            elapsed = time.perf_counter() - started
            training["seconds"] += elapsed
            peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            if peak_memory is not None:
                training["peak_memory_bytes"] = max(training["peak_memory_bytes"] or 0, peak_memory)
            mean_losses = (loss_sum / total).tolist()
            metrics = {"epoch": epoch + 1,
                       **{f"train/{key}": value for key, value in accuracy_metrics(correct, total).items()},
                       **{f"train/loss_layer_{index}": value
                          for index, value in enumerate(mean_losses[:-1], start=1)},
                       "train/fusion_loss": mean_losses[-1], "train/lr": learning_rate,
                       "train/fusion_lr": fusion_learning_rate, "train/samples": total,
                       "time/epoch_seconds": elapsed,
                       "memory/peak_allocated_mib": peak_memory / 1024 ** 2
                       if peak_memory is not None else None}
            evaluation_metrics = evaluate(model, testing_loader, device)
            metrics.update({f"{evaluation_prefix}/{key}": value
                            for key, value in evaluation_metrics.items()})
            history.append(metrics)
            for scheduler in schedulers:
                scheduler.step()
            save_checkpoint(run_dir / "last.pt", config, epoch + 1, model, optimizers,
                            schedulers, training_loader, training, history)
            write_json(run_dir / "history.json", history)
            if tracker is not None:
                tracker.log(metrics, step=epoch + 1)
            print(json.dumps({"dataset": config["dataset"], "seed": config["seed"], **metrics}), flush=True)
        if evaluation_metrics is None:
            evaluation_metrics = evaluate(model, testing_loader, device)
        result = write_result(run_dir, config, model, evaluation_metrics, training, device)
        if tracker is not None:
            tracker.summary.update({**{f"test/{key}": value for key, value in evaluation_metrics.items()},
                                    "completed_epochs": config["epochs"],
                                    "time/training_seconds": training["seconds"],
                                    "memory/training_peak_allocated_mib": training["peak_memory_bytes"] / 1024 ** 2
                                    if training["peak_memory_bytes"] is not None else None})
        print(json.dumps({"run": str(run_dir), "evaluation_scope": config["evaluation_scope"],
                          **evaluation_metrics}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train ASGE on the common CIFAR backbone.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["cifar10", "cifar100"])
    parser.add_argument("--method", choices=["asge"], default="asge")
    parser.add_argument("--seeds", nargs="+", type=int, choices=[1, 2, 3])
    parser.add_argument("--batch-size", default="128")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wandb-mode", choices=["online", "disabled"], default="online")
    parser.add_argument("--wandb-entity", default="pheonizard-university-of-nottingham")
    parser.add_argument("--wandb-project", default="FF-Research")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "cifar")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--prepare-only", action="store_true")
    operation.add_argument("--probe-only", action="store_true")
    operation.add_argument("--resume", type=Path)
    operation.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    args.seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    if args.smoke_test:
        args.wandb_mode = "disabled"
        if args.output == ROOT / "results":
            args.output = ROOT / "tmp" / "smoke" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if not args.output.resolve().is_relative_to(ROOT / "tmp"):
            parser.error("--smoke-test output must be inside the project tmp directory")
    if args.prepare_only:
        for dataset in args.datasets:
            config = experiment_config(dataset, 1)
            training, testing = cifar_datasets(config, args.data_root, args.download)
            print(json.dumps({"dataset": dataset, "train": len(training), "test": len(testing),
                              "normalization": config["normalization"]}))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in GPU-enabled WSL")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        run(checkpoint["config"], args, device, checkpoint)
        return
    for dataset in args.datasets:
        config = experiment_config(dataset, args.seeds[0])
        capacity = probe_batch(config, device, args.batch_size)
        print(json.dumps(capacity), flush=True)
        if args.probe_only:
            probe_dir = ROOT / "tmp" / "smoke" / "capacity"
            probe_dir.mkdir(parents=True, exist_ok=True)
            write_json(probe_dir / f"{dataset}_capacity.json", capacity)
            continue
        for seed in args.seeds:
            config = experiment_config(dataset, seed)
            config["batch_size"] = capacity["batch_size"]
            if args.smoke_test:
                config.update(epochs=2, run_kind="smoke-test", evaluation_scope="64_training_images")
            run(config, args, device)


if __name__ == "__main__":
    main()
