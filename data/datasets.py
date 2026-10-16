import random
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def cifar_datasets(config, root, download=False):
    dataset_type = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[config["dataset"]]
    training = dataset_type(root=root, train=True, download=download)
    testing = dataset_type(root=root, train=False, download=download)
    if config["normalization"] is None:
        pixels = training.data
        config["normalization"] = {
            "mean": [float(pixels[..., channel].mean() / 255) for channel in range(3)],
            "std": [float(pixels[..., channel].std() / 255) for channel in range(3)],
        }
    normalization = transforms.Normalize(**config["normalization"])
    augmentation = config["augmentation"]
    training.transform = transforms.Compose([
        transforms.RandomCrop(augmentation["crop_size"], padding=augmentation["padding"]),
        transforms.RandomHorizontalFlip(augmentation["horizontal_flip"]),
        transforms.ToTensor(), normalization,
    ])
    testing.transform = transforms.Compose([transforms.ToTensor(), normalization])
    return training, testing


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluation_dataset(config, training, testing):
    if config["run_kind"] != "smoke-test":
        return testing
    evaluation = copy(training)
    evaluation.transform = testing.transform
    return Subset(evaluation, range(min(64, len(evaluation))))


def make_loader(dataset, batch_size, seed, workers, shuffle, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)
