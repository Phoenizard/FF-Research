import json
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive


class TinyImageNetValidation(Dataset):
    def __init__(self, root, class_to_idx):
        self.samples = []
        self.transform = None
        for line in (root / "val_annotations.txt").read_text().splitlines():
            filename, wnid, *_ = line.split()
            self.samples.append((root / "images" / filename, class_to_idx[wnid]))
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = pil_loader(path)
        return (self.transform(image) if self.transform else image), target


def tinyimagenet_datasets(config, root, download):
    directory = root / "tiny-imagenet-200"
    if download and not (directory / "val" / "val_annotations.txt").exists():
        download_and_extract_archive(
            "https://cs231n.stanford.edu/tiny-imagenet-200.zip", root,
            md5="90528d7ca1a48142e341f4ef8d21d0de")
    training = datasets.ImageFolder(directory / "train", loader=pil_loader)
    testing = TinyImageNetValidation(directory / "val", training.class_to_idx)
    if training.classes != sorted((directory / "wnids.txt").read_text().split()):
        raise ValueError("Tiny-ImageNet training classes do not match wnids.txt")
    if len(training.classes) != 200 or any(
            np.bincount(dataset.targets, minlength=200).tolist() != [count] * 200
            for dataset, count in ((training, 500), (testing, 50))):
        raise ValueError("Tiny-ImageNet requires 500 training and 50 evaluation images per class")
    if config.get("class_to_idx", training.class_to_idx) != training.class_to_idx:
        raise ValueError("Tiny-ImageNet class indices differ from the saved configuration")
    config["class_to_idx"] = training.class_to_idx.copy()
    if config["normalization"] is None:
        cache = directory / "normalization.json"
        if cache.exists():
            config["normalization"] = json.loads(cache.read_text())
        else:
            print("Computing Tiny-ImageNet normalization from 100,000 training images.", flush=True)
            total = np.zeros(3, dtype=np.float64)
            squared = np.zeros(3, dtype=np.float64)
            count = 0
            for path, _ in training.samples:
                pixels = np.asarray(pil_loader(path), dtype=np.float64).reshape(-1, 3) / 255
                total += pixels.sum(axis=0)
                squared += (pixels ** 2).sum(axis=0)
                count += len(pixels)
            mean = total / count
            config["normalization"] = {
                "mean": mean.tolist(), "std": np.sqrt(squared / count - mean ** 2).tolist()}
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(config["normalization"], indent=2, allow_nan=False) + "\n")
            temporary.replace(cache)
    return training, testing


def image_datasets(config, root=None, download=False):
    dataset = config["dataset"]
    root = Path(root) if root is not None else Path(__file__).resolve().parent / (
        "tinyimagenet" if dataset == "tinyimagenet" else "cifar")
    if dataset == "tinyimagenet":
        training, testing = tinyimagenet_datasets(config, root, download)
    else:
        dataset_type = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[dataset]
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
