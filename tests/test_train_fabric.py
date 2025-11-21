# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MockCIFAR10(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.transform = transform
        # Create a few fake images
        self.length = 16
        self.images = [
            Image.new("RGB", (32, 32), color=(i * 10, i * 10, i * 10))
            for i in range(self.length)
        ]
        self.targets = [i % 10 for i in range(self.length)]

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


def test_train_elic_cifar10_fabric(tmp_path):
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent
    script_path = rootdir / "examples/train_elic_cifar10.py"

    module = load_module("examples.train_elic_cifar10", script_path)

    argv = [
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--test-batch-size",
        "4",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--save",
        "--N",
        "32",  # Smaller model for speed
        "--M",
        "128",
    ]

    # Mock CIFAR10 to return our small dataset
    with patch("torchvision.datasets.CIFAR10", side_effect=MockCIFAR10):
        # Run in tmp_path to avoid clutter
        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            module.main(argv)
        finally:
            os.chdir(original_cwd)

    # Check if checkpoint was saved
    assert (tmp_path / "elic_cifar10_best.pth.tar").exists()


def test_train_video_fabric(tmp_path):
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent
    script_path = rootdir / "examples/train_video.py"

    module = load_module("examples.train_video", script_path)

    # Setup fake video dataset
    dataset_dir = tmp_path / "video_dataset"
    dataset_dir.mkdir()

    # Create sequences dir
    seq_dir = dataset_dir / "sequences"
    seq_dir.mkdir()

    # Create a fake video sequence
    vid_name = "vid0"
    vid_dir = seq_dir / vid_name
    vid_dir.mkdir()

    # VideoFolder expects at least some frames.
    for i in range(3):
        img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
        img.save(vid_dir / f"frame_{i:03d}.png")

    # Create lists
    with open(dataset_dir / "train.list", "w") as f:
        f.write(f"{vid_name}\n")

    with open(dataset_dir / "test.list", "w") as f:
        f.write(f"{vid_name}\n")

    argv = [
        "-d",
        str(dataset_dir),
        "-e",
        "1",
        "--batch-size",
        "1",
        "--test-batch-size",
        "1",
        "--patch-size",
        "128",
        "128",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--save",
    ]

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        module.main(argv)
    finally:
        os.chdir(original_cwd)

    # Checkpoint check
    assert (tmp_path / "checkpoint.pth.tar").exists()


def test_train_image_fabric(tmp_path):
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent
    script_path = rootdir / "examples/train.py"

    module = load_module("examples.train_image", script_path)

    # Use existing fake data
    dataset_path = rootdir / "tests/assets/fakedata/imagefolder"

    argv = [
        "-d",
        str(dataset_path),
        "-e",
        "1",
        "--batch-size",
        "2",
        "--patch-size",
        "48",
        "48",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--save",
    ]

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        module.main(argv)
    finally:
        os.chdir(original_cwd)

    assert (tmp_path / "checkpoint.pth.tar").exists()
