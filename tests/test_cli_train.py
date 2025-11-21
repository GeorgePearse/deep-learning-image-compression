# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

import tinify.cli


def test_cli_train_image_fabric(tmp_path):
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    # Use existing fake data
    dataset_path = rootdir / "tests/assets/fakedata/imagefolder"

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write("""
        model:
          name: bmshj2018-factorized
          quality: 1
        dataset:
          patch_size: [48, 48]
        training:
          clip_max_norm: 1.0
        """)

    argv = [
        "train",
        "image",
        "-c",
        str(config_path),
        "-d",
        str(dataset_path),
        "-e",
        "1",
        "--batch-size",
        "2",
    ]

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # main returns 0 on success
        assert tinify.cli.main(argv) == 0
    finally:
        os.chdir(original_cwd)

    assert (tmp_path / "checkpoints" / "checkpoint.pth.tar").exists()


def test_cli_train_video_fabric(tmp_path):
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    # Setup fake video dataset
    dataset_dir = tmp_path / "video_dataset"
    dataset_dir.mkdir()

    seq_dir = dataset_dir / "sequences"
    seq_dir.mkdir()

    vid_name = "vid0"
    vid_dir = seq_dir / vid_name
    vid_dir.mkdir()

    for i in range(3):
        img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
        img.save(vid_dir / f"frame_{i:03d}.png")

    with open(dataset_dir / "train.list", "w") as f:
        f.write(f"{vid_name}\n")

    with open(dataset_dir / "test.list", "w") as f:
        f.write(f"{vid_name}\n")

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write("""
        model:
          name: ssf2020
          quality: 1
        dataset:
          patch_size: [128, 128]
        training:
          clip_max_norm: 1.0
        """)

    argv = [
        "train",
        "video",
        "-c",
        str(config_path),
        "-d",
        str(dataset_dir),
        "-e",
        "1",
        "--batch-size",
        "1",
    ]

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert tinify.cli.main(argv) == 0
    finally:
        os.chdir(original_cwd)

    assert (tmp_path / "checkpoints" / "checkpoint.pth.tar").exists()
