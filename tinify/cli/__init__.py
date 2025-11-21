# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Tinify Command Line Interface.

Usage:
    tinify train image --config config.yaml
    tinify train video --config config.yaml
    tinify train list-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .train import list_models, train


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinify",
        description="Tinify: End-to-end learned compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  tinify train image --config configs/mbt2018-mean.yaml

  # Train with CLI arguments
  tinify train image -m mbt2018-mean -d /path/to/dataset --epochs 100

  # List available models
  tinify train list-models --domain image

  # Resume from checkpoint
  tinify train image --config config.yaml --checkpoint checkpoint.pth.tar
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train compression models")
    train_subparsers = train_parser.add_subparsers(
        dest="domain", help="Compression domain"
    )

    # Image training
    image_parser = train_subparsers.add_parser(
        "image",
        help="Train image compression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_train_args(image_parser)

    # Video training
    video_parser = train_subparsers.add_parser(
        "video",
        help="Train video compression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_train_args(video_parser)

    # Point cloud training
    pointcloud_parser = train_subparsers.add_parser(
        "pointcloud",
        help="Train point cloud compression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_train_args(pointcloud_parser)

    # List models command
    list_parser = train_subparsers.add_parser(
        "list-models",
        help="List available models",
    )
    list_parser.add_argument(
        "--domain",
        choices=["image", "video", "pointcloud"],
        help="Filter by domain",
    )

    return parser


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments to parser."""
    # Config file
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file (YAML/JSON/TOML)",
    )

    # Model
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model architecture name",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        help="Quality level (1-8)",
    )

    # Dataset
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to training dataset",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        help="Size of the patches to be cropped (e.g. 256 256)",
    )

    # Training
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Training batch size",
    )
    parser.add_argument(
        "--lambda",
        "--lmbda",
        type=float,
        dest="lmbda",
        help="Rate-distortion trade-off parameter",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Learning rate for main optimizer",
    )

    # Device
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=None,
        help="Use CUDA",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_false",
        dest="cuda",
        help="Disable CUDA",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        dest="save_dir",
        help="Directory to save checkpoints",
    )


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 1

    if parsed_args.command == "train":
        if parsed_args.domain is None:
            parser.parse_args(["train", "--help"])
            return 1

        if parsed_args.domain == "list-models":
            list_models(getattr(parsed_args, "domain", None))
            return 0

        # Load config
        if hasattr(parsed_args, "config") and parsed_args.config:
            config = Config.from_file(parsed_args.config)
        else:
            config = Config()

        # Set domain from subcommand
        config.domain = parsed_args.domain

        # Merge CLI arguments
        config.merge_cli_args(parsed_args)

        # Validate required arguments
        if not config.dataset.path:
            print(
                "Error: Dataset path is required. Use -d/--dataset or specify in config."
            )
            return 1

        # Run training
        train(config)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
