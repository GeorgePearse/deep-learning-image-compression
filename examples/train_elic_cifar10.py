# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Train ELIC on CIFAR-10 dataset.

Example usage:
    python examples/train_elic_cifar10.py --epochs 50 --accelerator auto
"""

import argparse
import random
import sys

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tinify.losses import RateDistortionLoss
from tinify.optimizers import net_aux_optimizer
from tinify.registry import MODELS


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    fabric,
    model,
    criterion,
    dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
):
    model.train()

    for i, (d, _) in enumerate(dataloader):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        fabric.backward(out_criterion["loss"])

        if clip_max_norm > 0:
            fabric.clip_gradients(
                model, optimizer, max_norm=clip_max_norm, error_if_nonfinite=False
            )

        optimizer.step()

        aux_loss = model.aux_loss()
        fabric.backward(aux_loss)
        aux_optimizer.step()

        if i % 100 == 0 and fabric.is_global_zero:
            print(
                f"\033[34mTrain epoch {epoch}: [{i * len(d)}/{len(dataloader.dataset)}"
                f" ({100.0 * i / len(dataloader):.0f}%)]\033[0m"
                f"\t\033[34mLoss:\033[0m {out_criterion['loss'].item():.3f} |"
                f"\t\033[34mMSE:\033[0m {out_criterion['mse_loss'].item():.5f} |"
                f"\t\033[34mBpp:\033[0m {out_criterion['bpp_loss'].item():.2f}"
            )


def test_epoch(fabric, epoch, dataloader, model, criterion):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d, _ in dataloader:
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    if fabric.is_global_zero:
        print(
            f"\033[95mTest epoch\033[0m {epoch}: "
            f"\033[95mLoss:\033[0m {loss.avg:.3f} | "
            f"\033[95mMSE:\033[0m {mse_loss.avg:.5f} | "
            f"\033[95mBpp:\033[0m {bpp_loss.avg:.2f}\n"
        )
    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train ELIC on CIFAR-10.")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=64, help="Test batch size"
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.01,
        help="Rate-distortion tradeoff",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--clip-max-norm", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--save", action="store_true", help="Save checkpoint")
    parser.add_argument(
        "--N", type=int, default=64, help="Network width (default: 64 for small images)"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=128,
        help="Latent channels (default: 128 for small images)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator (default: %(default)s)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Precision (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    fabric = L.Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
    )
    fabric.launch()
    fabric.seed_everything(args.seed)

    # CIFAR-10: 32x32 -> resize to 64x64 for ELIC (needs 16x downsampling)
    train_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
        ]
    )

    if fabric.is_global_zero:
        print("Downloading CIFAR-10...")

    # Download on rank 0 only usually, but torchvision datasets handle checks.
    # Fabric doesn't have a specific tool for this, but standard practice is ok.
    # If running multi-node, might need care. For single node multi-gpu it's fine.

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Use ELIC Chandelier (lighter variant) with reduced channels for small images
    if fabric.is_global_zero:
        print(f"Creating ELIC model (N={args.N}, M={args.M})...")

    # Adjust groups for smaller M
    groups = [8, 8, 16, 32, args.M - 64]  # Scaled down from [16, 16, 32, 64, M-128]

    net = MODELS["elic2022-chandelier"](N=args.N, M=args.M, groups=groups)

    # Configure optimizers
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": 1e-3},
    }
    optimizers = net_aux_optimizer(net, conf)
    optimizer = optimizers["net"]
    aux_optimizer = optimizers["aux"]

    # Setup with Fabric
    net, optimizer, aux_optimizer = fabric.setup(net, optimizer, aux_optimizer)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    if fabric.is_global_zero:
        print(f"\nTraining ELIC on CIFAR-10 for {args.epochs} epochs")
        print(f"Lambda: {args.lmbda}, LR: {args.learning_rate}")
        print("-" * 60)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        if fabric.is_global_zero:
            print(f"\033[95mLR:\033[0m {optimizer.param_groups[0]['lr']:.6f}")

        train_one_epoch(
            fabric,
            net,
            criterion,
            train_loader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(fabric, epoch, test_loader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and is_best and fabric.is_global_zero:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                },
                "elic_cifar10_best.pth.tar",
            )
            print(f"Saved best checkpoint (loss: {loss:.4f})")

    if fabric.is_global_zero:
        print(f"\nDone! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
