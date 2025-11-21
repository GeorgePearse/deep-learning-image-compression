# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Unified training module for CompressAI models."""

from __future__ import annotations

import copy
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.optim as optim
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tinify.datasets import ImageFolder, VideoFolder
from tinify.losses import RateDistortionLoss, VideoRateDistortionLoss
from tinify.optimizers import net_aux_optimizer
from tinify.registry import MODELS
from tinify.zoo import image_models, video_models

from .config import Config


class AverageMeter:
    """Compute running average."""

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float | Tensor, n: int = 1) -> None:
        if isinstance(val, Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRFinder:
    """Learning Rate Finder."""

    fabric: Fabric
    model: torch.nn.Module
    optimizer: Optimizer
    criterion: torch.nn.Module
    dataloader: DataLoader[Any]

    def __init__(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        dataloader: DataLoader[Any],
    ) -> None:
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

    def range_test(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
    ) -> tuple[list[float], list[float]]:
        lrs = []
        losses = []
        best_loss = float("inf")
        avg_loss = 0.0

        # Save state
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())

        self.model.train()
        iter_loader = iter(self.dataloader)

        current_lr = start_lr
        lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)

        for i in range(num_iter):
            try:
                d = next(iter_loader)
            except StopIteration:
                iter_loader = iter(self.dataloader)
                d = next(iter_loader)

            # Update LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            self.optimizer.zero_grad()

            out_net = self.model(d)
            out_criterion = self.criterion(out_net, d)
            loss = out_criterion["loss"]

            self.fabric.backward(loss)
            self.optimizer.step()

            loss_val = loss.item()
            if i == 0:
                avg_loss = loss_val
            else:
                avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss

            if avg_loss < best_loss:
                best_loss = avg_loss

            if i > 0 and avg_loss > 4 * best_loss:
                break

            lrs.append(current_lr)
            losses.append(avg_loss)

            current_lr *= lr_multiplier

        # Restore state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)

        return lrs, losses


def get_model(config: Config) -> torch.nn.Module:
    """Get model instance from config."""
    model_name = config.model.name
    quality = config.model.quality
    kwargs = config.model.kwargs

    # Try zoo first (has pretrained weights)
    if config.domain == "image":
        if model_name in image_models:
            return image_models[model_name](
                quality=quality, pretrained=config.model.pretrained, **kwargs
            )
    elif config.domain == "video":
        if model_name in video_models:
            return video_models[model_name](
                quality=quality, pretrained=config.model.pretrained, **kwargs
            )

    # Fall back to registry
    if model_name in MODELS:
        return MODELS[model_name](**kwargs)

    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")


def get_dataset(config: Config, split: str) -> Dataset[Any]:
    """Get dataset instance from config."""
    patch_size = tuple(config.dataset.patch_size)

    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(patch_size),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
            ]
        )

    if config.domain == "image":
        return ImageFolder(
            config.dataset.path,
            split=split,
            transform=transform,
        )
    elif config.domain == "video":
        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(patch_size),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(patch_size),
                ]
            )
        return VideoFolder(
            config.dataset.path,
            rnd_interval=(split == "train"),
            rnd_temp_order=(split == "train"),
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported domain for dataset: {config.domain}")


def configure_optimizers(
    net: torch.nn.Module, config: Config
) -> tuple[Optimizer, Optimizer]:
    """Configure optimizers from config."""
    conf = {
        "net": {
            "type": config.optimizer_net.type,
            "lr": config.optimizer_net.lr,
        },
        "aux": {
            "type": config.optimizer_aux.type,
            "lr": config.optimizer_aux.lr,
        },
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_dataloader: DataLoader[Any],
    optimizer: Optimizer,
    aux_optimizer: Optimizer,
    epoch: int,
    config: Config,
    scheduler: LRScheduler | None = None,
) -> None:
    """Train for one epoch."""
    model.train()
    domain = config.domain

    for i, d in enumerate(train_dataloader):
        # Fabric handles device placement if setup_dataloaders is used.

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        fabric.backward(out_criterion["loss"])

        if config.training.clip_max_norm > 0:
            fabric.clip_gradients(
                model,
                optimizer,
                max_norm=config.training.clip_max_norm,
                error_if_nonfinite=False,
            )

        optimizer.step()

        # Step scheduler if provided and not plateau
        if scheduler is not None and not isinstance(
            scheduler, optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        aux_loss = model.aux_loss()
        if isinstance(aux_loss, list):
            aux_loss = sum(aux_loss)

        fabric.backward(aux_loss)
        aux_optimizer.step()

        if i % config.training.log_interval == 0 and fabric.is_global_zero:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d) if domain != 'video' else i}/{len(train_dataloader.dataset)}"
                f" ({100.0 * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {out_criterion['loss'].item():.3f} |"
                f"\tMSE loss: {out_criterion.get('mse_loss', out_criterion.get('ms_ssim_loss', 0)):.5f} |"
                f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(
    fabric: Fabric,
    epoch: int,
    test_dataloader: DataLoader[Any],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    config: Config,
) -> float:
    """Evaluate for one epoch."""
    model.eval()
    _ = config.domain  # Used for potential domain-specific logging

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux = model.aux_loss()
            if isinstance(aux, list):
                aux = sum(aux)

            aux_loss.update(aux)
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(
                out_criterion.get("mse_loss", out_criterion.get("ms_ssim_loss", 0))
            )

    if fabric.is_global_zero:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    save_dir: str,
    filename: str = "checkpoint.pth.tar",
) -> None:
    """Save checkpoint to disk."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, save_path / "checkpoint_best_loss.pth.tar")


def train(config: Config) -> None:
    """Main training function."""
    # Setup Fabric
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )
    fabric.launch()

    # Set seed for reproducibility
    if config.training.seed is not None:
        fabric.seed_everything(config.training.seed)

    # Create datasets
    if fabric.is_global_zero:
        print(f"Loading dataset from: {config.dataset.path}")

    train_dataset = get_dataset(config, config.dataset.split_train)
    test_dataset = get_dataset(config, config.dataset.split_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.training.test_batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    train_dataloader, test_dataloader = fabric.setup_dataloaders(
        train_dataloader, test_dataloader
    )

    # Create model
    if fabric.is_global_zero:
        print(f"Creating model: {config.model.name} (quality={config.model.quality})")

    net = get_model(config)

    # Setup optimizers and scheduler
    optimizer, aux_optimizer = configure_optimizers(net, config)

    # Setup model and optimizers with Fabric
    net, optimizer, aux_optimizer = fabric.setup(net, optimizer, aux_optimizer)

    # Setup loss
    if config.domain == "video":
        criterion = VideoRateDistortionLoss(lmbda=config.training.lmbda)
    else:
        criterion = RateDistortionLoss(
            lmbda=config.training.lmbda, metric=config.training.metric
        )

    lr_scheduler = None

    # Load checkpoint if resuming
    last_epoch = 0
    best_loss = float("inf")

    if config.training.checkpoint:
        if fabric.is_global_zero:
            print(f"Loading checkpoint: {config.training.checkpoint}")

        # Default scheduler for resumption if not cyclic
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.mode,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )

        checkpoint = torch.load(config.training.checkpoint, map_location=fabric.device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    else:
        # Run LR Finder and setup CyclicLR
        if fabric.is_global_zero:
            print("\nRunning LR Finder to determine cyclic LR limits...")

        lr_finder = LRFinder(fabric, net, optimizer, criterion, train_dataloader)
        lrs, losses = lr_finder.range_test()

        best_lr = 1e-4
        if fabric.is_global_zero:
            min_loss_idx = losses.index(min(losses))
            best_lr = lrs[min_loss_idx]
            print(f"LR Finder: Best LR found = {best_lr:.6f}")

            try:
                import uniplot

                print("\nLR Finder Results:")
                uniplot.plot(
                    losses,
                    xs=lrs,
                    title="Loss vs Learning Rate (Log Scale)",
                    x_as_log=True,
                )
            except ImportError:
                print("uniplot not installed, skipping chart.")

        # Broadcast best_lr to all ranks
        best_lr_tensor = torch.tensor(best_lr, device=fabric.device)
        best_lr = fabric.broadcast(best_lr_tensor).item()

        max_lr = best_lr
        base_lr = max_lr / 6.0

        if fabric.is_global_zero:
            print(f"Setting CyclicLR: base_lr={base_lr:.6f}, max_lr={max_lr:.6f}")

        # Update optimizer to base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr

        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=len(train_dataloader) * 4,  # 4 epochs
            mode="triangular2",
            cycle_momentum=False,
        )

    # Training loop
    if fabric.is_global_zero:
        print(f"\nStarting training for {config.training.epochs} epochs...")
        print(f"Lambda: {config.training.lmbda}, Metric: {config.training.metric}")
        print("-" * 80)

    for epoch in range(last_epoch, config.training.epochs):
        if fabric.is_global_zero:
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_one_epoch(
            fabric,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            config,
            lr_scheduler,
        )

        loss = test_epoch(fabric, epoch, test_dataloader, net, criterion, config)

        # Step scheduler (plateau)
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if config.training.save and fabric.is_global_zero:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": config.to_dict(),
                },
                is_best,
                config.training.save_dir,
            )

    if fabric.is_global_zero:
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Checkpoints saved to: {config.training.save_dir}")


def list_models(domain: str | None = None) -> None:
    """List available models."""
    print("\nAvailable Models")
    print("=" * 60)

    if domain is None or domain == "image":
        print("\nImage Compression Models:")
        print("-" * 40)
        for name in sorted(image_models.keys()):
            print(f"  {name}")

    if domain is None or domain == "video":
        print("\nVideo Compression Models:")
        print("-" * 40)
        for name in sorted(video_models.keys()):
            print(f"  {name}")

    if domain is None or domain == "pointcloud":
        print("\nPoint Cloud Compression Models:")
        print("-" * 40)
        pcc_models = [k for k in MODELS.keys() if "pcc" in k.lower()]
        for name in sorted(pcc_models):
            print(f"  {name}")

    print("\nAll Registered Models:")
    print("-" * 40)
    for name in sorted(MODELS.keys()):
        print(f"  {name}")
