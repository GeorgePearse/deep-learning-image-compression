# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

import argparse
import math
import random
import shutil
import sys

from collections import defaultdict
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from tinify.datasets import VideoFolder
from tinify.optimizers import net_aux_optimizer
from tinify.zoo import video_models


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target):
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)

            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()

        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        lambdas = torch.full_like(bpp_loss, self.lmbda)

        bpp_loss = bpp_loss.mean()
        out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss

        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        return out


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


def compute_aux_loss(aux_list: List, fabric=None, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True:
            if fabric is None:
                raise ValueError("Fabric instance required for backward pass")
            fabric.backward(aux_loss)

    return aux_loss_sum


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    fabric,
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
):
    model.train()

    for i, batch in enumerate(train_dataloader):
        d = batch
        # Ensure d is on the correct device.
        # Although fabric.setup_dataloaders handles this, for lists of tensors we double check.
        # If batch is a list of tensors, fabric.to_device handles it.
        # Note: Fabric dataloader might not recursively move list items if collate_fn returns a list.
        # So manual to_device is safer here.
        # d = fabric.to_device(d)
        # Actually, let's rely on the dataloader first, if it fails we add it.
        # But wait, previous code explicitly did: d = [frames.to(device) for frames in batch]
        # This means batch is a list of frames.
        # I will use fabric.to_device(d) which handles lists.

        # NOTE: If d is already on device, to_device checks and does nothing.
        # But wait, fabric.to_device might detach if not careful? No, it just moves.

        # However, if `batch` coming from dataloader is on CPU (because standard collate_fn for list of images isn't used/doesn't stack?),
        # then `fabric.setup_dataloaders` wraps it. The wrapper attempts to move data to device.
        # If the data structure is custom (list of tensors), the wrapper might not know how to move it unless it supports arbitrary structures.
        # Lightning Fabric's `_FabricDataLoader` generally tries to move batch to device.
        # It supports dicts, lists, tuples. So it should work.
        # But I'll add a safety measure or just assume it works.

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

        aux_loss = compute_aux_loss(model.aux_loss(), fabric=fabric, backward=True)
        aux_optimizer.step()

        if i % 10 == 0 and fabric.is_global_zero:
            print(
                f"\033[34mTrain epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100.0 * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {out_criterion['loss'].item():.3f} |"
                f"\tMSE loss: {out_criterion['mse_loss'].item():.3f} |"
                f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
                f"\tAux loss: {aux_loss.item():.2f}\033[0m"
            )


def test_epoch(fabric, epoch, test_dataloader, model, criterion):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            d = batch  # Assumed on device via Fabric wrapper
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    if fabric.is_global_zero:
        print(
            f"\033[95mTest epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n\033[0m"
        )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="ssf2020",
        choices=video_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
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
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    fabric = L.Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
    )
    fabric.launch()

    if args.seed is not None:
        fabric.seed_everything(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = VideoFolder(
        args.dataset,
        rnd_interval=True,
        rnd_temp_order=True,
        split="train",
        transform=train_transforms,
    )
    test_dataset = VideoFolder(
        args.dataset,
        rnd_interval=False,
        rnd_temp_order=False,
        split="test",
        transform=test_transforms,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    train_dataloader, test_dataloader = fabric.setup_dataloaders(
        train_dataloader, test_dataloader
    )

    net = video_models[args.model](quality=3)

    optimizer, aux_optimizer = configure_optimizers(net, args)

    # Setup with Fabric
    net, optimizer, aux_optimizer = fabric.setup(net, optimizer, aux_optimizer)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, return_details=True)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=fabric.device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if fabric.is_global_zero:
            print(f"\033[95mLearning rate: {optimizer.param_groups[0]['lr']}\033[0m")
        train_one_epoch(
            fabric,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(fabric, epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and fabric.is_global_zero:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
