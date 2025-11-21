# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from pytorch_msssim import ms_ssim

from tinify.registry import register_criterion


def collect_likelihoods_list(
    likelihoods_list: list[dict[str, dict[str, Tensor]]], num_pixels: int
) -> tuple[Tensor, dict[str, Tensor]]:
    bpp_info_dict: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0))
    bpp_loss: Tensor = torch.tensor(0.0)

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp: Tensor = torch.tensor(0.0)
        for label, likelihoods in frame_likelihoods.items():
            label_bpp: Tensor = torch.tensor(0.0)
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss = bpp_loss + bpp
                frame_bpp = frame_bpp + bpp
                label_bpp = label_bpp + bpp

                bpp_info_dict[f"bpp_loss.{label}"] = (
                    bpp_info_dict[f"bpp_loss.{label}"] + bpp.sum()
                )
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    lmbda: float
    return_type: str
    metric: nn.Module | Callable[..., Tensor]

    def __init__(
        self, lmbda: float = 0.01, metric: str = "mse", return_type: str = "all"
    ) -> None:
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(
        self, output: dict[str, Any], target: Tensor
    ) -> dict[str, Tensor] | Tensor:
        N, _, H, W = target.size()
        out: dict[str, Tensor] = {}
        num_pixels = N * H * W

        bpp_sum: Tensor = torch.tensor(0.0, device=target.device)
        for likelihoods in output["likelihoods"].values():
            bpp_sum = bpp_sum + torch.log(likelihoods).sum() / (
                -math.log(2) * num_pixels
            )
        out["bpp_loss"] = bpp_sum
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


@register_criterion("VideoRateDistortionLoss")
class VideoRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter for video."""

    mse: nn.MSELoss
    lmbda: float
    _scaling_functions: Callable[[Tensor], Tensor]
    return_details: bool

    def __init__(
        self, lmbda: float = 1e-2, return_details: bool = False, bitdepth: int = 8
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    def _get_scaled_distortion(
        self, x: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        x_chunks: tuple[Tensor, ...]
        target_chunks: tuple[Tensor, ...]

        if isinstance(x, torch.Tensor):
            x_chunks = x.chunk(x.size(1), dim=1)
        else:
            x_chunks = tuple(x)

        if isinstance(target, torch.Tensor):
            target_chunks = target.chunk(target.size(1), dim=1)
        else:
            target_chunks = tuple(target)

        # compute metric over each component (eg: y, u and v)
        metric_values_list: list[Tensor] = []
        for x0, x1 in zip(x_chunks, target_chunks):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values_list.append(v)
        metric_values = torch.stack(metric_values_list)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x: Tensor | tuple[Tensor, ...] | list[Tensor]) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst: list[Tensor] | tuple[Tensor, ...]) -> None:
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(
        self, output: dict[str, Any], target: list[Tensor]
    ) -> dict[str, Tensor]:
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out: dict[str, Tensor] = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions: list[Tensor] = []
        distortions: list[Tensor] = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)

            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()

        # average scaled_distortions across the frames
        avg_scaled_distortions = torch.stack(scaled_distortions).sum(dim=0) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        lambdas = torch.full_like(bpp_loss, self.lmbda)

        bpp_loss_mean = bpp_loss.mean()
        out["loss"] = (lambdas * avg_scaled_distortions).mean() + bpp_loss_mean

        out["distortion"] = avg_scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss_mean
        return out
